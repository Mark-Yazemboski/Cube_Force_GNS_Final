"""
force_gns.py  (v2 - COM fluid head)

NEW FILE - does not modify any existing code.

The force-based version of the GNS model plus the rigid-body dynamics layer
that turns forces into motion. Training supervision is unchanged (position
error only) - the model is never shown a force label.

WHAT THE MODEL OUTPUTS (the v2 change):
  - CONTACT: per-node forces, gated by a geometric contact indicator. Only
    nodes near the floor can exert contact force. Tangential + softplus normal
    (normal can push, never pull -> non-penetration is architectural).
  - FLUID: ONE body-level wrench (force + torque at the COM), read out from
    the pooled node latents. NOT per-node.

WHY THE FLUID FORCE IS A COM WRENCH, NOT PER-NODE (design decision with the
advisors, replacing the proposal's per-node complementarity for this system):
  1. At N=8 corner nodes, hard complementarity pins the fluid resultant to the
     wrong height (top nodes only, when the cube sits on the floor), creating
     a fictitious pitching torque ~F*s/2 that the contact branch must absorb
     with compensating fictitious forces. The trajectory still fits; the force
     DECOMPOSITION - the thing this architecture exists to recover - gets
     structurally corrupted.
  2. MuJoCo's fluid model is itself a body-level wrench with no occlusion, so
     a COM head matches the data-generating process exactly, and the logged
     wrench labels compare against it one-to-one.
  3. From rigid-body motion, a per-node fluid field is only identifiable
     through its net 6-DOF wrench anyway; the COM head predicts exactly the
     identifiable quantity. Contact keeps the distributed representation
     because contact distribution IS partially identifiable and structured.
  On a dense mesh (aerial manipulator) with real occlusion, per-node fluid
  returns and this head becomes its pooled limit.

Built in as exact physics (not learned): gravity; the cube's mass and inertia
(I = (1/6) m s^2 * Identity - ISOTROPIC, so the gyroscopic term w x Iw is
exactly zero and alpha = tau/I is the full Euler equation); exact rigidity
(state is COM + rotation, so the mesh cannot deform - shape matching is
deleted, not approximated); optionally the analytic quadratic drag baseline at
the COM with the calibrated k/m, so the fluid head only learns the residual
(its final layer is zero-initialized: the model STARTS as contact + gravity +
analytic drag).

UNITS (matches the existing pipeline): positions in meters, one "step" is one
recorded frame (DT seconds). Per-node contact outputs are SPECIFIC forces
phi_i = f_i * dt^2 / m (m/step^2, same units as the old acceleration targets).
The fluid head outputs a COM acceleration (m/step^2) and an angular
acceleration (rad/step^2) directly. In these units:
    a_com = sum_i phi_i + g dt^2 + a_drag + a_fluid
    alpha = sum_i r_i x phi_i / (I/m) + alpha_fluid
Mass and dt cancel out of the contact-torque term; only the radius of
gyration (I/m = s^2/6) enters.
"""

import math
import torch
import torch.nn as nn

# Same geometry constants as the rest of the codebase
BLOCK_HALF_WIDTH = 0.0524
BLOCK_WIDTH = 2.0 * BLOCK_HALF_WIDTH

# Solid cube inertia over mass: I/m = s^2 / 6, identical about every axis.
# Isotropic inertia => w x (I w) = I_scalar * (w x w) = 0 exactly.
I_OVER_M = (BLOCK_WIDTH ** 2) / 6.0


# ======================================================================
# Rotation utilities (batched, differentiable, small-angle safe)
# ======================================================================

def quat_wxyz_to_R(q):
    """Batched scalar-first quaternion -> rotation matrix.  q: (..., 4) -> (..., 3, 3)."""
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(-1)
    R = torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],     dim=-1),
        torch.stack([2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],     dim=-1),
        torch.stack([2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)], dim=-1),
    ], dim=-2)
    return R


def _hat(w):
    """Rotation vector -> skew-symmetric matrix.  w: (..., 3) -> (..., 3, 3)."""
    zeros = torch.zeros_like(w[..., 0])
    wx, wy, wz = w.unbind(-1)
    return torch.stack([
        torch.stack([zeros, -wz,    wy], dim=-1),
        torch.stack([wz,    zeros, -wx], dim=-1),
        torch.stack([-wy,   wx,    zeros], dim=-1),
    ], dim=-2)


def so3_exp(w):
    """Rodrigues' formula, safe near theta -> 0 (Taylor series for the sinc terms).
    w: (..., 3) rotation vector -> (..., 3, 3) rotation matrix."""
    theta = w.norm(dim=-1, keepdim=True).unsqueeze(-1)          # (..., 1, 1)
    K = _hat(w)
    small = theta < 1e-4
    theta_safe = theta.clamp_min(1e-12)
    A = torch.where(small, 1.0 - theta ** 2 / 6.0, torch.sin(theta_safe) / theta_safe)
    B = torch.where(small, 0.5 - theta ** 2 / 24.0, (1.0 - torch.cos(theta_safe)) / theta_safe ** 2)
    eye = torch.eye(3, device=w.device, dtype=w.dtype).expand(K.shape)
    return eye + A * K + B * (K @ K)


def so3_log(R):
    """Rotation matrix -> rotation vector, safe near theta -> 0. Per-step
    rotations in this project are tiny (~0.04 rad), so the theta -> pi branch
    never occurs mid-rollout; still clamped for safety.
    R: (..., 3, 3) -> (..., 3)."""
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_theta = ((tr - 1.0) * 0.5).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    theta = torch.acos(cos_theta).unsqueeze(-1)                  # (..., 1)
    vee = torch.stack([
        R[..., 2, 1] - R[..., 1, 2],
        R[..., 0, 2] - R[..., 2, 0],
        R[..., 1, 0] - R[..., 0, 1],
    ], dim=-1) * 0.5
    small = theta < 1e-4
    sin_safe = torch.sin(theta).clamp_min(1e-12)
    factor = torch.where(small, 1.0 + theta ** 2 / 6.0, theta / sin_safe)
    return factor * vee


# ======================================================================
# GNS encoder / processor - mirrors GNSLayer in train_gnn_multi_step.py.
# Copied (not imported) so this module stays standalone and unit-testable
# without pulling in torch_geometric. The math is identical.
# ======================================================================

class GNSLayer(nn.Module):

    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim + node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim)
        )
        self.edge_norm = nn.LayerNorm(hidden_dim)
        self.node_norm = nn.LayerNorm(node_dim)

    def forward(self, x, edge_index, edge_attr):
        senders, receivers = edge_index[0], edge_index[1]
        edge_input = torch.cat([x[senders], x[receivers], edge_attr], dim=-1)
        edge_attr = edge_attr + self.edge_norm(self.edge_mlp(edge_input))
        node_agg = torch.zeros(x.size(0), edge_attr.size(1), device=x.device, dtype=x.dtype)
        node_agg.index_add_(0, receivers, edge_attr)
        x = x + self.node_norm(self.node_mlp(torch.cat([x, node_agg], dim=-1)))
        return x, edge_attr


class ForceGNSModel(nn.Module):
    """
    Encoder and processor are identical to the existing GNSModel. Two heads:

    CONTACT head (per node, 4 raw numbers):
        [0:3] tangential force raw (the wall-normal component is projected out
              downstream, so only the in-plane part survives)
        [3]   normal force magnitude raw -> softplus (can push, never pull)

    FLUID head (per graph, 6 raw numbers, from MEAN-POOLED node latents):
        [0:3] COM force  -> a_fluid (m/step^2)
        [3:6] COM torque -> alpha_fluid (rad/step^2)
        Final layer ZERO-INITIALIZED: with the analytic drag baseline on, the
        model starts exactly as "contact + gravity + calibrated drag" and the
        head learns only the residual (PIROM practice).

    forward(x, edge_index, edge_attr, num_graphs) with plain tensors (no PyG):
        returns (contact_raw (B, N, 4), fluid_raw (B, 6)).
    """

    N_CONTACT_OUT = 4
    N_FLUID_OUT = 6

    def __init__(self, node_in_dim, edge_in_dim, latent_dim=128, L=5, K=1,
                 normal_bias_init=-2.0):
        super().__init__()
        self.K = K
        self.L = L
        self.node_encoder = nn.Sequential(
            nn.Linear(node_in_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_in_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        self.processor_layers = nn.ModuleList([
            GNSLayer(latent_dim, latent_dim, latent_dim) for _ in range(L)
        ])
        self.decoder_contact = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, self.N_CONTACT_OUT)
        )
        self.fluid_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, self.N_FLUID_OUT)
        )
        with torch.no_grad():
            # Normal-force channel starts small: softplus(-2) ~ 0.127, so four
            # resting contact nodes supply roughly the cube's weight at init
            # instead of several times it.
            self.decoder_contact[-1].bias[3] = normal_bias_init
            # Fluid head starts at exactly zero (baseline carries at init).
            self.fluid_head[-1].weight.zero_()
            self.fluid_head[-1].bias.zero_()

    def forward(self, x, edge_index, edge_attr, num_graphs):
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)
        for _ in range(self.K):
            for layer in self.processor_layers:
                x, edge_attr = layer(x, edge_index, edge_attr)
        N = x.shape[0] // num_graphs
        x_g = x.reshape(num_graphs, N, -1)
        contact_raw = self.decoder_contact(x_g)                  # (B, N, 4)
        fluid_raw = self.fluid_head(x_g.mean(dim=1))             # (B, 6)
        return contact_raw, fluid_raw


# ======================================================================
# Output assembly
# ======================================================================

def contact_weight(dist, d0=0.02, tau=0.005):
    """Soft geometric contact indicator (1 = contact, 0 = free). Not learned:
    contact detection for a cube on a flat floor is trivial geometry, and
    fixing it removes a failure mode while the force representation is
    validated. dist: (..., 1) signed node distance from the wall along its
    normal. d0 is generous on purpose - MuJoCo's solref compliance means
    contact forces act slightly before geometric touchdown. Soft (not hard) so
    forces cannot pop discontinuously as a node crosses the boundary."""
    return torch.sigmoid((d0 - dist) / tau)


def assemble_contact_forces(contact_raw, c_w, wall_normal, scale_vec):
    """
    contact_raw: (B, N, 4) contact-head output
    c_w:         (B, N, 1) contact weight in [0, 1]
    wall_normal: (3,) wall normal (normalized here)
    scale_vec:   (3,) output scale in m/step^2, [s_xy, s_xy, s_z] from the
                 acceleration-target stats (equal x/y keeps the scaling
                 z-rotation equivariant, so the rotation augmentation is valid)

    Returns phi_c (B, N, 3) per-node contact specific force, and a parts dict.
    """
    n_hat = wall_normal / wall_normal.norm().clamp_min(1e-12)
    t_raw = contact_raw[..., 0:3]
    n_raw = contact_raw[..., 3:4]

    # Scale BEFORE projecting. scale_vec is an elementwise world-frame scale,
    # and elementwise scaling does not commute with projection unless the wall
    # normal lies along a coordinate axis: with s = [s_xy, s_xy, s_z] and a
    # tilted n_hat, sum_k t_k n_k = 0 does NOT imply sum_k s_k t_k n_k = 0, so
    # the tangential head could inject a normal component and defeat the
    # softplus. Scaling first makes the projection exact for ANY wall normal.
    # Bit-identical to the previous order for axis-aligned walls (floor
    # (0,0,1), vertical inspection wall (1,0,0)); differs only on a ramp.
    t_scaled = t_raw * scale_vec
    t_vec = t_scaled - (t_scaled * n_hat).sum(-1, keepdim=True) * n_hat

    # Output scale along the wall normal. Reduces to scale_vec[2] for a floor.
    s_n = (scale_vec * n_hat).norm()
    n_mag = torch.nn.functional.softplus(n_raw) * s_n

    phi_c = c_w * (t_vec + n_mag * n_hat)
    parts = {"phi_contact": phi_c,
             "normal_mag": n_mag,
             "contact_weight": c_w}
    return phi_c, parts


def fluid_wrench_from_raw(fluid_raw, scale_vec, ang_scale_vec):
    """fluid_raw (B, 6) -> (a_fluid (B, 3) in m/step^2,
                            alpha_fluid (B, 3) in rad/step^2).
    ang_scale_vec is the empirical angular-acceleration std [a_xy, a_xy, a_z],
    the rotational analog of the existing acceleration normalization."""
    return fluid_raw[:, 0:3] * scale_vec, fluid_raw[:, 3:6] * ang_scale_vec


def drag_accel_step(wind, v_com_step, dt, k_over_m):
    """Analytic quadratic body-drag baseline as a per-step^2 COM acceleration.
    Uses the k/m coefficient CALIBRATED FROM DATA (wind_error_analysis.py).
    Applied at the COM => zero torque, exactly like MuJoCo.
    wind: (B, 3) m/s;  v_com_step: (B, 3) m/step (= com_curr - com_prev)."""
    u = wind - v_com_step / dt
    return k_over_m * u.norm(dim=-1, keepdim=True) * u * (dt * dt)


# ======================================================================
# Rigid-body dynamics layer (Verlet on COM, Lie-group Verlet on rotation)
# ======================================================================

def nodes_from_state(com, R, rest_nodes):
    """World node positions from rigid state.
    com: (B, 3)   R: (B, 3, 3)   rest_nodes: (N, 3)  ->  (B, N, 3)"""
    return com.unsqueeze(1) + torch.einsum('bij,nj->bni', R, rest_nodes)


def rigid_step(phi_contact, com_prev, com_curr, R_prev, R_curr, rest_nodes,
               g_step, extra_accel=None, extra_alpha=None):
    """
    One Newton-Euler step in per-step units. Differentiable end to end; no SVD
    anywhere (SVD gradients are ill-conditioned for a cube's isotropic corner
    set, which is why (COM, R) is carried as explicit state instead of fitting
    the rotation from node positions each step).

    phi_contact: (B, N, 3) per-node contact specific forces (m/step^2)
    extra_accel: optional (B, 3) COM acceleration (m/step^2) - analytic drag
                 plus the learned fluid force live here (both act at the COM,
                 so neither contributes torque)
    extra_alpha: optional (B, 3) angular acceleration (rad/step^2) - the
                 learned fluid torque lives here
    g_step:      (3,) gravity as m/step^2, i.e. [0, 0, -g] * dt^2

    Returns (com_next, R_next).
    """
    a_com = phi_contact.sum(dim=1) + g_step
    if extra_accel is not None:
        a_com = a_com + extra_accel
    com_next = 2.0 * com_curr - com_prev + a_com

    # Contact torque from lever arms; fluid torque arrives via extra_alpha.
    r = torch.einsum('bij,nj->bni', R_curr, rest_nodes)          # (B, N, 3)
    alpha = torch.cross(r, phi_contact, dim=-1).sum(dim=1) / I_OVER_M
    if extra_alpha is not None:
        alpha = alpha + extra_alpha

    w_prev = so3_log(R_curr @ R_prev.transpose(-1, -2))          # (B, 3) rad/step
    R_next = so3_exp(w_prev + alpha) @ R_curr
    return com_next, R_next
