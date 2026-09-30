import torch
from generate_node_states import BLOCK_HALF_WIDTH, BLOCK_WIDTH

#This file is used to compute all of the different metrics that are used to compare the truth cube toss, to the
# predicted cube toss from the GNN. The metrics we compute are:
# 1. Relative center position error: the distance between the predicted and true center of mass, normalized
#    by the block width.
# 2. Absolute angle difference: the angle difference between the predicted and true rotation matrices,
#    converted to degrees.
# 3. Floor penetration: the maximum depth that any part of the cube goes below the
#    floor (z=0), normalized by the block width.

# --- thresholds for phase segmentation ---
CONTACT_Z_THRESH    = 0.2 * BLOCK_HALF_WIDTH   # lowest node within ~1cm of floor => in contact
SETTLE_SPEED_THRESH = 0.01 * BLOCK_WIDTH       # per-step COM displacement below this => "stopped"
SETTLE_RUN          = 5                          # consecutive sub-threshold frames => settled

#Finds the different phase boundaries (airborne, contact, settled)
# for a given trajectory based on the true positions.
def compute_phase_boundaries(true_positions,
                             contact_z_thresh=CONTACT_Z_THRESH,
                             settle_speed_thresh=SETTLE_SPEED_THRESH,
                             settle_run=SETTLE_RUN):
    """
    Returns (t_contact, t_settle) splitting a trajectory into:
        airborne : [0, t_contact)
        contact  : [t_contact, t_settle)   (impact + bounce/slide transient)
        settled  : [t_settle, T)
    Defined on ground truth so boundaries are identical across models/ablations.
    """
    T = true_positions.shape[0]
    min_z = true_positions[..., 2].min(dim=1).values        # (T,) lowest node each frame
    cm    = true_positions.mean(dim=1)                       # (T,3)
    speed = torch.norm(cm[1:] - cm[:-1], dim=-1)             # (T-1,) per-step COM displacement

    contact_mask = min_z <= contact_z_thresh
    nz = torch.nonzero(contact_mask, as_tuple=False)
    t_contact = int(nz[0].item()) if nz.numel() > 0 else T   # T => never contacts (all airborne)

    t_settle = T
    below = speed < settle_speed_thresh
    start = min(t_contact, len(speed))
    for t in range(start, len(speed) - settle_run + 1):
        if (bool(below[t:t + settle_run].all())
                and bool((speed[t:] < 2.0 * settle_speed_thresh).all())):   # and STAYS slow
            t_settle = t
            break
    return t_contact, t_settle


#Computes the three metrics for a single trajectory, given the predicted and true positions over time,
#as well as the rest positions of the cube's nodes.
def compute_metrics(pred_positions, true_positions, rest_positions):

    #Makes sure everything is on the same device for computation.
    device = pred_positions.device
    rest_positions = rest_positions.to(device)

    T, N, _ = pred_positions.shape

    # Centers of mass per timestep
    pred_cm = pred_positions.mean(dim=1)  # (T,3)
    true_cm = true_positions.mean(dim=1)  # (T,3)

    # Rest relative positions (constant)
    q = rest_positions - rest_positions.mean(dim=0)  # (N,3)

    # Pred/true relative positions per timestep
    p_pred = pred_positions - pred_cm[:, None, :]  # (T,N,3)
    p_true = true_positions - true_cm[:, None, :]  # (T,N,3)

    # Apq per timestep: (T,3,3)
    # Apq[t] = sum_n p[t,n]^T outer q[n]
    Apq_pred = torch.einsum('tni,nj->tij', p_pred, q)  # (T,3,3)
    Apq_true = torch.einsum('tni,nj->tij', p_true, q)  # (T,3,3)

    # Batched SVD
    U_p, S_p, Vh_p = torch.linalg.svd(Apq_pred)  # U:(T,3,3), Vh:(T,3,3)
    U_t, S_t, Vh_t = torch.linalg.svd(Apq_true)

    # Rotation fix to ensure a proper rotation (det = +1)
    # d = det(U @ Vh) for each timestep
    d_p = torch.linalg.det(U_p @ Vh_p)  # (T,)
    d_t = torch.linalg.det(U_t @ Vh_t)  # (T,)

    # Build D matrices batched: (T,3,3)
    D_p = torch.eye(3, device=device).expand(T, 3, 3).clone()
    D_t = torch.eye(3, device=device).expand(T, 3, 3).clone()
    D_p[:, 2, 2] = d_p
    D_t[:, 2, 2] = d_t

    R_pred = U_p @ D_p @ Vh_p  # (T,3,3)
    R_true = U_t @ D_t @ Vh_t  # (T,3,3)

    # --- Metric 1: Relative center position error
    center_errors = torch.norm(pred_cm - true_cm, dim=1) / BLOCK_WIDTH  # (T,)

    # --- Metric 2: Absolute angle difference ---
    R_rel = R_pred.transpose(-1, -2) @ R_true  # (T,3,3)
    trace = R_rel.diagonal(dim1=-2, dim2=-1).sum(-1)  # (T,)
    cos_angle = torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0)
    angle_errors = torch.arccos(cos_angle)  # (T,)

    # --- Metric 3: Floor penetration  ---
    z = pred_positions[..., 2]  # (T,N)
    floor_penetrations = torch.clamp(-z, min=0.0).amax(dim=1) / BLOCK_WIDTH  # (T,)

    t_contact, t_settle = compute_phase_boundaries(true_positions)

    def _phase_avg(vec):
        T_ = vec.shape[0]
        out = {}
        for name, sl in (('airborne', slice(0, t_contact)),
                         ('contact',  slice(t_contact, t_settle)),
                         ('settled',  slice(t_settle, T_))):
            seg = vec[sl]
            out[name] = seg.mean().item() if seg.numel() > 0 else float('nan')
        return out

    center_phase = _phase_avg(center_errors)
    angle_phase  = _phase_avg(torch.rad2deg(angle_errors))

    #Returns all of the metrics averaged across the trajectory, and split by phase.
    return {
        'center_error':            center_errors.mean().item(),
        'angle_error_deg':         torch.rad2deg(angle_errors).mean().item(),
        'floor_penetration':       floor_penetrations.mean().item(),
        'center_error_airborne': center_phase['airborne'],
        'center_error_contact':  center_phase['contact'],
        'center_error_settled':  center_phase['settled'],
        'angle_error_airborne':  angle_phase['airborne'],
        'angle_error_contact':   angle_phase['contact'],
        'angle_error_settled':   angle_phase['settled'],
        't_contact': t_contact, 't_settle': t_settle,
    }
