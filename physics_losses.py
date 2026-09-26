"""
physics_losses.py

NEW FILE - the physics-informed violation losses from the proposal (Eq. 5-6),
mapped onto the force architecture. Each term is its own function so it can be
explained, ablated, and weighted independently.

======================================================================
MAPPING FROM THE PROPOSAL TO THIS IMPLEMENTATION
======================================================================

Proposal Eq. (5) has three terms. Where each one lives here:

  h_diss   "frictional forces must maximize power loss"
           -> h_dissipation() below. The proposal's expression
              || ||J_t v'|| lam_t + lam_n J_t v' ||  is zero exactly when the
              tangential impulse is anti-parallel to slip with magnitude tied
              to the normal impulse - i.e. kinetic Coulomb friction
              phi_t = -mu * phi_n * v_hat. We implement that zero-set
              directly, with mu made EXPLICIT and (by default) LEARNABLE, so
              the model recovers the friction coefficient as a byproduct -
              the same "recovered the physical parameter" story as the drag
              coefficient. Gated on slip speed: static friction may sit
              anywhere inside the cone, so the equality applies only while
              sliding.

  h_pen    "contact impulses must remain non-negative"
           -> ARCHITECTURAL. The normal force goes through a softplus in
              force_gns.py, so min(0, phi_n)^2 == 0 by construction. There is
              no loss term because violation is impossible, not merely
              discouraged. gamma_2 is not needed.

  h_smooth "regularize the predicted fluid forces ... to promote smooth
            fluid force distributions"
           -> Two terms, because our fluid head is a single COM wrench, not a
              per-node field (see force_gns.py for why):
              (a) h_fluid_anchor():   the fluid wrench should match the
                  ANALYTIC drag law k|u|u evaluated at the measured relative
                  wind (and the fluid torque should be ~0, as it is in
                  MuJoCo). This is the physics-infused version of
                  "regularize the fluid forces": shrink toward the law, not
                  toward zero.
                  NOTE the zero-torque default is a MuJoCo-specific claim;
                  see h_fluid_torque for why it must not be carried to real
                  data unchanged.
              (b) h_fluid_temporal_smooth(): the fluid wrench must vary
                  SMOOTHLY IN TIME. Drag is a smooth function of relative
                  wind, which changes slowly; contact events are the jumpy
                  thing. With space collapsed to a point, "smooth
                  distribution" becomes smoothness along the trajectory.
                  Requires multistep >= 2 (needs consecutive predictions).

  plus     "sparsity regularization for concentrated forces" (Fig. 1)
           -> h_contact_sparsity(): L1 on contact force magnitudes.

Overall loss (Eq. 6):  L = L_pred + sum_j gamma_j h_j
The gammas are the w_* weights in run_force_multi_step.py.

======================================================================
WHY THESE TERMS, GIVEN WHAT WE MEASURED
======================================================================
The wrench-label evaluation showed the failure precisely: the fluid channel
carries a ~0.2 mg force during contact - the size of the friction force
mu*m*g - identically at every wind level, while free-flight drag is predicted
well. The loss only observes the net wrench (6 numbers) but the model outputs
30, so the split is underdetermined and the optimizer parks friction in the
easiest channel. These terms remove that degeneracy from both sides:
h_fluid_anchor pins what fluid IS ALLOWED to be (the drag law),
h_fluid_temporal_smooth pins how it may CHANGE (slowly), and h_dissipation
gives the displaced friction a correctly-structured home in the contact
channel (anti-parallel to slip, proportional to local normal force, one
global mu).

======================================================================
NORMALIZATION
======================================================================
Every force-like residual is divided by phi_g = g*dt^2, the specific weight
of the cube per step^2 - so a raw value of 1.0 means "a violation the size of
gravity". Torque-like residuals are divided by the empirical angular
acceleration std. This makes the printed raw magnitudes interpretable and the
weights transferable across datasets. Calibration rule: after one epoch, read
the printed raws and set each weight so (weight * raw) is 1-10% of the
position loss.

Slip velocities and gates are DETACHED: the physics terms constrain the
predicted forces given the observed motion; they must not create an incentive
to change the motion to relax the constraint.
"""

from collections import deque

import numpy as np
import torch
import torch.nn as nn



# ======================================================================
# DIAGNOSTIC HISTORY  (module-level, so the run script can read it after
# train_force_gnn returns without the trainer having to hand it back)
# ----------------------------------------------------------------------
# slip_gate_report() appends to DIAG_HISTORY on every call, so the alignment
# / mu_implied / gate-occupancy trace accumulates for free. The per-epoch
# numbers come from ONE batch and are noisy (measured spread: alignment
# 0.726 +/- 0.094, with a 0.503 outlier), so summarize_diagnostics()
# averages the tail rather than reporting the last value.
#
# RAW_HISTORY is optional: if the trainer calls push_raw_terms(raw_terms)
# once per epoch, the physics-term magnitudes land in the CSV too. Those
# were the strongest signal in the DIR arm (fric_dir 1.48e-2 -> 6.9e-3 with
# no overlap between arms), so they are worth recording.
# ======================================================================
DIAG_HISTORY = deque(maxlen=200)
RAW_HISTORY = deque(maxlen=200)


def push_raw_terms(raw_terms):
    """Optional trainer hook: record one epoch of raw physics-term values.
    raw_terms: {name: float} - the same dict the trainer prints."""
    if raw_terms:
        RAW_HISTORY.append({k: float(v) for k, v in raw_terms.items()})


def reset_diagnostics():
    """Clear both buffers. Call at the start of a run if several trainings
    share one Python process."""
    DIAG_HISTORY.clear()
    RAW_HISTORY.clear()


def summarize_diagnostics(last_n=20):
    """Mean over the last `last_n` recorded epochs, for the run report.

    Returns a flat dict of floats ready to drop into the settings dict.
    Empty buffers give an empty dict, so this is safe to call unconditionally.
    misalign_deg is arccos(mean align), NOT the mean of the per-epoch angles:
    arccos is nonlinear, so averaging degrees would bias the result.
    """
    out = {}
    if DIAG_HISTORY:
        tail = list(DIAG_HISTORY)[-last_n:]
        al = np.array([r["mean_align"] for r in tail], dtype=float)
        mi = np.array([r["mu_implied"] for r in tail], dtype=float)
        gf = np.array([r["gate_frac"] for r in tail], dtype=float)
        al, mi, gf = al[np.isfinite(al)], mi[np.isfinite(mi)], gf[np.isfinite(gf)]
        if al.size:
            m = float(al.mean())
            out["diag_align"] = m
            out["diag_align_std"] = float(al.std(ddof=1)) if al.size > 1 else 0.0
            out["diag_misalign_deg"] = float(
                np.degrees(np.arccos(min(1.0, max(-1.0, m)))))
        if mi.size:
            out["diag_mu_implied"] = float(mi.mean())
            out["diag_mu_implied_std"] = float(mi.std(ddof=1)) if mi.size > 1 else 0.0
        if gf.size:
            out["diag_gate_frac"] = float(gf.mean())
        for key in ("cancel_slide", "cancel_static"):
            v = np.array([r.get(key, np.nan) for r in tail], dtype=float)
            v = v[np.isfinite(v)]
            if v.size:
                out[f"diag_{key}"] = float(v.mean())
        out["diag_n_epochs"] = len(tail)
    if RAW_HISTORY:
        tail = list(RAW_HISTORY)[-last_n:]
        for k in tail[-1]:
            v = np.array([r[k] for r in tail if k in r], dtype=float)
            v = v[np.isfinite(v)]
            if v.size:
                out[f"raw_{k}"] = float(v.mean())
    return out


class PhysicsLosses(nn.Module):
    """Holds the physics-loss state (the friction coefficient mu) and exposes
    one method per violation term. Instantiate once in training, move to the
    device, and include .parameters() in the optimizer when mu is learnable.

    mu modes:
      fixed_mu = None, learn_mu = True   -> mu is a learnable parameter
                                            (init mu_init), recovered from
                                            data. Parameterized as log(mu) so
                                            it stays positive.
      fixed_mu = <float>                 -> mu clamped to the known value
                                            (stronger physics infusion; use
                                            for the ablation arm).
    """

    def __init__(self, phi_g, ang_scale_vec,
                 mu_init=0.2, learn_mu=True, fixed_mu=None,
                 k_init=0.0285, learn_k=False, fixed_k=None,
                 slip_v0=1e-3, slip_tau=1e-4, eps=1e-9):
        super().__init__()
        self.register_buffer("phi_g", torch.as_tensor(float(phi_g)))
        self.register_buffer("ang_scale_vec",
                             torch.as_tensor(ang_scale_vec, dtype=torch.float32))
        self.fixed_mu = fixed_mu
        if fixed_mu is None:
            log_mu = torch.log(torch.tensor(float(mu_init)))
            if learn_mu:
                self.log_mu = nn.Parameter(log_mu)
            else:
                self.register_buffer("log_mu", log_mu)

        # k/m, the quadratic drag coefficient, gets exactly the same treatment
        # as mu: log-space so it stays positive by construction, learnable so
        # the pipeline carries no constant fitted offline against MuJoCo.
        # Identification is the mirror of mu's: the prediction loss determines
        # the fluid force, and the analytic drag law reads the coefficient off
        # it. With use_drag_baseline=False the anchor makes that an explicit
        # least-squares fit, k* = sum(a.w)/sum(w.w) with w = ||u||u, which is a
        # one-parameter regression on a single basis function - unbiased at
        # 100% noise on the fluid head in simulation. With the baseline ON the
        # anchor reduces to ||residual||^2 and k cancels out of it, so k is
        # identified only through the prediction loss and only insofar as the
        # anchor suppresses the residual. Baseline OFF is the clean case.
        self.fixed_k = fixed_k
        if fixed_k is None:
            log_k = torch.log(torch.tensor(float(k_init)))
            if learn_k:
                self.log_k = nn.Parameter(log_k)
            else:
                self.register_buffer("log_k", log_k)

        self.slip_v0 = slip_v0        # m/step: slip-speed gate center
        self.slip_tau = slip_tau      # m/step: gate softness
        self.eps = eps

    @property
    def mu(self):
        if self.fixed_mu is not None:
            return torch.as_tensor(self.fixed_mu, device=self.phi_g.device)
        return torch.exp(self.log_mu)

    @property
    def k_over_m(self):
        """Quadratic drag coefficient k/m, as a tensor. drag_accel_step()
        multiplies by it, so passing this instead of a float is all that is
        needed to put k in the graph."""
        if self.fixed_k is not None:
            return torch.as_tensor(self.fixed_k, device=self.phi_g.device)
        return torch.exp(self.log_k)

    # ------------------------------------------------------------------
    # h_diss  (proposal Eq. 5, first term)
    # ------------------------------------------------------------------
    def h_dissipation(self, phi_contact, c_w, v_node, wall_n):
        """Kinetic Coulomb friction on sliding contact nodes.

        Zero exactly when  phi_t = -mu * phi_n * v_hat_t  on every sliding
        contact node: friction opposes slip (=> can never add energy, the
        proposal's 'maximize power loss') AND its magnitude is mu times the
        LOCAL normal force. This couples the tangential channel to the normal
        channel through one global mu - structure the position loss alone
        cannot supply.

        Gating: contact weight c_w (geometric) x a soft slip gate
        sigma((|v_t| - v0)/tau). Static contact (settled cube) is NOT forced
        to the cone boundary - static friction may be anything inside it.

        phi_contact: (B, N, 3) contact specific forces (m/step^2)
        c_w:         (B, N, 1) contact weight
        v_node:      (B, N, 3) per-node velocity (m/step) - detached inside
        wall_n:      (3,) unit wall normal
        Returns a scalar: weighted mean squared residual in units of phi_g.
        """
        v = v_node.detach()
        v_t = v - (v * wall_n).sum(-1, keepdim=True) * wall_n
        speed = v_t.norm(dim=-1, keepdim=True)
        v_hat = v_t / (speed + self.eps)
        slip_gate = torch.sigmoid((speed - self.slip_v0) / self.slip_tau)

        phi_n = (phi_contact * wall_n).sum(-1, keepdim=True)      # signed normal
        phi_t = phi_contact - phi_n * wall_n

        # phi_n is DETACHED: Coulomb says what friction may be GIVEN the
        # normal force. Left attached, this residual carries a normal-direction
        # gradient of size mu*|dL/dphi_t|, so the cheapest fix for
        # ||phi_t|| != mu*phi_n is to move phi_n - a well-determined,
        # position-loss-observable quantity - instead of the friction.
        resid = (phi_t + self.mu * phi_n.detach() * v_hat) / self.phi_g
        # Normalize by CONTACT weight only; the slip gate lives in the
        # numerator. If it were in the denominator too, a batch where every
        # node is equally (barely) gated would cancel the gate entirely and
        # static contact would be penalized at full strength. The gate is
        # sharp (tau default 1e-4 m/step) and DETACHED, so sharpness costs no
        # gradient pathology.
        w = (c_w * slip_gate).detach()                            # (B, N, 1)
        norm = c_w.detach().sum() + self.eps
        return (w * resid.pow(2).sum(-1, keepdim=True)).sum() / norm

    @torch.no_grad()
    def slip_gate_report(self, phi_contact, c_w, v_node, wall_n, dt=None):
            """Is h_dissipation awake? Mirrors h_dissipation's gating exactly.
    
            h_dissipation weights every node by (c_w * slip_gate). If the
            slip_gate factor is ~0 across the batch, the term contributes
            nothing at ANY weight and mu - whose only gradient path is this
            term - is being fit from whatever sliver of frames does get through.
    
            Returns a dict; see fmt_slip_gate_report() for the one-line print.
            phi_contact/c_w/v_node/wall_n: the SAME tensors you pass to
            compute_step_terms. dt (s) is optional and only converts the
            m/step figures to m/s for readability.
            """
            v = v_node.detach()
            v_t = v - (v * wall_n).sum(-1, keepdim=True) * wall_n
            speed = v_t.norm(dim=-1, keepdim=True)                 # (B,N,1) m/step
            v_hat_d = v_t / (speed + self.eps)
            gate = torch.sigmoid((speed - self.slip_v0) / self.slip_tau)
    
            w_c = c_w.detach()
            contact_mass = w_c.sum().clamp_min(self.eps)
            # This ratio IS the multiplier on h_dissipation's effective size.
            gate_frac = float((w_c * gate).sum() / contact_mass)
    
            # Slip-speed distribution over nodes that are actually in contact,
            # which is the population the gate is deciding about.
            in_contact = (w_c > 0.5).squeeze(-1)
            s = speed.squeeze(-1)[in_contact]
            if s.numel() == 0:
                pct = {q: float('nan') for q in (10, 50, 90, 99)}
            else:
                qs = torch.tensor([0.10, 0.50, 0.90, 0.99], device=s.device,
                                  dtype=s.dtype)
                vals = torch.quantile(s, qs)
                pct = {q: float(x) for q, x in zip((10, 50, 90, 99), vals)}
    
            # Counterfactuals: what would gate_frac be at a lower threshold?
            # This is the number that decides whether slip_v0 is the real knob.
            cf = {}
            for div in (3.0, 10.0, 30.0):
                g2 = torch.sigmoid((speed - self.slip_v0 / div) / self.slip_tau)
                cf[div] = float((w_c * g2).sum() / contact_mass)
    
            # mu implied by the model's OWN predicted forces on sliding nodes.
            # Coulomb says ||phi_t|| = mu * phi_n while sliding, so this is the
            # mu the force decomposition is currently consistent with -
            # independent of the learnable mu parameter.
            phi_n = (phi_contact.detach() * wall_n).sum(-1, keepdim=True)
            phi_t = phi_contact.detach() - phi_n * wall_n
            wg = w_c * gate
            mag = phi_t.norm(dim=-1, keepdim=True)
            num = (wg * mag).sum()
            den = (wg * phi_n.clamp_min(0.0)).sum()
            mu_implied = float(num / den) if float(den) > self.eps else float('nan')

            # DIRECTIONAL alignment, measured directly instead of inferred from
            # the mu_param/mu_implied ratio. +1 = friction exactly opposes slip
            # (perfect Coulomb), 0 = perpendicular, -1 = friction DRIVES the
            # slip. Magnitude-weighted, so a negligible misaimed force does not
            # drag the average down. This is the crossing-arrow defect as a
            # scalar: mu_param = mu_implied * mean_align.
            # CANCELLATION FRACTION, per branch. 1 - ||sum phi_t|| / sum||phi_t||
            # over the nodes of each cube: 0 = all friction pulling together,
            # 1 = perfect cancellation (large opposing forces summing to
            # nothing). The prediction loss sees only the net, so a cancelling
            # field is free; and in the STATIC branch every other term is gated
            # off, so nothing else measures this at all.
            def _cancel(weight):
                num = (weight * phi_t).sum(dim=1).norm(dim=-1)      # ||sum||
                den = (weight * mag).sum(dim=1).squeeze(-1)         # sum|| ||
                ok = den > self.eps
                if not bool(ok.any()):
                    return float('nan')
                return float((1.0 - num[ok] / den[ok]).mean())

            cancel_slide = _cancel(w_c * gate)
            cancel_static = _cancel(w_c * (1.0 - gate))

            align = -(phi_t * v_hat_d).sum(-1, keepdim=True) / (mag + self.eps)
            wgm = wg * mag
            mean_align = (float((wgm * align).sum() / wgm.sum())
                          if float(wgm.sum()) > self.eps else float('nan'))

            report = dict(gate_frac=gate_frac,
                        slip_v0=float(self.slip_v0),
                        slip_tau=float(self.slip_tau),
                        pct=pct, counterfactual=cf,
                        mu_implied=mu_implied, mu_param=float(self.mu),
                        mean_align=mean_align,
                        cancel_slide=cancel_slide, cancel_static=cancel_static,
                        misalign_deg=float(np.degrees(np.arccos(
                            min(1.0, max(-1.0, mean_align)))))
                        if mean_align == mean_align else float('nan'),
                        n_contact_nodes=float(contact_mass), dt=dt)
            DIAG_HISTORY.append(report)
            return report
    
    
    @staticmethod
    def fmt_slip_gate_report(r):
        """Two lines for the epoch log. Import alongside PhysicsLosses."""
        dt = r["dt"]
        to_ms = (lambda x: x / dt) if dt else (lambda x: float('nan'))
        u = "m/s" if dt else "m/step"
        conv = to_ms if dt else (lambda x: x)
        p = r["pct"]
        cf = r["counterfactual"]
        return (
            f"  Slip gate | OPEN {r['gate_frac']:6.1%} of contact weight  "
            f"| v0={conv(r['slip_v0']):.3f} {u}  "
            f"| contact slip p10/p50/p90/p99 = "
            f"{conv(p[10]):.3f}/{conv(p[50]):.3f}/{conv(p[90]):.3f}/{conv(p[99]):.3f} {u}\n"
            f"            | if v0 were /3: {cf[3.0]:5.1%}   /10: {cf[10.0]:5.1%}   "
            f"/30: {cf[30.0]:5.1%}   "
            f"| mu implied by predicted forces = {r['mu_implied']:.3f} "
            f"(mu param = {r['mu_param']:.3f})\n"
            f"            | friction alignment = {r['mean_align']:+.3f} "
            f"({r['misalign_deg']:.0f} deg off anti-parallel; 1.000 = perfect Coulomb)\n"
            f"            | cancellation  sliding {r['cancel_slide']:.3f}  "
            f"static {r['cancel_static']:.3f}   (0 = forces pull together, "
            f"1 = they cancel to nothing)"
        )


    # ==================================================================
    # SPLIT COULOMB  (replaces the joint h_dissipation above)
    # ------------------------------------------------------------------
    # h_dissipation minimizes || phi_t + mu phi_n vhat ||^2, one squared
    # residual over a VECTOR. That couples magnitude and direction through a
    # single global scalar, and mu can only absorb the coupling one way:
    #
    #     mu*  =  -sum w phi_n (phi_t . vhat) / sum w phi_n^2
    #          =  mu_true * <cos(misalignment)>
    #
    # Verified numerically: at 44 deg of misalignment the joint term reports
    # mu = 0.140 where the truth is 0.198, while a magnitude-only term reports
    # 0.198 at ANY misalignment. Splitting therefore does two things at once:
    # it de-biases the recovered friction coefficient, and it isolates a term
    # whose only job is fixing the crossing-arrow defect.
    # ==================================================================
    def h_friction_direction(self, phi_contact, c_w, v_node, wall_n):
        """DIRECTION half of Coulomb: friction opposes THAT node's own slip.

        No mu appears, so alignment error cannot leak into the mu estimate.
        Correct under spin: each node is compared against its own local slip
        direction, so the genuine fanning-out of friction on a yawing cube is
        allowed, unlike a global "all frictions parallel" penalty.

        The per-node cost is  ||phi_t|| * (1 + phi_hat_t . vhat), which is 0
        when friction exactly opposes slip and 2||phi_t|| when it drives it.
        Weighting by magnitude means a large misaimed force is expensive and a
        negligible one is nearly free. LINEAR, not squared: the constant
        gradient keeps pushing all the way to alignment, the same reason L1
        drives exact sparsity where L2 only shrinks.

        Returns a scalar in units of phi_g.
        """
        v = v_node.detach()
        v_t = v - (v * wall_n).sum(-1, keepdim=True) * wall_n
        speed = v_t.norm(dim=-1, keepdim=True)
        v_hat = v_t / (speed + self.eps)
        slip_gate = torch.sigmoid((speed - self.slip_v0) / self.slip_tau)

        phi_n = (phi_contact * wall_n).sum(-1, keepdim=True)
        phi_t = phi_contact - phi_n * wall_n
        mag = phi_t.norm(dim=-1, keepdim=True)
        misalign = 1.0 + (phi_t * v_hat).sum(-1, keepdim=True) / (mag + self.eps)

        w = (c_w * slip_gate).detach()
        return (w * (mag / self.phi_g) * misalign).sum() / (c_w.detach().sum() + self.eps)

    def h_friction_magnitude(self, phi_contact, c_w, v_node, wall_n):
        """MAGNITUDE half of Coulomb: ||phi_t|| = mu * phi_n while sliding.

        This is mu's ONLY gradient path in the split formulation. Because the
        direction is excluded, mu converges to the friction coefficient itself
        rather than to mu * <cos misalignment>, so `recovered mu` becomes a
        measurement of friction instead of a measurement of friction times
        alignment.

        Same gating and normalization as h_dissipation, so their raw
        magnitudes are directly comparable in the epoch log.
        """
        v = v_node.detach()
        v_t = v - (v * wall_n).sum(-1, keepdim=True) * wall_n
        speed = v_t.norm(dim=-1, keepdim=True)
        slip_gate = torch.sigmoid((speed - self.slip_v0) / self.slip_tau)

        phi_n = (phi_contact * wall_n).sum(-1, keepdim=True)
        phi_t = phi_contact - phi_n * wall_n
        # phi_n DETACHED (see h_dissipation): the normal force is the given,
        # friction is the unknown. mu keeps its gradient - this is its only path.
        resid = (phi_t.norm(dim=-1, keepdim=True)
                 - self.mu * phi_n.detach()) / self.phi_g

        w = (c_w * slip_gate).detach()
        return (w * resid.pow(2)).sum() / (c_w.detach().sum() + self.eps)

    def h_friction_cone(self, phi_contact, c_w, wall_n, v_node=None):
        """The COULOMB CONE bound  ||phi_t|| <= mu * phi_n.

        The other half of Coulomb friction, and the half that has been missing.
        h_dissipation / h_friction_* enforce the SLIDING equality and are
        therefore gated off below slip_v0 - which leaves the STATIC regime, the
        settled frames where the rollout jitters, with no constraint on
        friction whatsoever. The cone is an INEQUALITY, valid in both regimes:
        it says nothing about direction and only bounds magnitude, so it cannot
        over-constrain static contact the way the equality would.

        Gated by contact weight ONLY - no slip gate. Satisfied forces cost
        exactly zero (hinge), so this is free wherever the model is already
        physical.

        mu is DETACHED here. With gradient, the cheapest way to reduce a hinge
        penalty is to inflate mu until the constraint is never active, which
        would destroy the mu measurement that h_friction_magnitude provides.
        """
        mu = self.mu.detach()
        phi_n = (phi_contact * wall_n).sum(-1, keepdim=True)
        phi_t = phi_contact - phi_n * wall_n
        # BOTH mu and phi_n are held fixed: the only way to satisfy the cone
        # is to shrink the offending friction force, not to widen the cone.
        excess = phi_t.norm(dim=-1, keepdim=True) - mu * phi_n.detach().clamp_min(0.0)

        # STATIC BRANCH ONLY, when v_node is supplied.
        #
        # Complementary slackness has two disjoint branches. On a SLIDING node
        # the constraint is ACTIVE and h_friction_magnitude enforces the
        # equality ||phi_t|| = mu phi_n, which already implies the inequality -
        # the cone is redundant there. On a STATIC node the constraint is
        # INACTIVE, the equality is switched off by the slip gate, and the cone
        # is the only law left.
        #
        # Applying it to sliding nodes as well is not merely redundant, it is
        # unstable: the bound sits at a mu that h_friction_magnitude is fitting
        # from those same forces, so clipping lowers mu, which tightens the
        # bound, which clips harder. Measured: mu_implied 0.194 -> 0.186 ->
        # 0.172 as w_fric_cone went 0 -> 0.5 -> 1.5.
        #
        # This is the SAME Coulomb cone, restricted to the branch where it is
        # the operative condition. No margin, no weakened inequality.
        w = c_w.detach()
        if v_node is not None:
            v = v_node.detach()
            v_t = v - (v * wall_n).sum(-1, keepdim=True) * wall_n
            speed = v_t.norm(dim=-1, keepdim=True)
            slip_gate = torch.sigmoid((speed - self.slip_v0) / self.slip_tau)
            w = w * (1.0 - slip_gate)
        # Denominator is the TOTAL contact weight, not the gated weight -
        # matching h_friction_magnitude. Normalizing by the gated weight would
        # make this a weighted mean over static nodes only, in which case a
        # uniform gate cancels top and bottom and the gating does nothing.
        return ((w * (excess.clamp_min(0.0) / self.phi_g).pow(2)).sum()
                / (c_w.detach().sum() + self.eps))

    # ------------------------------------------------------------------
    # h_pen  (proposal Eq. 5, second term) - architectural, no code needed.
    # softplus in force_gns.assemble_contact_forces makes phi_n >= 0 always,
    # so min(0, phi_n)^2 == 0 by construction.
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # h_smooth part (a): anchor the fluid wrench to the analytic drag law
    # ------------------------------------------------------------------
    def h_fluid_anchor(self, a_fluid_total, drag_target):
        """The fluid FORCE must be what aerodynamics permits: the TOTAL
        predicted fluid acceleration (learned residual plus baseline, if
        enabled) is pulled toward the analytic quadratic law k|u|u evaluated
        at the MEASURED relative wind. This is shrinkage toward the physics,
        not toward zero - with the drag baseline on it reduces to keeping the
        residual small (PIROM: the physics carries, the network corrects).

        a_fluid_total: (B, 3) m/step^2 (learned + baseline)
        drag_target:   (B, 3) m/step^2, analytic law at measured u - DETACHED
                       by the caller (it is a target, not a pathway).
        """
        return ((a_fluid_total - drag_target) / self.phi_g).pow(2).sum(-1).mean()

    def h_fluid_torque(self, alpha_fluid, alpha_target=None):
        """Anchor the fluid TORQUE to what the low-fidelity aero model
        predicts - exactly parallel to h_fluid_anchor for the force.

        alpha_target=None means "the model predicts zero torque". That is a
        DATASET-SPECIFIC claim, not a law of physics:

          * It is approximately right for a symmetric cube in MuJoCo, whose
            passive fluid model produces only weak angular drag. VERIFY it
            against the tau_fluid labels before relying on it - the evaluation
            prints the true fluid torque magnitude for this purpose.
          * It is WRONG in general. A body whose center of pressure is offset
            from its COM experiences real aerodynamic torque, and that offset
            is exactly what changes when contact alters the exposed geometry
            (the proposal's contact/fluid/robot trinity). On real data, and on
            the aerial manipulator in particular, fluid torque is a genuine
            physical quantity and must not be regularized to zero.

        For those cases pass alpha_target from an analytic or reduced-order
        aero model, so this becomes shrinkage toward the physics rather than
        toward zero - the same PIROM pattern as the force anchor, and the
        nested f_theta(f_phi(...)) structure the proposal describes.

        Kept separate from the force anchor (own weight, own printed raw)
        because the two can sit in very different regimes: a rotation event
        the contact channel has not learned to explain shows up HERE, and you
        want to see it rather than have it averaged into the force number.
        """
        resid = alpha_fluid if alpha_target is None else (alpha_fluid - alpha_target)
        return (resid / self.ang_scale_vec).pow(2).sum(-1).mean()

    # ------------------------------------------------------------------
    # h_smooth part (b): the fluid force may only change slowly in time
    # ------------------------------------------------------------------
    def h_fluid_temporal_smooth(self, fluid_series, torque_series=None):
        """Fluid loads vary smoothly in time; contact impulses are the jumpy
        thing. Penalizing the step-to-step change of the fluid wrench pushes
        any rapidly-switching, contact-synchronized compensation (what stolen
        friction looks like at a contact event) out of the fluid channel.

        THIS IS THE TERM THAT TRANSFERS TO REAL DATA. Unlike the anchors, it
        makes no claim about the MAGNITUDE of the fluid wrench - only that
        aerodynamic loads cannot switch discontinuously. That holds for a
        cube in MuJoCo, for a tumbling body in a wind tunnel, and for the
        aerial manipulator, whether or not any analytic model is available.
        Torque is included for exactly that reason: real fluid torque is
        nonzero but still smooth, so smoothness constrains it without
        pretending to know its value.

        Needs consecutive predictions: multistep >= 2. Returns 0 at K=1 (the
        trainer warns once).

        fluid_series:  list of (B, 3) total fluid accelerations, one per unroll
                       step, in graph (not detached - both ends get gradient).
        torque_series: optional matching list of (B, 3) fluid angular
                       accelerations. Normalized by the angular scale so the
                       two contributions are commensurate under one weight.
        """
        if len(fluid_series) < 2:
            return fluid_series[0].new_zeros(())
        diffs = [((b - a) / self.phi_g).pow(2).sum(-1).mean()
                 for a, b in zip(fluid_series[:-1], fluid_series[1:])]
        total = torch.stack(diffs).mean()
        if torque_series is not None and len(torque_series) >= 2:
            tdiffs = [((b - a) / self.ang_scale_vec).pow(2).sum(-1).mean()
                      for a, b in zip(torque_series[:-1], torque_series[1:])]
            total = total + torch.stack(tdiffs).mean()
        return total

    # ------------------------------------------------------------------
    # contact sparsity  (proposal Fig. 1: "sparsity regularization for
    # concentrated forces")
    # ------------------------------------------------------------------
    def h_contact_sparsity(self, phi_contact):
        """L1 on contact force magnitudes: contact should be a few loaded
        points, not a diffuse field. Use with care - it also shrinks the
        legitimate resting normal forces, so keep the weight small."""
        return (phi_contact.norm(dim=-1) / self.phi_g).mean()

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------
    def compute_step_terms(self, phi_contact, c_w, v_node, wall_n,
                           a_fluid_total, alpha_fluid, drag_target, weights,
                           alpha_target=None):
        """All per-step terms as a dict of RAW (unweighted) scalars. Terms
        whose weight is zero are skipped (no wasted compute)."""
        raws = {}
        if weights.get("w_diss", 0) > 0:
            raws["diss"] = self.h_dissipation(phi_contact, c_w, v_node, wall_n)
        if weights.get("w_fric_dir", 0) > 0:
            raws["fric_dir"] = self.h_friction_direction(
                phi_contact, c_w, v_node, wall_n)
        if weights.get("w_fric_mag", 0) > 0:
            raws["fric_mag"] = self.h_friction_magnitude(
                phi_contact, c_w, v_node, wall_n)
        if weights.get("w_fric_cone", 0) > 0:
            raws["fric_cone"] = self.h_friction_cone(
                phi_contact, c_w, wall_n, v_node)
        if weights.get("w_fluid_anchor", 0) > 0:
            raws["fluid_anchor"] = self.h_fluid_anchor(a_fluid_total, drag_target)
        if weights.get("w_sparse", 0) > 0:
            raws["sparse"] = self.h_contact_sparsity(phi_contact)
        return raws

    @staticmethod
    def weighted_total(raws, weights):
        """sum_j gamma_j h_j  (proposal Eq. 6). raws holds RAW magnitudes;
        weights maps 'w_<name>' -> gamma."""
        total = 0.0
        for name, val in raws.items():
            total = total + weights.get("w_" + name, 0.0) * val
        return total
