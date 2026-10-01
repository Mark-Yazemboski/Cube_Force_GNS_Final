"""Compute the numerical measures used to judge predicted cube motion and contact
impulses. This module compares predicted and true node positions to measure
center-of-mass error, orientation error, and floor penetration, and uses the
true trajectory to divide results into airborne, contact, and settled phases.
It also compares framewise impulse errors with errors in the total impulse
over each contact interval, then aggregates those diagnostics across
trajectories. These calculations support rollout validation and final model
evaluation.
"""

import numpy as np
import torch
from force_data import BLOCK_HALF_WIDTH, BLOCK_WIDTH

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


# Contact-impulse error and timing diagnostics.

# This function identifies contiguous contact intervals in a boolean contact mask.
def contact_intervals(contact_mask):
    """
    An 'interval' is one impact or one continuous period of contact.
    """

    # Convert the contact mask to a boolean numpy array.
    m = np.asarray(contact_mask).astype(bool)
    if m.size == 0:
        return []

    # Find the edges where contact starts and stops.
    edges = np.diff(np.concatenate(([0], m.view(np.int8), [0])))

    # Starts are where edges == 1, stops are where edges == -1.
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)

    # Pair up starts and stops into intervals.
    return list(zip(starts, stops))

# Computes the split of impulse error into timing and magnitude components for a single trajectory.
# Basically we are separating the error due to mistimed impulses from the error due to incorrect impulse magnitudes.
# Without this, all errors would be lumped together, making it hard to tell if the model is wrong because it
# mistimed the impulses or because it predicted the wrong magnitudes.
def compute_impulse_split(J_pred, J_true, contact_mask, eps=1e-12):

    # Convert inputs to numpy arrays of type float.
    J_pred = np.asarray(J_pred, dtype=float)
    J_true = np.asarray(J_true, dtype=float)

    # Initialize the output dictionary with default values.
    out = dict(n_intervals=0, E_frame=0.0, E_total=0.0,
               true_impulse=0.0, timing_fraction=float("nan"))

    # Loop over each contact interval and accumulate errors.
    for a, b in contact_intervals(contact_mask):

        # Extract the predicted and true impulses for this contact interval.
        dp, dt_ = J_pred[a:b], J_true[a:b]

        # Compute the framewise error (sum of per-frame norms) and the total error (norm of summed impulses).
        out["E_frame"] += float(np.linalg.norm(dp - dt_, axis=1).sum())
        out["E_total"] += float(np.linalg.norm(dp.sum(0) - dt_.sum(0)))
        out["true_impulse"] += float(np.linalg.norm(dt_.sum(0)))
        out["n_intervals"] += 1

    # Compute the timing fraction if the framewise error is significant.
    if out["E_frame"] > eps:
        out["timing_fraction"] = 1.0 - out["E_total"] / out["E_frame"]

    # Compute the errors as a share of the total true impulse if it is significant.
    if out["true_impulse"] > eps:
        # Both errors as a share of the impulse that was actually delivered,
        # so they sit on the same scale as force_contact_err_over_signal.
        out["E_frame_over_signal"] = out["E_frame"] / out["true_impulse"]
        out["E_total_over_signal"] = out["E_total"] / out["true_impulse"]
    return out


# Aggregates per-trajectory impulse diagnostics into overall statistics.
def aggregate_impulse_metrics(per_traj):

    # Compute the mean and standard deviation of each key across trajectories, ignoring NaNs.
    keys = ("timing_fraction", "E_frame_over_signal", "E_total_over_signal")
    out = {}
    for k in keys:
        v = np.array([d.get(k, np.nan) for d in per_traj], dtype=float)
        v = v[np.isfinite(v)]
        if v.size:
            out[f"impulse_{k}"] = float(v.mean())
            out[f"impulse_{k}_std"] = float(v.std(ddof=1)) if v.size > 1 else 0.0
    out["impulse_n_traj"] = len(per_traj)
    return out
