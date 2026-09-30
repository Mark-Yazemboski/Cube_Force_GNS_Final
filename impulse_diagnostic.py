"""Diagnostics for separating contact-impulse error from timing error.

Computes framewise and timing-blind impulse errors for each contact interval,
then aggregates the results across trajectories. Used by
evaluate_force_model.py.
"""

import numpy as np

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
def aggregate(per_traj):

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
