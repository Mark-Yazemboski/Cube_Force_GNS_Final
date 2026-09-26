"""
impulse_diagnostic.py

Answers one question about the contact channel's 46% error:
is the model getting the contact force WRONG, or getting it RIGHT but
SMEARED IN TIME?

THE SETUP

  The wrench labels are IMPULSES, not forces - add_wrench_labels.py accumulates
        acc["Jc"] += data.qfrc_constraint[:3] * dt
  so J_c(t) is the contact impulse delivered during recorded frame t.

  force_contact_err is a PER-FRAME comparison: it asks whether the impulse in
  frame t matches. An impact lasts one or two frames, so if the model delivers
  the right impulse one frame late:

        frame 10:  true = J,  pred = 0   ->  error ||J||
        frame 11:  true = 0,  pred = J   ->  error ||J||
        per-frame error = 2||J||,  which is 200% of the signal

  ...while the TOTAL impulse over the impact is exactly right, so the velocity
  change is exactly right, and the trajectory is unaffected. That is consistent
  with what we see: contact error 46% while center error is 0.05 block-widths.

THE TWO NUMBERS

  E_frame = sum_t || J_pred(t) - J_true(t) ||      <- what we report now
  E_total = || sum_t J_pred(t) - sum_t J_true(t) || <- new: timing-blind

  Both over one contact interval. Then

      timing_fraction = 1 - E_total / E_frame

  1.0  the total impulse is exactly right and every bit of the per-frame error
       is smearing -> the 46% is a TIME-RESOLUTION statement, not a force error
  0.0  the total is as wrong as the frames -> genuine magnitude/direction error
       and the physics terms should in principle be able to reach it

WHY IT MATTERS

  It decides whether the contact-channel error is in scope. The force-structure
  priors (direction, Coulomb magnitude, cone) constrain what the force IS. None
  of them constrains WHEN an impulse lands. If timing_fraction is high, no
  amount of weight tuning will move that 46%, and it should be reported as a
  bounded, attributed limitation instead of an open problem.

USAGE

  Drop compute_impulse_split() into evaluate_force_model.py and call it with
  the per-frame predicted and true contact impulses and the contact mask that
  the evaluator already computes for the phase split.
"""

import numpy as np


def contact_intervals(contact_mask, min_len=1):
    """Contiguous runs of True in a 1-D boolean mask -> [(start, stop), ...].

    An 'interval' is one impact or one continuous period of contact. Splitting
    per interval rather than pooling the whole trajectory matters: a cube that
    bounces twice has two impulses, and summing across both would let an error
    in one cancel an error in the other.
    """
    m = np.asarray(contact_mask).astype(bool)
    if m.size == 0:
        return []
    edges = np.diff(np.concatenate(([0], m.view(np.int8), [0])))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    return [(a, b) for a, b in zip(starts, stops) if b - a >= min_len]


def compute_impulse_split(J_pred, J_true, contact_mask, min_len=1, eps=1e-12):
    """J_pred, J_true: (T, 3) per-frame contact impulses for ONE trajectory.
    contact_mask: (T,) bool.

    Returns a dict. timing_fraction is the headline: the share of the
    per-frame error that vanishes once you stop caring when the impulse landed.
    """
    J_pred = np.asarray(J_pred, dtype=float)
    J_true = np.asarray(J_true, dtype=float)
    out = dict(n_intervals=0, E_frame=0.0, E_total=0.0,
               true_impulse=0.0, timing_fraction=float("nan"))

    for a, b in contact_intervals(contact_mask, min_len):
        dp, dt_ = J_pred[a:b], J_true[a:b]
        out["E_frame"] += float(np.linalg.norm(dp - dt_, axis=1).sum())
        out["E_total"] += float(np.linalg.norm(dp.sum(0) - dt_.sum(0)))
        out["true_impulse"] += float(np.linalg.norm(dt_.sum(0)))
        out["n_intervals"] += 1

    if out["E_frame"] > eps:
        out["timing_fraction"] = 1.0 - out["E_total"] / out["E_frame"]
    if out["true_impulse"] > eps:
        # Both errors as a share of the impulse that was actually delivered,
        # so they sit on the same scale as force_contact_err_over_signal.
        out["E_frame_over_signal"] = out["E_frame"] / out["true_impulse"]
        out["E_total_over_signal"] = out["E_total"] / out["true_impulse"]
    return out


def aggregate(per_traj):
    """Mean over trajectories, skipping any with no contact interval."""
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


# ======================================================================
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    T = 60
    mask = np.zeros(T, bool); mask[20:26] = True        # one 6-frame impact
    J = np.zeros((T, 3)); J[21] = [0.0, 0.0, 1.0]       # impulse in frame 21

    print("  Three synthetic cases, one impact each:\n")
    print(f"  {'case':<34}{'E_frame/sig':>13}{'E_total/sig':>13}{'timing_frac':>13}")

    r = compute_impulse_split(J.copy(), J, mask)
    print(f"  {'perfect':<34}{r['E_frame_over_signal']:>13.3f}"
          f"{r['E_total_over_signal']:>13.3f}{r['timing_fraction']:>13.3f}")

    late = np.zeros_like(J); late[22] = J[21]           # right impulse, 1 frame late
    r = compute_impulse_split(late, J, mask)
    print(f"  {'one frame late (pure timing)':<34}{r['E_frame_over_signal']:>13.3f}"
          f"{r['E_total_over_signal']:>13.3f}{r['timing_fraction']:>13.3f}")

    small = J * 0.5                                     # right timing, half size
    r = compute_impulse_split(small, J, mask)
    print(f"  {'50% too small (pure magnitude)':<34}{r['E_frame_over_signal']:>13.3f}"
          f"{r['E_total_over_signal']:>13.3f}{r['timing_fraction']:>13.3f}")

    smear = np.zeros_like(J); smear[20:23] = J[21] / 3.0   # spread over 3 frames
    r = compute_impulse_split(smear, J, mask)
    print(f"  {'smeared over 3 frames':<34}{r['E_frame_over_signal']:>13.3f}"
          f"{r['E_total_over_signal']:>13.3f}{r['timing_fraction']:>13.3f}")

    print("\n  timing_fraction = 1.0 means the total impulse is exactly right and")
    print("  every bit of the per-frame error is smearing. 0.0 means the total is")
    print("  as wrong as the frames. Read it on the real data and the 46% is")
    print("  either attributed or genuinely open.")
