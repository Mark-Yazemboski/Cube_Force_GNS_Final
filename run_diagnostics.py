"""
run_diagnostics.py

Reads the two checkpoints train_force_gns.py writes and turns them into extra
metrics for the run report. Nothing here trains or evaluates anything; it is
pure post-processing.

WHAT IT PULLS

  <model>_physics.pt        recovered_mu, recovered_k_over_m
  <model>_loss_history.pt   train loss, best val, mu_trace, k_trace,
                            total optimizer steps

WHY THIS EXISTS

  recovered_k_over_m was only ever written to _physics.pt and printed to the
  log, so comparing k across runs meant grepping four log files. The converged
  prediction loss had the same problem, and it is the denominator of the
  physics-weight calibration (gamma = 0.03 * L_pred / raw), so it is needed
  often enough to belong in the CSV.

  final_train_loss is a MEAN OVER THE LAST N EPOCHS, not the final value. The
  per-epoch loss bounces by ~10% (0.146 to 0.193 across the last five epochs of
  one run), and reading a single epoch off the end of a log is how the first
  weight calibration ended up 75x wrong.
"""

import os

import numpy as np
import torch


def _load(path):
    """Load one of our own checkpoints (written by train_force_gns.py, so
    unpickling is safe). Returns None if the file is missing or unreadable."""
    if not os.path.exists(path):
        return None
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"  [run_diagnostics] could not read {os.path.basename(path)}: {e}")
        return None


def _trace_arrays(trace):
    """[(epoch, value), ...] -> (epochs, values) as float arrays, or None."""
    if not trace:
        return None
    a = np.asarray(trace, dtype=float)
    if a.ndim != 2 or a.shape[1] != 2 or a.shape[0] == 0:
        return None
    return a[:, 0], a[:, 1]


def collect_run_diagnostics(save_model_path, last_n=20):
    """-> flat dict of floats to merge into the run report's metrics.

    save_model_path is the same path handed to train_force_gnn; the
    checkpoints are found next to it.

    Every key is optional: a run with learn_k=False has no k_trace, a run
    with Train_model=False has no checkpoints at all, and in both cases the
    corresponding keys are simply absent rather than NaN-filled.
    """
    stem = os.path.splitext(save_model_path)[0]
    phys_path, hist_path = stem + "_physics.pt", stem + "_loss_history.pt"
    out = {}

    ph = _load(phys_path)
    if ph is None:
        print(f"  [run_diagnostics] MISSING {os.path.basename(phys_path)}"
              " -> no recovered_mu / recovered_k_over_m")
    else:
        for key in ("recovered_mu", "recovered_k_over_m"):
            if key in ph:
                out[key] = float(ph[key])

    hi = _load(hist_path)
    if hi is None:
        print(f"  [run_diagnostics] MISSING {os.path.basename(hist_path)}"
              " -> no loss curves, no mu_trace, no k_trace")
    else:
        tv = hi.get("train_loss_values") or []
        if tv:
            tail = np.asarray(tv[-last_n:], dtype=float)
            tail = tail[np.isfinite(tail)]
            if tail.size:
                # The denominator of gamma = 0.03 * L_pred / raw. Averaged,
                # because the per-epoch value bounces by ~10%.
                out["final_train_loss"] = float(tail.mean())
                out["final_train_loss_std"] = (float(tail.std(ddof=1))
                                               if tail.size > 1 else 0.0)
                out["final_train_loss_n"] = float(tail.size)
        for src, dst in (("best_val_loss", "best_val_loss"),
                         ("best_val_epoch", "best_val_epoch"),
                         ("global_step", "total_optimizer_steps")):
            if hi.get(src) is not None:
                out[dst] = float(hi[src])

        # Drift of each recovered parameter (its endpoint is recovered_mu /
        # recovered_k_over_m above). The drift is what distinguishes
        # "identified" from "never moved" - a monotone descent that has not
        # arrived is a different result from a value that parked.
        for name, key in (("mu", "mu_trace"), ("k", "k_trace")):
            tr = _trace_arrays(hi.get(key))
            if tr is None:
                # Say WHY rather than leaving a silent NaN in the CSV: an
                # absent key and an empty list mean different things.
                print(f"  [run_diagnostics] no usable '{key}' "
                      + ("(key absent - trainer did not save it)"
                         if key not in hi else "(present but empty - "
                         "was the parameter learnable?)")
                      + f" -> no {name} drift")
                continue
            ep, val = tr
            out[f"{name}_init"] = float(val[0])
            out[f"{name}_drift"] = float(val[-1] - val[0])
            # movement over the last 10% of training: ~0 means converged
            n_tail = max(2, len(val) // 10)
            out[f"{name}_tail_slope_per_1k_ep"] = float(
                np.polyfit(ep[-n_tail:], val[-n_tail:], 1)[0] * 1000.0)
    return out
