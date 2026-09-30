"""Collect post-training diagnostics from saved model checkpoints."""

import os

import numpy as np
import torch


# Load a checkpoint from disk and return None when it is unavailable.
def _load(path):
    if not os.path.exists(path):
        return None
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"  [run_diagnostics] could not read {os.path.basename(path)}: {e}")
        return None


# Convert a saved [(epoch, value), ...] trace into separate arrays.
def _trace_arrays(trace):
    if not trace:
        return None
    a = np.asarray(trace, dtype=float)
    if a.ndim != 2 or a.shape[1] != 2 or a.shape[0] == 0:
        return None
    return a[:, 0], a[:, 1]


# Read all available diagnostics associated with one saved model.
def collect_run_diagnostics(save_model_path, last_n=20):

    # Checkpoint files use the model path as their filename stem.
    stem = os.path.splitext(save_model_path)[0]
    phys_path, hist_path = stem + "_physics.pt", stem + "_loss_history.pt"
    out = {}

    # Collect recovered physical parameters from the physics checkpoint.
    ph = _load(phys_path)
    if ph is None:
        print(f"  [run_diagnostics] MISSING {os.path.basename(phys_path)}"
              " -> no recovered_mu / recovered_k_over_m")
    else:
        # Extract the recovered physical parameters if they exist.
        for key in ("recovered_mu", "recovered_k_over_m"):
            if key in ph:
                out[key] = float(ph[key])

    # Collect losses, optimizer progress, and learned-parameter traces.
    hi = _load(hist_path)
    if hi is None:
        print(f"  [run_diagnostics] MISSING {os.path.basename(hist_path)}"
              " -> no loss curves, no mu_trace, no k_trace")
    else:
        # Average the final training-loss values to reduce epoch-to-epoch noise.
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
        # Copy scalar training-summary values into the report metrics.
        for src, dst in (("best_val_loss", "best_val_loss"),
                         ("best_val_epoch", "best_val_epoch"),
                         ("global_step", "total_optimizer_steps")):
            if hi.get(src) is not None:
                out[dst] = float(hi[src])

        # Measure how much each learned physical parameter changed during training.
        # Its endpoint is recovered_mu / recovered_k_over_m above. The drift is
        # what distinguishes "identified" from "never moved" - a monotone descent
        # that has not arrived is a different result from a value that parked.
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
            # Estimate recent movement; a slope near zero suggests convergence.
            n_tail = max(2, len(val) // 10)
            out[f"{name}_tail_slope_per_1k_ep"] = float(
                np.polyfit(ep[-n_tail:], val[-n_tail:], 1)[0] * 1000.0)
    return out
