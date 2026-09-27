"""
evaluate_force_model.py

Evaluation for the force-based GNS. Two jobs:

  1. ROLLOUT METRICS - same numbers as the acceleration pipeline (center error
     per block width, angle error, per-phase split), so the Stage-1 parity
     comparison is apples to apples.

  2. WRENCH-DECOMPOSITION VALIDATION - the payoff of the force representation.
     If the dataset folder was labeled by add_wrench_labels.py, the predicted
     contact and fluid wrenches are compared per frame against MuJoCo's ground
     truth. The model was trained on POSITIONS ONLY, so agreement here means it
     recovered a force split it was never shown. Also fits the drag coefficient
     implied by the predicted fluid force on airborne frames and compares it to
     the calibrated k/m - the "recovered the physical law" check.

ALIGNMENT: the k-th force prediction of a rollout comes from the window ending
at original frame t = 2h + k and drives interval t -> t+1, so it pairs with
wrench label index (2h + k). Average label force over that interval = J[t]/DT.
"""

import os
import csv
import numpy as np

from impulse_diagnostic import compute_impulse_split, aggregate
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import wall
from train_force_gns import build_force_dataset, rollout_force_batched, load_trained_model


def evaluate_force_model(model_folder, data_folder, test_indices, weights_only, unscale):
    """Roll out the test set, print metrics, and (if wrench labels are present)
    validate the force decomposition. Returns the metrics dict. Figures and the
    per-trajectory CSV are written to <model_folder>/force_eval_*.
    """
    out_prefix = os.path.join(model_folder, "force_eval")
    # ======================================================================
    # Load model + config (force_cfg in the norms file is the single source of
    # truth for dt / gravity / gates / drag flag - no chance of eval mismatch)
    # ======================================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Floor = wall.wall(center_position=(0, 0, 0), size=(2, 2), normal=(0, 0, 1))

    model, norms, cfg, rest_nodes, edge_index, _, k_over_m = load_trained_model(
        model_folder, device)
    print("Loaded force_cfg:", {k: v for k, v in cfg.items() if k != "scale_vec"})
    print(f"Rolling out with the checkpoint's k/m = {k_over_m:.5f}")

    # ======================================================================
    # Rollout the test set
    # ======================================================================
    trajs, _ = build_force_dataset(test_indices, data_folder,
                                   weights_only=weights_only, unscale_data=unscale)
    for d in trajs:
        d["edge_index"] = edge_index

    h, dt, mass, g = cfg["h"], cfg["dt"], cfg["mass"], cfg["gravity"]
    g_step = torch.tensor([0.0, 0.0, -g]) * dt * dt

    center, angle, forces, (per_traj, _, _, lengths) = rollout_force_batched(
        model, trajs, Floor, h, rest_nodes,
        norms["x_mean"], norms["x_std"], norms["e_mean"], norms["e_std"],
        cfg["scale_vec"], cfg["ang_scale_vec"], g_step, dt, device,
        use_wind=cfg["use_wind"], use_drag_baseline=cfg["use_drag_baseline"],
        k_over_m=k_over_m, contact_d0=cfg["contact_d0"],
        contact_tau=cfg["contact_tau"], mass=mass,
        return_forces=True, return_per_traj=True)

    print("\n" + "=" * 70)
    print(f"ROLLOUT METRICS ({len(per_traj)} test trajectories)")
    print("=" * 70)


    def _mean(key):
        vals = [float(m[key]) for m in per_traj if np.isfinite(float(m[key]))]
        return float(np.mean(vals)) if vals else float("nan")

    def _std(key):
        vals = [float(m[key]) for m in per_traj if np.isfinite(float(m[key]))]
        return float(np.std(vals)) if vals else float("nan")


    print(f"Center error (/width):  {center:.4f}")
    print(f"Angle error (deg):      {angle:.2f}")
    print(f"Phase center (air/contact/settled): "
          f"{_mean('center_error_airborne'):.4f} / {_mean('center_error_contact'):.4f} / "
          f"{_mean('center_error_settled'):.4f}")
    print(f"Phase angle  (air/contact/settled): "
          f"{_mean('angle_error_airborne'):.2f} / {_mean('angle_error_contact'):.2f} / "
          f"{_mean('angle_error_settled'):.2f}")

    # ======================================================================
    # Wrench validation against the logged ground truth (if present)
    # ======================================================================
    mg = mass * g
    rows = []
    imp_rows = []          # impulse-vs-force split, one dict per trajectory
    pooled = {"Fc_pred": [], "Fc_true": [], "Ff_pred": [], "Ff_true": [],
              "Tc_pred": [], "Tc_true": [], "Tf_pred": [], "Tf_true": [],
              "phase": []}
    drag_num = drag_den = 0.0

    for b, idx in enumerate(test_indices):
        raw = torch.load(os.path.join(data_folder, f"{idx}.pt"), weights_only=False)
        row = dict(idx=idx, center=float(per_traj[b]["center_error"]),
                   angle=float(per_traj[b]["angle_error_deg"]))
        if len(raw) > 4 and isinstance(raw[4], dict) and "J_contact" in raw[4]:
            wr = raw[4]
            DT_lbl = float(wr["dt_record"])
            T_b = lengths[b]
            n_pred = T_b - 1 - 2 * h
            if n_pred <= 0:
                rows.append(row); continue

            Fc_p = forces["F_contact"][b, :n_pred].numpy()
            Ff_p = forces["F_fluid"][b, :n_pred].numpy()
            Tc_p = forces["tau_contact"][b, :n_pred].numpy()
            Tf_p = forces["tau_fluid"][b, :n_pred].numpy()
            Fc_t = (wr["J_contact"].numpy() / DT_lbl)[2 * h: 2 * h + n_pred]
            Ff_t = (wr["J_fluid"].numpy() / DT_lbl)[2 * h: 2 * h + n_pred]
            Tc_t = (wr["tau_contact"].numpy() / DT_lbl)[2 * h: 2 * h + n_pred]
            Tf_t = (wr["tau_fluid"].numpy() / DT_lbl)[2 * h: 2 * h + n_pred]

            # ---- IMPULSE vs FORCE split -------------------------------
            # force_contact_err compares frame by frame. An impact lasts one
            # or two frames, so an impulse delivered one frame late scores
            # 2||J|| of per-frame error while the TOTAL impulse - and
            # therefore the velocity change and the trajectory - is exactly
            # right. Summing over each contact episode makes the comparison
            # blind to WHEN the impulse landed; the gap between the two says
            # how much of the 46% is timing rather than force.
            #
            # Episodes are contiguous runs where the TRUE contact force is
            # non-negligible, so a trajectory that bounces twice contributes
            # two episodes rather than one pooled block in which an early
            # error could cancel a late one.
            Jc_p = Fc_p * DT_lbl
            Jc_t = Fc_t * DT_lbl
            active = np.linalg.norm(Fc_t, axis=1) > 0.05 * mg
            imp_rows.append(compute_impulse_split(Jc_p, Jc_t, active))

            tc = int(per_traj[b]["t_contact"])            # from-h indexing
            ts = int(per_traj[b]["t_settle"])
            # force prediction k sits at from-h index (h + k)
            k_idx = np.arange(n_pred) + h
            phase = np.where(k_idx < tc, 0, np.where(k_idx < ts, 1, 2))

            pooled["Fc_pred"].append(Fc_p); pooled["Fc_true"].append(Fc_t)
            pooled["Ff_pred"].append(Ff_p); pooled["Ff_true"].append(Ff_t)
            pooled["Tc_pred"].append(Tc_p); pooled["Tc_true"].append(Tc_t)
            pooled["Tf_pred"].append(Tf_p); pooled["Tf_true"].append(Tf_t)
            pooled["phase"].append(phase)

            row.update(
                Fc_mae_over_mg=float(np.linalg.norm(Fc_p - Fc_t, axis=1).mean() / mg),
                Ff_mae_over_mg=float(np.linalg.norm(Ff_p - Ff_t, axis=1).mean() / mg),
            )

            # drag-coefficient recovery from PREDICTED fluid force, airborne only
            wind = trajs[b]["wind"].numpy()
            if np.linalg.norm(wind) > 1e-6:
                com = trajs[b]["com"].numpy()
                v = (com[1:] - com[:-1]) / DT_lbl                     # m/s at interval t
                for k in range(n_pred):
                    if k_idx[k] >= tc - 1:
                        break
                    t_orig = 2 * h + k
                    u = wind - v[t_orig]
                    p = np.linalg.norm(u) * u
                    a_f = Ff_p[k] / mass
                    drag_num += float(np.dot(a_f[:2], p[:2]))
                    drag_den += float(np.dot(p[:2], p[:2]))
        rows.append(row)

    wrench_metrics = {}
    if imp_rows:
        wrench_metrics.update(aggregate(imp_rows))
        tf = wrench_metrics.get("impulse_timing_fraction")
        if tf is not None:
            print("\n" + "=" * 70)
            print("CONTACT ERROR: TIMING vs MAGNITUDE")
            print("=" * 70)
            print(f"  per-frame error / true impulse   "
                  f"{wrench_metrics['impulse_E_frame_over_signal']:.3f}")
            print(f"  total-impulse error / true       "
                  f"{wrench_metrics['impulse_E_total_over_signal']:.3f}")
            print(f"  timing fraction                  {tf:.3f}"
                  "   (1 = pure smearing, 0 = genuine force error)")
            print("=" * 70)

    have_labels = len(pooled["Fc_pred"]) > 0
    if have_labels:
        Fc_p = np.concatenate(pooled["Fc_pred"]); Fc_t = np.concatenate(pooled["Fc_true"])
        Ff_p = np.concatenate(pooled["Ff_pred"]); Ff_t = np.concatenate(pooled["Ff_true"])
        ph = np.concatenate(pooled["phase"])

        print("\n" + "=" * 70)
        print("WRENCH DECOMPOSITION vs MuJoCo ground truth")
        print("(model trained on positions only - it was never shown these forces)")
        print("=" * 70)
        names = ["airborne", "contact", "settled"]
        # TRUE magnitudes are printed alongside the errors: an error of 0.17 mg
        # means something completely different when the true force is 1.0 mg
        # (17% off) than when the true force is 0.001 mg (invented from nothing).
        print(f"{'phase':>10} {'n':>7} {'|dF_con|/mg':>12} {'true|F_con|':>12}"
              f" {'|dF_fld|/mg':>12} {'true|F_fld|':>12}")
        for p in (0, 1, 2):
            sel = ph == p
            if sel.sum() == 0:
                continue
            ec = np.linalg.norm(Fc_p[sel] - Fc_t[sel], axis=1).mean() / mg
            ef = np.linalg.norm(Ff_p[sel] - Ff_t[sel], axis=1).mean() / mg
            tc_ = np.linalg.norm(Fc_t[sel], axis=1).mean() / mg
            tf_ = np.linalg.norm(Ff_t[sel], axis=1).mean() / mg
            print(f"{names[p]:>10} {int(sel.sum()):>7} {ec:>12.4f} {tc_:>12.4f}"
                  f" {ef:>12.4f} {tf_:>12.4f}")
            wrench_metrics[f"force_contact_err_{names[p]}"] = float(ec)
            wrench_metrics[f"force_contact_true_{names[p]}"] = float(tc_)
            wrench_metrics[f"force_fluid_err_{names[p]}"] = float(ef)
            wrench_metrics[f"force_fluid_true_{names[p]}"] = float(tf_)

        def _fit_stats(a, b):
            """R^2 plus the ratio that actually reads well when the true signal
            is small. R^2 = 1 - SSE/SS_var, so a huge negative number just means
            SSE >> SS_var - unreadable. sqrt(SSE/SS_var) says the same thing as
            'the prediction error is N times the true signal's own RMS', which
            is interpretable at any signal scale."""
            var = float(((b - b.mean(0)) ** 2).sum())
            sse = float(((a - b) ** 2).sum())
            n = b.size
            rms_true = float(np.sqrt(var / n))          # RMS deviation of truth
            rms_err = float(np.sqrt(sse / n))
            if var <= 0:
                return None, rms_true, rms_err, float("inf")
            return 1.0 - sse / var, rms_true, rms_err, float(np.sqrt(sse / var))

        print()
        for nm, P, Tt in (("contact", Fc_p, Fc_t), ("fluid  ", Ff_p, Ff_t)):
            r2, rms_true, rms_err, ratio = _fit_stats(P, Tt)
            print(f"{nm} force:  RMS(true) = {rms_true/mg:.5f} mg   "
                  f"RMS(error) = {rms_err/mg:.5f} mg   "
                  f"error/signal = {ratio:.1f}x"
                  + (f"   R^2 = {r2:.3f}" if r2 is not None and r2 > -1 else ""))
            if r2 is not None and r2 <= -1:
                print(f"{'':>13}-> prediction is {ratio:.0f}x the true signal's own RMS; "
                      f"R^2 ({r2:.0f}) is not informative at this scale")
            tag = nm.strip()
            wrench_metrics[f"force_{tag}_rms_true"] = float(rms_true / mg)
            wrench_metrics[f"force_{tag}_rms_err"] = float(rms_err / mg)
            wrench_metrics[f"force_{tag}_err_over_signal"] = float(ratio)
            if r2 is not None:
                wrench_metrics[f"force_{tag}_r2"] = float(r2)

        if drag_den > 0:
            k_rec = drag_num / drag_den
            print(f"\nDrag recovery: implied k/m from PREDICTED fluid force = {k_rec:.5f} 1/m"
                  f"   (calibrated reference: {cfg['k_over_m']:.5f}, "
                  f"model's k/m: {k_over_m:.5f})")

        # ---------------- figure ----------------
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        b0 = 0                                  # first labeled trajectory as the example
        n0 = lengths[b0] - 1 - 2 * h
        t_ax = np.arange(n0)
        axes[0, 0].plot(t_ax, pooled["Fc_true"][0][:, 2], "k-", lw=1.5, label="MuJoCo")
        axes[0, 0].plot(t_ax, pooled["Fc_pred"][0][:, 2], "C1--", lw=1.2, label="predicted")
        axes[0, 0].axhline(mg, color="gray", ls=":", lw=1, label="m g")
        axes[0, 0].set_title(f"contact force z - traj {list(test_indices)[b0]}")
        axes[0, 0].set_xlabel("prediction step"); axes[0, 0].set_ylabel("N"); axes[0, 0].legend()

        axes[0, 1].plot(t_ax, np.linalg.norm(pooled["Ff_true"][0][:, :2], axis=1), "k-",
                        lw=1.5, label="MuJoCo")
        axes[0, 1].plot(t_ax, np.linalg.norm(pooled["Ff_pred"][0][:, :2], axis=1), "C0--",
                        lw=1.2, label="predicted")
        axes[0, 1].set_title("fluid force, horizontal magnitude")
        axes[0, 1].set_xlabel("prediction step"); axes[0, 1].set_ylabel("N"); axes[0, 1].legend()

        for ax, P, Tt, nm in ((axes[1, 0], Fc_p, Fc_t, "contact"),
                              (axes[1, 1], Ff_p, Ff_t, "fluid")):
            ax.scatter(Tt.ravel(), P.ravel(), s=3, alpha=0.15)
            lim = np.percentile(np.abs(Tt), 99.5)
            ax.plot([-lim, lim], [-lim, lim], "r-", lw=1)
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
            ax.set_title(f"{nm} force: predicted vs true (all comps)")
            ax.set_xlabel("true (N)"); ax.set_ylabel("predicted (N)")

        for ax in axes.flat:
            ax.grid(alpha=0.3)
        fig.suptitle(f"Force decomposition validation - {os.path.basename(data_folder)}")
        fig.tight_layout()
        fig.savefig(out_prefix + "_wrench.png", dpi=150)
        print(f"\nSaved figure to {out_prefix}_wrench.png")
    else:
        print("\n(no wrench labels in this folder - run add_wrench_labels.py to "
              "enable the decomposition validation)")

    with open(out_prefix + "_per_traj.csv", "w", newline="") as f:
        keys = sorted({k for r in rows for k in r})
        w = csv.DictWriter(f, fieldnames=keys, restval="")
        w.writeheader(); w.writerows(rows)
    print(f"Saved per-trajectory CSV to {out_prefix}_per_traj.csv")

    # phase_center / phase_angle are split into one CSV column per phase by
    # run_report.save_run_report.
    out = dict(
        center_error=float(center),
        angle_error_deg=float(angle),
        floor_penetration=_mean("floor_penetration"),
        center_error_std=_std("center_error"),
        angle_error_std=_std("angle_error_deg"),
        floor_penetration_std=_std("floor_penetration"),
        phase_center=[_mean("center_error_airborne"),
                      _mean("center_error_contact"),
                      _mean("center_error_settled")],
        phase_angle=[_mean("angle_error_airborne"),
                     _mean("angle_error_contact"),
                     _mean("angle_error_settled")],
        n_test=float(len(per_traj)),
        have_wrench_labels=float(bool(have_labels)),
    )
    out.update(wrench_metrics)
    return out
