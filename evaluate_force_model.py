"""
evaluate_force_model.py

NEW FILE - evaluation for the force-based GNS. Two jobs:

  1. ROLLOUT METRICS - same numbers as the acceleration pipeline (center error
     per block width, angle error, per-phase split), so the Stage-1 parity
     comparison is apples to apples. compute_metrics is imported unchanged.

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

Set the CONFIG block, then:  python evaluate_force_model.py
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
from force_gns import ForceGNSModel
from train_force_gns import build_force_dataset, rollout_force_batched
from generate_node_states import mesh_cube_surface, knn_adjacency, BLOCK_HALF_WIDTH


def evaluate_force_model(model_folder, data_folder, test_indices,
                         model_prefix=None, weights_only=False, unscale=False,
                         out_prefix=None, checkpoint="best"):
    """Roll out the test set, print metrics, and (if wrench labels are present)
    validate the force decomposition. Returns the metrics dict so the caller
    can log or compare. Safe to call right after training in the same script.

    checkpoint: "best" or "final" - which saved model to evaluate.
    """
    MODEL_FOLDER, DATA_FOLDER, TEST_INDICES = model_folder, data_folder, test_indices
    MODEL_PREFIX, WEIGHTS_ONLY, UNSCALE = model_prefix, weights_only, unscale
    OUT_PREFIX = out_prefix or os.path.join(model_folder, "force_eval")
    # ======================================================================
    # Load model + config (force_cfg in the norms file is the single source of
    # truth for dt / gravity / gates / drag flag - no chance of eval mismatch)
    # ======================================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Floor = wall.wall(center_position=(0, 0, 0), size=(2, 2), normal=(0, 0, 1))

    if MODEL_PREFIX is None:
        cands = [f[:-len("_norms.pt")] for f in os.listdir(MODEL_FOLDER)
                 if f.endswith("_norms.pt")]
        assert len(cands) == 1, f"set MODEL_PREFIX explicitly, found: {cands}"
        MODEL_PREFIX = cands[0]

    norms = torch.load(os.path.join(MODEL_FOLDER, MODEL_PREFIX + "_norms.pt"),
                       weights_only=False)
    cfg = norms["force_cfg"]
    print("Loaded force_cfg:", {k: v for k, v in cfg.items() if k != "scale_vec"})

    rest_nodes = torch.tensor(
        mesh_cube_surface(BLOCK_HALF_WIDTH * 2, cfg["nodes_per_edge"]), dtype=torch.float32)
    edge_index = torch.tensor(
        knn_adjacency(rest_nodes.numpy(), k=cfg["nearest_neighbors"]), dtype=torch.long)

    model = ForceGNSModel(norms["x_mean"].shape[0], norms["e_mean"].shape[0],
                          latent_dim=cfg["latent_dim"], L=cfg["L"], K=cfg["K"])
    ckpt_name = MODEL_PREFIX + ("_best_model.pt" if checkpoint == "best" else "_final.pt")
    sd = torch.load(os.path.join(MODEL_FOLDER, ckpt_name),
                    map_location=device, weights_only=False)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.to(device).eval()

    # Recovered friction coefficient, written by training to <prefix>_physics.pt.
    # Captured here (not just printed) so it flows into the returned metrics ->
    # run_report -> the master CSV, alongside every other headline number.
    recovered_mu = None
    phys_path = os.path.join(MODEL_FOLDER, MODEL_PREFIX + "_physics.pt")
    if os.path.exists(phys_path):
        pinfo = torch.load(phys_path, map_location="cpu", weights_only=False)
        recovered_mu = float(pinfo.get("recovered_mu", float("nan")))
        if pinfo.get("mu_mode") == "learnable":
            print(f"Recovered friction coefficient mu = {recovered_mu:.4f}"
                  f"   (replica ground truth: 0.198)")

    # ======================================================================
    # Rollout the test set
    # ======================================================================
    trajs, _ = build_force_dataset(TEST_INDICES, DATA_FOLDER,
                                   weights_only=WEIGHTS_ONLY, unscale_data=UNSCALE)
    for d in trajs:
        d["edge_index"] = edge_index

    h, dt, mass, g = cfg["h"], cfg["dt"], cfg["mass"], cfg["gravity"]
    g_step = torch.tensor([0.0, 0.0, -g]) * dt * dt

    center, angle, forces, (per_traj, _, _, lengths) = rollout_force_batched(
        model, trajs, Floor, h, rest_nodes,
        norms["x_mean"], norms["x_std"], norms["e_mean"], norms["e_std"],
        cfg["scale_vec"], cfg["ang_scale_vec"], g_step, dt, device,
        use_wind=cfg["use_wind"], use_drag_baseline=cfg["use_drag_baseline"],
        k_over_m=cfg["k_over_m"], contact_d0=cfg["contact_d0"],
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

    for b, idx in enumerate(TEST_INDICES):
        raw = torch.load(os.path.join(DATA_FOLDER, f"{idx}.pt"), weights_only=False)
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
                  f"   (calibrated reference: {cfg['k_over_m']:.5f})")

        # ---------------- figure ----------------
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        b0 = 0                                  # first labeled trajectory as the example
        n0 = lengths[b0] - 1 - 2 * h
        t_ax = np.arange(n0)
        axes[0, 0].plot(t_ax, pooled["Fc_true"][0][:, 2], "k-", lw=1.5, label="MuJoCo")
        axes[0, 0].plot(t_ax, pooled["Fc_pred"][0][:, 2], "C1--", lw=1.2, label="predicted")
        axes[0, 0].axhline(mg, color="gray", ls=":", lw=1, label="m g")
        axes[0, 0].set_title(f"contact force z - traj {list(TEST_INDICES)[b0]}")
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
        fig.suptitle(f"Force decomposition validation - {os.path.basename(DATA_FOLDER)}")
        fig.tight_layout()
        fig.savefig(OUT_PREFIX + "_wrench.png", dpi=150)
        print(f"\nSaved figure to {OUT_PREFIX}_wrench.png")
    else:
        print("\n(no wrench labels in this folder - run add_wrench_labels.py to "
              "enable the decomposition validation)")

    with open(OUT_PREFIX + "_per_traj.csv", "w", newline="") as f:
        keys = sorted({k for r in rows for k in r})
        w = csv.DictWriter(f, fieldnames=keys, restval="")
        w.writeheader(); w.writerows(rows)
    print(f"Saved per-trajectory CSV to {OUT_PREFIX}_per_traj.csv")

    # Key names deliberately mirror evaluate_model() in evaluate_metrics.py so
    # run_report.py's formatter and comparison table work on force runs too.
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
    if recovered_mu is not None:
        out["recovered_mu"] = recovered_mu
    out.update(wrench_metrics)

    # ---- checkpoint-derived values, folded into the RETURNED metrics ------
    # recovered_k_over_m, the mu/k traces and the converged prediction loss
    # live in _physics.pt and _loss_history.pt, which only the trainer writes.
    # Reading them HERE rather than in the run file means they reach the CSV
    # for every experiment: evaluate_force_model is the one function every run
    # passes through, whereas the run file is copied per experiment
    # (Test_37.py, Test_43.py, ...) and only one copy ever carries the call.
    # They land as metrics.* instead of settings.*; the values are identical.
    try:
        import glob as _glob
        from run_diagnostics import collect_run_diagnostics
        # Per-epoch diagnostics too. physics_losses.DIAG_HISTORY is module
        # state in THIS process, so the evaluator can read what training just
        # accumulated - no need for the run file to forward it. Doing both
        # here means every diagnostic lands under metrics.* for every run,
        # regardless of which copy of the run script launched it.
        try:
            from physics_losses import summarize_diagnostics
            _diag = summarize_diagnostics(last_n=20)
            if _diag:
                out.update({k: float(v) for k, v in _diag.items()
                            if isinstance(v, (int, float))
                            and not isinstance(v, bool)})
                print(f"  per-epoch diagnostics: {len(_diag)} value(s) added")
        except Exception as _e:
            print(f"  [evaluate] per-epoch diagnostics unavailable: {_e}")

        _hits = sorted(_glob.glob(os.path.join(model_folder, "*_physics.pt")))
        if _hits:
            _stem = _hits[0][: -len("_physics.pt")] + ".pt"
            _extra = collect_run_diagnostics(_stem)
            # NUMERIC ONLY. run_report.py does float(v) on every metric, so a
            # string here (mu_mode / k_mode are "learnable" / "frozen" /
            # "fixed") raises ValueError and kills the CSV write AFTER a
            # completed 10k-epoch run. Those two are recoverable from
            # settings.learn_mu / settings.learn_k anyway.
            _extra = {k: float(v) for k, v in _extra.items()
                      if isinstance(v, (int, float)) and not isinstance(v, bool)}
            if _extra:
                print(f"  checkpoint diagnostics: {len(_extra)} numeric "
                      "value(s) added to the run report")
                out.update(_extra)
    except Exception as _e:                # never fail an eval over reporting
        print(f"  [evaluate] checkpoint diagnostics unavailable: {_e}")

    return out


# ======================================================================
# Standalone use:  python evaluate_force_model.py
# ======================================================================
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))

    DATA_FOLDER = os.path.join(script_dir, "data/mojoco_paper_replica_0_wind")
    MODEL_FOLDER = os.path.join(script_dir, "models/force_stage1")
    MODEL_PREFIX = None            # None -> auto-detect "<n>_force_gns_model" in MODEL_FOLDER
    TEST_INDICES = range(454, 568)
    WEIGHTS_ONLY = False
    UNSCALE = False

    OUT_PREFIX = os.path.join(MODEL_FOLDER, "force_eval")

    evaluate_force_model(MODEL_FOLDER, DATA_FOLDER, TEST_INDICES,
                         model_prefix=MODEL_PREFIX, weights_only=WEIGHTS_ONLY,
                         unscale=UNSCALE, out_prefix=OUT_PREFIX)
