"""
visualize_force_model.py

3D animation of a force-model rollout (red predicted cube, blue ground-truth
cube, wireframe edges, optional GIF) that also draws the forces the model is
predicting:

  * TWO arrows per node in contact:
      - NORMAL    (green)  along the wall normal, >= 0 by construction
      - TANGENTIAL(orange) in the floor plane - this is the friction force
    Nodes the contact gate has switched off draw nothing, so you can watch
    corners turn on and off as the cube tumbles.
  * ONE arrow at the COM (magenta) for the fluid force (learned residual plus
    the analytic drag baseline if that was enabled). No torque arrow.

ARROW SCALE IS PHYSICAL, NOT PER-FRAME NORMALIZED. A force equal to the cube's
weight (m g) is drawn mg_arrow_widths block-widths long, so arrow length means
the same thing in every frame and every trajectory (see arrow_len for how
larger forces are compressed). The HUD prints the summed normal force as a
multiple of m g: when the cube comes to rest that number should sit near 1.0,
which is a free sanity check on the model that costs nothing to look at.

Rollout only - the cube follows the model's own predictions and the forces are
whatever it predicts as it drifts.

Called by run_force_multi_step.py (GIFs) and make_filmstrip.py (vector frames
for figures).
"""

import os
import numpy as np
import torch
import matplotlib
# Headless-safe (HPC compute nodes have no display). Must precede pyplot import.
if not os.environ.get("DISPLAY") and os.name != "nt":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

import wall
from generate_node_states import BLOCK_WIDTH
from train_force_gns import build_force_dataset, rollout_force_batched, load_trained_model


# ======================================================================
# Pick the frames that are worth a filmstrip panel
# ======================================================================
def auto_frames(f_norm, f_tang, MG, h, L, t_contact, t_settle, n_panels=6,
                min_active=4, min_frac=0.02):
    """Choose frames by EVENT rather than by even spacing.

    Forces only exist for frames h .. h + n_steps - 1. Frame L-1 usually has NO
    force index (k = L-1-h lands one past the end), which is why a naive
    "settled" pick came out with no arrows on it even though the GIF clearly
    shows normal forces at rest. Everything below is clamped into the valid
    range so every exported panel has forces.
    """
    fn_nodes = f_norm.numpy()                       # (n_steps, N, 3)
    Fn = np.linalg.norm(fn_nodes.sum(axis=1), axis=-1) / MG   # summed, in m g
    Ft = np.linalg.norm(f_tang.numpy().sum(axis=1), axis=-1) / MG
    n_steps = len(Fn)

    # How many nodes are actually loaded in each step. A sliding panel that
    # catches a micro-bump with two corners loaded reads as a bad prediction;
    # we want a frame where the cube is flat on its face.
    n_active = (np.linalg.norm(fn_nodes, axis=-1) > min_frac * MG).sum(axis=1)

    first_forced = h                       # earliest frame that has forces
    last_forced = h + n_steps - 1          # latest frame that has forces

    def clamp(frame):
        return int(np.clip(frame, first_forced, last_forced))

    def to_frame(k):
        return clamp(k + h)

    # (priority, label, frame) - lower priority number is dropped last
    picks = []

    if t_contact > h + 2:
        picks.append((3, "mid-flight", clamp(h + (t_contact - h) // 2)))
    if t_contact - 1 > h:
        picks.append((2, "pre-impact", clamp(t_contact - 1)))

    # contact episodes: rising edges of the normal force
    on = Fn > 0.15
    edges = np.flatnonzero(np.diff(on.astype(int)) == 1) + 1
    if on.size and on[0]:
        edges = np.r_[0, edges]

    for e_i, start in enumerate(edges[:2]):          # first two impacts
        stop = edges[e_i + 1] if e_i + 1 < len(edges) else n_steps
        seg = Fn[start:stop]
        if seg.size:
            picks.append((1, f"impact {e_i + 1} peak",
                          to_frame(start + int(np.argmax(seg)))))

    # SLIDING: inside the window between the second contact and settling,
    # pick the frame with the MOST loaded nodes (ties broken toward the
    # middle of the window, and toward larger friction). Taking the midpoint
    # blindly often lands on a micro-bump with only two corners in contact.
    if len(edges) >= 2:
        lo_k, hi_k = int(edges[1]), int(np.clip(t_settle - h, 0, n_steps - 1))
    else:
        lo_k, hi_k = (int(np.clip(t_contact - h, 0, n_steps - 1)),
                      int(np.clip(t_settle - h, 0, n_steps - 1)))

    if hi_k > lo_k + 1:
        ks = np.arange(lo_k, hi_k)
        mid = (lo_k + hi_k) / 2.0
        best = n_active[ks].max()
        if best < min_active:
            print(f"    (sliding: only {best} nodes ever loaded in the slide "
                  f"window, wanted {min_active})")
        ok = ks[n_active[ks] >= min(min_active, best)]
        # among qualifying frames prefer strong friction, then centrality
        score = Ft[ok] - 0.002 * np.abs(ok - mid)
        k_slide = int(ok[int(np.argmax(score))])
        picks.append((1, "sliding", to_frame(k_slide)))
        print(f"    (sliding frame has {n_active[k_slide]} loaded nodes)")

    if Ft.size:
        picks.append((2, "max friction", to_frame(int(np.argmax(Ft)))))

    picks.append((1, "settled", clamp(t_settle + 5)))

    # de-duplicate by frame, keeping the highest-priority label
    picks.sort(key=lambda t: (t[2], t[0]))
    unique, seen = [], set()
    for prio, label, f in picks:
        if f not in seen:
            seen.add(f)
            unique.append((prio, label, f))

    # if we have too many, drop the lowest-priority ones first
    if len(unique) > n_panels:
        unique.sort(key=lambda t: (t[0], t[2]))
        unique = unique[:n_panels]

    unique.sort(key=lambda t: t[2])
    for prio, label, f in unique:
        print(f"    frame {f:3d}  {label}")
    return [f for _, _, f in unique]


def visualize_force_rollout(model_folder, data_folder, trajectory,
                            model_prefix=None, save_path=None, show=False,
                            weights_only=False, unscale=False, interval=50,
                            mg_arrow_widths=1.0, min_arrow_frac=0.002,
                            draw_floor=True, draw_ground_truth=True,
                            checkpoint="best",
                            # per-channel display gain on top of the physical scale
                            normal_gain=1.0, tangent_gain=8.0, fluid_gain=1.0,
                            # vector frame export (filmstrip)
                            save_frames=None, n_panels=6, frame_dir=None,
                            frame_format="pdf", frame_clean=True,
                            elev=18, azim=-62, make_gif=True,
                            # arrow length mapping (see arrow_len)
                            arrow_mode="log", max_arrow_widths=2.2,
                            # camera zoom, >1 is closer
                            zoom=1.13,
                            # loaded nodes required in the auto-picked sliding frame
                            slide_min_nodes=4):
    """Roll out ONE trajectory and animate it with per-node contact-force
    arrows (normal + tangential) and a COM fluid-force arrow.

    save_frames: None (no frames), "auto" (auto_frames picks the event
                 frames), or a list of frame numbers; each is written as a
                 separate vector file for a filmstrip figure.

    Returns {"gif", "frames", "t_contact", "t_settle", "n_frames"}.
    show=False is the default so this is safe to call from a batch job on a
    compute node."""
    if save_path is None:
        save_path = os.path.join(model_folder, f"force_rollout_{trajectory}.gif")
    C_NORMAL, C_TANGENT, C_FLUID = "tab:green", "tab:orange", "magenta"

    # ======================================================================
    # Load model + config
    # ======================================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Floor = wall.wall(center_position=(0, 0, 0), size=(2, 2), normal=(0, 0, 1))

    model, norms, cfg, rest_nodes, edge_index, model_prefix, k_over_m = load_trained_model(
        model_folder, device, prefix=model_prefix, checkpoint=checkpoint)
    h, dt, mass, g = cfg["h"], cfg["dt"], cfg["mass"], cfg["gravity"]
    MG = mass * g
    print(f"Model: {model_prefix}   dt={dt:.6f}s  m={mass}kg  g={g}  "
          f"drag_baseline={cfg['use_drag_baseline']}  use_wind={cfg['use_wind']}")

    # ======================================================================
    # Roll out the one trajectory, keeping the forces
    # ======================================================================
    trajs, _ = build_force_dataset([trajectory], data_folder,
                                   weights_only=weights_only, unscale_data=unscale,
                                   verbose_every=0)
    trajs[0]["edge_index"] = edge_index
    g_step = torch.tensor([0.0, 0.0, -g]) * dt * dt

    center, angle, forces, (per_traj, pred_all, true_all, lengths) = rollout_force_batched(
        model, trajs, Floor, h, rest_nodes,
        norms["x_mean"], norms["x_std"], norms["e_mean"], norms["e_std"],
        cfg["scale_vec"], cfg["ang_scale_vec"], g_step, dt, device,
        use_wind=cfg["use_wind"], use_drag_baseline=cfg["use_drag_baseline"],
        k_over_m=k_over_m, contact_d0=cfg["contact_d0"],
        contact_tau=cfg["contact_tau"], mass=mass,
        return_forces=True, return_per_traj=True)

    L = lengths[0] - h                              # real (unpadded) frames in this view
    pred = pred_all[0, :L]                          # (L, N, 3)
    true = true_all[0, :L]
    m0 = per_traj[0]
    t_contact, t_settle = int(m0["t_contact"]), int(m0["t_settle"])
    print(f"traj {trajectory}: center {m0['center_error']:.4f} widths | "
          f"angle {m0['angle_error_deg']:.2f} deg | contact@{t_contact} settle@{t_settle}")

    # Force arrays are per PREDICTION STEP. Step i is computed at the state
    # pred[h + i] and produces pred[h + 1 + i]. So frame f has forces iff
    # f >= h, and its force index is (f - h).
    f_norm = forces["node_normal"][0]               # (n_steps, N, 3) Newtons
    f_tang = forces["node_tangent"][0]
    f_fluid = forces["F_fluid"][0]                  # (n_steps, 3) Newtons
    n_steps = f_norm.shape[0]
    assert n_steps >= L - h - 1, "force/frame alignment mismatch"

    # ======================================================================
    # Figure setup
    # ======================================================================
    ARROW_SCALE = mg_arrow_widths * BLOCK_WIDTH / MG      # meters of arrow per Newton
    MIN_F = min_arrow_frac * MG
    MAX_LEN = max_arrow_widths * BLOCK_WIDTH

    def arrow_len(mag_N, gain):
        """Newtons -> arrow length in metres.

        "linear"  physical and honest: length is proportional to force, so a
                  30 N impact draws 30x the 1 N friction and leaves the frame.
        "log"     length = gain * mg_arrow_widths * log1p(F/MG) / log(2), so a
                  force of m*g still draws mg_arrow_widths long and everything
                  above it is compressed. Impact peaks and friction are both
                  legible in one frame. Ordering is preserved, ratios are NOT -
                  say "log-scaled arrow lengths" in the caption.
        "clip"    linear up to max_arrow_widths, then held. Ratios are true
                  below the cap and meaningless above it.
        """
        m = np.asarray(mag_N, dtype=float)
        if arrow_mode == "linear":
            return ARROW_SCALE * gain * m
        if arrow_mode == "clip":
            return np.minimum(ARROW_SCALE * gain * m, MAX_LEN)
        # log
        return (gain * mg_arrow_widths * BLOCK_WIDTH
                * np.log1p(m / MG ) / np.log(2.0))

    def scaled(vecs, gain):
        """Rescale each vector to arrow_len() while keeping its direction."""
        v = np.asarray(vecs, dtype=float)
        mags = np.linalg.norm(v, axis=-1, keepdims=True)
        want = arrow_len(mags[..., 0], gain)[..., None]
        return v / np.maximum(mags, 1e-12) * want

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')

    ei = edge_index.numpy()
    all_pos = torch.cat([pred, true], dim=0) if draw_ground_truth else pred
    pad = 0.5 * BLOCK_WIDTH
    lims = [
        (float(all_pos[:, :, 0].min()) - pad, float(all_pos[:, :, 0].max()) + pad),
        (float(all_pos[:, :, 1].min()) - pad, float(all_pos[:, :, 1].max()) + pad),
        (min(0.0, float(all_pos[:, :, 2].min())) - 0.2 * pad,
         float(all_pos[:, :, 2].max()) + pad),
    ]
    # zoom > 1 pulls the camera in by shrinking the view about its centre.
    # 1.13 is about 13% closer. The floor stays put because the z range is
    # shrunk about its own centre too, so raise `zoom` gently.
    if zoom and zoom != 1.0:
        lims = [(c - (hi - lo) / (2 * zoom), c + (hi - lo) / (2 * zoom))
                for lo, hi in lims for c in [(lo + hi) / 2]]
    ax.set_xlim(*lims[0]); ax.set_ylim(*lims[1]); ax.set_zlim(*lims[2])
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')

    if draw_floor:
        span = float(max(all_pos[:, :, 0].max() - all_pos[:, :, 0].min(),
                         all_pos[:, :, 1].max() - all_pos[:, :, 1].min())) + 2 * pad
        centre = (float(all_pos[:, :, 0].mean()), float(all_pos[:, :, 1].mean()), 0.0)
        wall.wall(center_position=centre, size=(span, span),
                  normal=(0, 0, 1)).show(ax, color="gray", alpha=0.15)

    pred_scatter = ax.scatter([], [], [], c='r', s=25, label='Pred')
    gt_scatter = (ax.scatter([], [], [], c='b', s=25, alpha=0.4, label='GT')
                  if draw_ground_truth else None)
    pred_edge_lines = [ax.plot([], [], [], c='r', alpha=0.35, linewidth=1.0)[0]
                       for _ in range(ei.shape[1])]
    gt_edge_lines = ([ax.plot([], [], [], c='b', alpha=0.2, linewidth=1.0)[0]
                      for _ in range(ei.shape[1])] if draw_ground_truth else [])

    # legend proxies for the arrow colors
    ax.plot([], [], [], c=C_NORMAL, lw=2, label='contact normal')
    ax.plot([], [], [], c=C_TANGENT, lw=2,
            label=f'contact tangential (x{tangent_gain:g})')
    ax.plot([], [], [], c=C_FLUID, lw=2, label='fluid @ COM')
    ax.legend(loc='upper left', fontsize=8)

    hud = fig.text(0.015, 0.015, "", fontsize=9, family='monospace', va='bottom')
    ax.set_title(f"traj {trajectory} - red=pred, blue=GT | "
             f"arrow: {mg_arrow_widths:g} width = m g = {MG:.2f} N"
             f"  (tangential x{tangent_gain:g})")

    quivers = []            # mutable holder so update() can clear last frame's arrows


    def _phase(frame):
        if frame < t_contact:
            return "airborne"
        return "contact" if frame < t_settle else "settled"


    def update(frame):
        p = pred[frame].numpy()
        pred_scatter._offsets3d = (p[:, 0], p[:, 1], p[:, 2])
        for i in range(ei.shape[1]):
            s, d = int(ei[0, i]), int(ei[1, i])
            pred_edge_lines[i].set_data([p[s, 0], p[d, 0]], [p[s, 1], p[d, 1]])
            pred_edge_lines[i].set_3d_properties([p[s, 2], p[d, 2]])

        if draw_ground_truth:
            t = true[frame].numpy()
            gt_scatter._offsets3d = (t[:, 0], t[:, 1], t[:, 2])
            for i in range(ei.shape[1]):
                s, d = int(ei[0, i]), int(ei[1, i])
                gt_edge_lines[i].set_data([t[s, 0], t[d, 0]], [t[s, 1], t[d, 1]])
                gt_edge_lines[i].set_3d_properties([t[s, 2], t[d, 2]])

        # ---- arrows: clear previous frame, redraw ----
        for q in quivers:
            q.remove()
        quivers.clear()

        k = frame - h                       # force index for this frame's state
        if 0 <= k < n_steps:
            fn = f_norm[k].numpy()
            ft = f_tang[k].numpy()
            ff = f_fluid[k].numpy()

            for vecs, color, gain in ((fn, C_NORMAL, normal_gain),
                                    (ft, C_TANGENT, tangent_gain)):
                mags = np.linalg.norm(vecs, axis=1)
                sel = mags > MIN_F
                if sel.any():
                    a = scaled(vecs[sel], gain)
                    quivers.append(ax.quiver(
                        p[sel, 0], p[sel, 1], p[sel, 2],
                        a[:, 0], a[:, 1], a[:, 2],
                        color=color, linewidth=2.4, arrow_length_ratio=0.25))

            com = p.mean(axis=0)
            if np.linalg.norm(ff) > MIN_F:
                a = scaled(ff, fluid_gain)
                quivers.append(ax.quiver(
                    com[0], com[1], com[2],
                    a[0], a[1], a[2],
                    color=C_FLUID, linewidth=2.2, arrow_length_ratio=0.25))

            n_active = int((np.linalg.norm(fn, axis=1) > MIN_F).sum())
            hud.set_text(
                f"frame {frame:3d}/{L-1}  [{_phase(frame)}]\n"
                f"sum |normal| = {np.linalg.norm(fn.sum(axis=0)) / MG:5.2f} m g   "
                f"({n_active} node{'s' if n_active != 1 else ''} in contact)\n"
                f"sum |tangent| = {np.linalg.norm(ft.sum(axis=0)) / MG:5.2f} m g   "
                f"|fluid| = {np.linalg.norm(ff) / MG:5.2f} m g")
        else:
            hud.set_text(f"frame {frame:3d}/{L-1}  [seeded from ground truth]")

        artists = [pred_scatter] + pred_edge_lines + quivers + [hud]
        if gt_scatter is not None:
            artists += [gt_scatter] + gt_edge_lines
        return tuple(artists)


    # Fixed camera and aspect BEFORE anything is rendered, so exported frames
    # and the GIF agree and the cube cannot appear to jump between panels.
    ax.view_init(elev=elev, azim=azim)
    try:
        ax.set_aspect('equal')
    except Exception:
        pass                                     # older matplotlib 3d has no equal aspect

    # ==================================================================
    # Vector frame export for the filmstrip
    # ==================================================================
    frame_paths = []
    if save_frames is not None:
        if isinstance(save_frames, str) and save_frames == "auto":
            idx = auto_frames(f_norm, f_tang, MG, h, L, t_contact, t_settle,
                              n_panels=n_panels, min_active=slide_min_nodes,
                              min_frac=min_arrow_frac)
            print(f"  auto-selected frames: {idx}")
        else:
            idx = sorted({int(f) for f in save_frames if 0 <= int(f) < L})

        fdir = frame_dir or os.path.join(
            os.path.dirname(save_path) or ".", f"frames_traj{trajectory}")
        os.makedirs(fdir, exist_ok=True)

        # Strip the on-screen furniture: a poster panel wants the cube and the
        # arrows, not a HUD, a legend and a title repeated six times.
        if frame_clean:
            keep_title, keep_hud = ax.get_title(), hud.get_text()
            leg = ax.get_legend()
            ax.set_title("")
            hud.set_text("")
            if leg is not None:
                leg.set_visible(False)
            ax.set_axis_off()

        for f in idx:
            update(f)
            if frame_clean:
                hud.set_text("")
            p_out = os.path.join(fdir, f"frame_{f:03d}.{frame_format}")
            fig.savefig(p_out, transparent=True, bbox_inches="tight",
                        pad_inches=0.02, dpi=300)
            frame_paths.append(p_out)
            print(f"  wrote {p_out}   [{_phase(f)}]")

        if frame_clean:                          # restore for the GIF
            ax.set_title(keep_title)
            hud.set_text(keep_hud)
            if leg is not None:
                leg.set_visible(True)
            ax.set_axis_on()

    ani = animation.FuncAnimation(fig, update, frames=L, interval=interval, blit=False)

    if make_gif:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        print(f"Saving animation to {save_path} ...")
        ani.save(save_path, writer='pillow', fps=max(1, 1000 // interval))
        print("Saved successfully.")

    if show:
        plt.show()
    else:
        plt.close(fig)
    return {"gif": save_path if make_gif else None,
            "frames": frame_paths,
            "t_contact": t_contact, "t_settle": t_settle, "n_frames": L}
