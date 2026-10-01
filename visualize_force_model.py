"""Visualize a saved force GNS model's predictions alongside the true motion of
a cube. This module loads one trajectory, runs the shared rollout, and draws
the predicted and ground-truth cubes in 3D with arrows for per-node normal
and tangential contact forces and the net fluid force at the center of mass.
It supports animation and still-frame exports, with automatic frame selection
around contact and settling events for filmstrips. The experiment runner
uses it to generate rollout animations, and make_filmstrip.py exposes its
figure-export options through the command line.

Force display details:
* TWO arrows per node in contact:
      - NORMAL    (green)  along the wall normal, >= 0 by construction
      - TANGENTIAL(orange) in the floor plane - this is the friction force
    Nodes the contact gate has switched off draw nothing, so you can watch
    corners turn on and off as the cube tumbles.
  * ONE arrow at the COM (magenta) for the fluid force (learned residual plus
    the analytic drag baseline if that was enabled). No torque arrow.

Rollout only - the cube follows the model's own predictions and the forces are
whatever it predicts as it drifts.
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
from mpl_toolkits.mplot3d import Axes3D  

import wall
from force_data import BLOCK_WIDTH
from force_data import build_force_dataset
from force_rollout import rollout_force_batched, load_trained_model


#This will automatically select key frames for the filmstrip based on events such as impacts and sliding.
#This is used by make_filmstrip.py to determine which frames to include in the exported filmstrip.
def auto_frames(f_norm, f_tang, MG, h, L, t_contact, t_settle, n_panels=6,
                min_active=4, min_frac=0.02):

    # Convert the input tensors to numpy arrays and compute the summed normal and tangential forces.
    fn_nodes = f_norm.numpy()                       # (n_steps, N, 3)
    Fn = np.linalg.norm(fn_nodes.sum(axis=1), axis=-1) / MG   # summed, in m g
    Ft = np.linalg.norm(f_tang.numpy().sum(axis=1), axis=-1) / MG
    n_steps = len(Fn)

    #Checks to see how many nodes are in contact in each step.
    n_active = (np.linalg.norm(fn_nodes, axis=-1) > min_frac * MG).sum(axis=1)

    first_forced = h                       # earliest frame that has forces
    last_forced = h + n_steps - 1          # latest frame that has forces

    # Helper functions to clamp frame indices within the valid range and to convert 
    #relative indices to absolute frame numbers.
    def clamp(frame):
        return int(np.clip(frame, first_forced, last_forced))

    def to_frame(k):
        return clamp(k + h)

    # (priority, label, frame) - lower priority number is dropped last
    picks = []

    # If the contact time is sufficiently after the initial frame, add a "mid-flight" frame.
    if t_contact > h + 2:
        picks.append((3, "mid-flight", clamp(h + (t_contact - h) // 2)))
    if t_contact - 1 > h:
        picks.append((2, "pre-impact", clamp(t_contact - 1)))

    # Detect the rising edges of the normal force to identify contact episodes.
    on = Fn > 0.15
    edges = np.flatnonzero(np.diff(on.astype(int)) == 1) + 1
    if on.size and on[0]:
        edges = np.r_[0, edges]

    # Iterate over the first two detected contact edges to pick the peak impact frames.
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

    # Determine the sliding window for selecting the "sliding" frame.
    if len(edges) >= 2:
        # The sliding window starts at the second contact edge and ends at the settling time.
        lo_k, hi_k = int(edges[1]), int(np.clip(t_settle - h, 0, n_steps - 1))
    else:
        # If there are fewer than two contact edges, start the sliding window from the first contact frame.
        lo_k, hi_k = (int(np.clip(t_contact - h, 0, n_steps - 1)),
                      int(np.clip(t_settle - h, 0, n_steps - 1)))

    # Ensure that the sliding window has a valid range.
    if hi_k > lo_k + 1:

        # Create an array of frame indices within the sliding window.
        ks = np.arange(lo_k, hi_k)

        # Compute the midpoint of the sliding window for scoring purposes.
        mid = (lo_k + hi_k) / 2.0

        # Find the frame within the sliding window that has the maximum number of active nodes.
        best = n_active[ks].max()
        if best < min_active:
            print(f"    (sliding: only {best} nodes ever loaded in the slide "
                  f"window, wanted {min_active})")

        # If the maximum number of active nodes is below the minimum threshold, print a warning.
        ok = ks[n_active[ks] >= min(min_active, best)]

        # Score the qualifying frames based on friction and proximity to the midpoint of the sliding window.
        score = Ft[ok] - 0.002 * np.abs(ok - mid)
        k_slide = int(ok[int(np.argmax(score))])
        picks.append((1, "sliding", to_frame(k_slide)))
        print(f"    (sliding frame has {n_active[k_slide]} loaded nodes)")

    # Pick the frame with the maximum friction for reference.
    if Ft.size:
        picks.append((2, "max friction", to_frame(int(np.argmax(Ft)))))

    # Pick the settled frame for reference.
    picks.append((1, "settled", clamp(t_settle + 5)))

    # De-duplicate frames by keeping the highest-priority label for each frame.
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

    # Sort the unique frames by frame number for final output.
    unique.sort(key=lambda t: t[2])
    for prio, label, f in unique:
        print(f"    frame {f:3d}  {label}")

    # Return the list of unique frame numbers for further processing or visualization.
    return [f for _, _, f in unique]


# Visualize the force rollout for a single trajectory.
# This will show a full animation of the force rollout for the specified trajectory.
# The truth cube will be blue. and the predicted cube will be red.
# The forces on the predicted cube will be shown as arrows (normal + tangential).
# Aswell as fluid forces at the COM of the predicted cube.
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

    # Ensure the save path directory exists.
    if save_path is None:
        save_path = os.path.join(model_folder, f"force_rollout_{trajectory}.gif")

    #sets the colors for normal, tangential, and fluid forces.
    C_NORMAL, C_TANGENT, C_FLUID = "tab:green", "tab:orange", "magenta"

    # ======================================================================
    # Load model + config
    # ======================================================================

    # Set the device for PyTorch computations (GPU if available, otherwise CPU).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create a floor object for the simulation.
    Floor = wall.wall(center_position=(0, 0, 0), size=(2, 2), normal=(0, 0, 1))

    # Load the trained model along with its configuration and normalization statistics.
    model, norms, cfg, rest_nodes, edge_index, model_prefix, k_over_m = load_trained_model(
        model_folder, device, prefix=model_prefix, checkpoint=checkpoint)

    # Extract simulation parameters from the configuration.
    h, dt, mass, g = cfg["h"], cfg["dt"], cfg["mass"], cfg["gravity"]

    # Compute the weight of the object under gravity.
    MG = mass * g

    print(f"Model: {model_prefix}   dt={dt:.6f}s  m={mass}kg  g={g}  "
          f"drag_baseline={cfg['use_drag_baseline']}  use_wind={cfg['use_wind']}")

    # ======================================================================
    # Roll out the one trajectory, keeping the forces
    # ======================================================================

    # Build the dataset (the node and edge information for each frame) for the specified trajectory
    trajs, _ = build_force_dataset([trajectory], data_folder,
                                   weights_only=weights_only, unscale_data=unscale,
                                   verbose_every=0)

    # Assign the edge index to the first trajectory in the dataset.
    trajs[0]["edge_index"] = edge_index

    # Compute the gravity step for the simulation.
    g_step = torch.tensor([0.0, 0.0, -g]) * dt * dt

    # Roll out the simulation for the first trajectory using the trained model.
    center, angle, forces, (per_traj, pred_all, true_all, lengths) = rollout_force_batched(
        model, trajs, Floor, h, rest_nodes,
        norms["x_mean"], norms["x_std"], norms["e_mean"], norms["e_std"],
        cfg["scale_vec"], cfg["ang_scale_vec"], g_step, dt, device,
        use_wind=cfg["use_wind"], use_drag_baseline=cfg["use_drag_baseline"],
        k_over_m=k_over_m, contact_d0=cfg["contact_d0"],
        contact_tau=cfg["contact_tau"], mass=mass,
        return_forces=True, return_per_traj=True)

    # Extract the relevant portions of the rollout results for the first trajectory.
    L = lengths[0] - h                              # real (unpadded) frames in this view
    pred = pred_all[0, :L]                          # (L, N, 3)
    true = true_all[0, :L]
    m0 = per_traj[0]

    # Extract the contact and settle times from the per-trajectory metadata.
    t_contact, t_settle = int(m0["t_contact"]), int(m0["t_settle"])

    print(f"traj {trajectory}: center {m0['center_error']:.4f} widths | "
          f"angle {m0['angle_error_deg']:.2f} deg | contact@{t_contact} settle@{t_settle}")

    # Force arrays are per PREDICTION STEP. Step i is computed at the state
    # pred[h + i] and produces pred[h + 1 + i]. So frame f has forces iff
    # f >= h, and its force index is (f - h).

    # Extract the force arrays for the first trajectory. These are per prediction step.
    f_norm = forces["node_normal"][0]               # (n_steps, N, 3) Newtons
    f_tang = forces["node_tangent"][0]
    f_fluid = forces["F_fluid"][0]                  # (n_steps, 3) Newtons
    n_steps = f_norm.shape[0]

    # Ensure that the number of force steps is sufficient for the number of frames being visualized.
    assert n_steps >= L - h - 1, "force/frame alignment mismatch"

    # ======================================================================
    # Figure setup
    # ======================================================================

    # Define the scaling parameters for the force arrows in the visualization.
    ARROW_SCALE = mg_arrow_widths * BLOCK_WIDTH / MG      # meters of arrow per Newton
    MIN_F = min_arrow_frac * MG
    MAX_LEN = max_arrow_widths * BLOCK_WIDTH


    # Define a helper function to convert force magnitudes to arrow lengths in the visualization.
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
        # Convert the input magnitude to a NumPy array for consistent processing.
        m = np.asarray(mag_N, dtype=float)

        #Applies the linear, clip, or log scaling to convert force magnitude to arrow length.
        if arrow_mode == "linear":
            return ARROW_SCALE * gain * m
        if arrow_mode == "clip":
            return np.minimum(ARROW_SCALE * gain * m, MAX_LEN)
        # log
        return (gain * mg_arrow_widths * BLOCK_WIDTH
                * np.log1p(m / MG ) / np.log(2.0))

    # Define a helper function to scale vectors to the appropriate arrow 
    # lengths while preserving their direction.
    def scaled(vecs, gain):
        """Rescale each vector to arrow_len() while keeping its direction."""
        v = np.asarray(vecs, dtype=float)
        mags = np.linalg.norm(v, axis=-1, keepdims=True)
        want = arrow_len(mags[..., 0], gain)[..., None]
        return v / np.maximum(mags, 1e-12) * want

    # Set up the 3D figure and axes for the visualization.
    fig = plt.figure(figsize=(9, 7))

    # Add a 3D subplot to the figure for rendering the scene.
    ax = fig.add_subplot(111, projection='3d')

    # Extract the edge index and concatenate predicted and ground truth positions for visualization.
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

    # Apply zoom to the axis limits if specified.
    if zoom and zoom != 1.0:
        lims = [(c - (hi - lo) / (2 * zoom), c + (hi - lo) / (2 * zoom))
                for lo, hi in lims for c in [(lo + hi) / 2]]
    ax.set_xlim(*lims[0]); ax.set_ylim(*lims[1]); ax.set_zlim(*lims[2])
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')

    # Draw the floor
    if draw_floor:
        span = float(max(all_pos[:, :, 0].max() - all_pos[:, :, 0].min(),
                         all_pos[:, :, 1].max() - all_pos[:, :, 1].min())) + 2 * pad
        centre = (float(all_pos[:, :, 0].mean()), float(all_pos[:, :, 1].mean()), 0.0)
        # Draw the floor as a semi-transparent gray plane.
        wall.wall(center_position=centre, size=(span, span),
                  normal=(0, 0, 1)).show(ax, color="gray", alpha=0.15)

    # Initialize scatter plots for predicted and ground truth positions, and lines for edges.
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

    # Define a helper function to determine the current phase of the trajectory based on the frame index.
    def _phase(frame):
        if frame < t_contact:
            return "airborne"
        return "contact" if frame < t_settle else "settled"

    # Define the update function for the animation, which will be called for each frame.
    def update(frame):
        # Extract the predicted positions for the current frame.
        p = pred[frame].numpy()
        pred_scatter._offsets3d = (p[:, 0], p[:, 1], p[:, 2])

        # Update the ground truth positions and edges for the current frame if available.
        for i in range(ei.shape[1]):
            s, d = int(ei[0, i]), int(ei[1, i])
            pred_edge_lines[i].set_data([p[s, 0], p[d, 0]], [p[s, 1], p[d, 1]])
            pred_edge_lines[i].set_3d_properties([p[s, 2], p[d, 2]])

        #draws the ground truth positions and edges if available.
        if draw_ground_truth:
            t = true[frame].numpy()
            gt_scatter._offsets3d = (t[:, 0], t[:, 1], t[:, 2])
            for i in range(ei.shape[1]):
                s, d = int(ei[0, i]), int(ei[1, i])
                gt_edge_lines[i].set_data([t[s, 0], t[d, 0]], [t[s, 1], t[d, 1]])
                gt_edge_lines[i].set_3d_properties([t[s, 2], t[d, 2]])

        # Clear the previous frame's force arrows.
        for q in quivers:
            q.remove()
        quivers.clear()

        k = frame - h                       # force index for this frame's state
        if 0 <= k < n_steps:
            fn = f_norm[k].numpy()
            ft = f_tang[k].numpy()
            ff = f_fluid[k].numpy()

            # Draw the normal and tangential force arrows for each node if above the minimum threshold.
            for vecs, color, gain in ((fn, C_NORMAL, normal_gain),
                                    (ft, C_TANGENT, tangent_gain)):
                mags = np.linalg.norm(vecs, axis=1)
                sel = mags > MIN_F
                if sel.any():

                    # Scale the selected force vectors by the gain factor before drawing.
                    a = scaled(vecs[sel], gain)

                    # Append the quiver object representing the force arrows to the list of quivers.
                    quivers.append(ax.quiver(
                        p[sel, 0], p[sel, 1], p[sel, 2],
                        a[:, 0], a[:, 1], a[:, 2],
                        color=color, linewidth=2.4, arrow_length_ratio=0.25))

            # Draw the fluid force arrow at the center of mass if above the minimum threshold.
            com = p.mean(axis=0)
            if np.linalg.norm(ff) > MIN_F:
                a = scaled(ff, fluid_gain)
                quivers.append(ax.quiver(
                    com[0], com[1], com[2],
                    a[0], a[1], a[2],
                    color=C_FLUID, linewidth=2.2, arrow_length_ratio=0.25))

            # Update the HUD with the current frame's force information.
            n_active = int((np.linalg.norm(fn, axis=1) > MIN_F).sum())
            hud.set_text(
                f"frame {frame:3d}/{L-1}  [{_phase(frame)}]\n"
                f"sum |normal| = {np.linalg.norm(fn.sum(axis=0)) / MG:5.2f} m g   "
                f"({n_active} node{'s' if n_active != 1 else ''} in contact)\n"
                f"sum |tangent| = {np.linalg.norm(ft.sum(axis=0)) / MG:5.2f} m g   "
                f"|fluid| = {np.linalg.norm(ff) / MG:5.2f} m g")
        else:
            hud.set_text(f"frame {frame:3d}/{L-1}  [seeded from ground truth]")

        # Collect all the artists (predicted scatter, edges, force arrows, HUD) for the current frame.
        artists = [pred_scatter] + pred_edge_lines + quivers + [hud]
        if gt_scatter is not None:
            artists += [gt_scatter] + gt_edge_lines

        # Return the tuple of all artists for the current frame.
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

    # Prepare to save individual frames if requested.
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

        # Strip the on-screen hud
        if frame_clean:
            keep_title, keep_hud = ax.get_title(), hud.get_text()
            leg = ax.get_legend()
            ax.set_title("")
            hud.set_text("")
            if leg is not None:
                leg.set_visible(False)
            ax.set_axis_off()

        # Loop over the selected frames and save each one as an individual image file.
        for f in idx:
            update(f)
            if frame_clean:
                hud.set_text("")
            p_out = os.path.join(fdir, f"frame_{f:03d}.{frame_format}")
            fig.savefig(p_out, transparent=True, bbox_inches="tight",
                        pad_inches=0.02, dpi=300)
            frame_paths.append(p_out)
            print(f"  wrote {p_out}   [{_phase(f)}]")

        # Restore the on-screen HUD and other elements after saving frames.
        if frame_clean:                          
            ax.set_title(keep_title)
            hud.set_text(keep_hud)
            if leg is not None:
                leg.set_visible(True)
            ax.set_axis_on()

    # Create the animation object using the update function and the total number of frames.
    ani = animation.FuncAnimation(fig, update, frames=L, interval=interval, blit=False)

    # Save the animation as a GIF if requested.
    if make_gif:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        print(f"Saving animation to {save_path} ...")
        ani.save(save_path, writer='pillow', fps=max(1, 1000 // interval))
        print("Saved successfully.")

    # Display the plot if requested, otherwise close it.
    if show:
        plt.show()
    else:
        plt.close(fig)

    # Return a dictionary containing paths to the saved GIF and frames, 
    # as well as timing and frame count information.
    return {"gif": save_path if make_gif else None,
            "frames": frame_paths,
            "t_contact": t_contact, "t_settle": t_settle, "n_frames": L}
