"""Implement the training procedure for the force GNS using observed cube
trajectories. This module computes normalization statistics, builds batches
with state-history and future-target windows, applies noise and rotation
augmentation, and differentiates through multiple predicted rigid-body steps
to compute trajectory and optional physics losses. It manages optimization,
the multistep curriculum, rollout-based validation, early stopping, checkpoint
selection, and saved loss and parameter histories. It also converts requested
optimizer-step budgets into whole epochs. Experiment settings come from
run_force_multi_step.py; data preparation, model dynamics, and shared rollout
operations are supplied by force_data.py, force_gns.py, and force_rollout.py.
"""

import os
import re
import time
import math
import torch
import torch.optim as optim

from force_data import (mesh_cube_surface, knn_adjacency,
                        build_force_dataset, add_random_walk_noise, relative_wind,
                        BLOCK_HALF_WIDTH, BLOCK_WIDTH)
from force_rollout import _build_features_for_unroll, rollout_force_batched

# ---- the force model + dynamics layer ----
from force_gns import (ForceGNSModel, so3_exp, so3_log,
                       rigid_step, nodes_from_state, contact_weight,
                       assemble_contact_forces, fluid_wrench_from_raw,
                       drag_accel_step)

from physics_losses import PhysicsLosses


#This function takes one trajectory's node positions, applies random-walk noise to them,
#and returns the node features, edge features, and target accelerations for every timestep.
#It is only used to compute the normalization stats.
def _noisy_features_and_targets(positions, wind_vector, edge_index, Wall, h,
                                noise_scale, use_wind):
    """
    positions: (T, N, 3) clean node positions of one trajectory.
    Returns (x (M*N, node_dim), e (M*E, 4), y (M*N, 3)) stacked over the M
    timestep samples
    """

    # Add random-walk noise to the clean positions to simulate realistic perturbations.
    noisy_positions, noise = add_random_walk_noise(positions, noise_scale=noise_scale)

    # Identify the sender and receiver nodes for each edge.
    sender = edge_index[0]
    receiver = edge_index[1]

    # Extract wall normal and center position as tensors.
    wall_n = torch.as_tensor(Wall.normal, dtype=torch.float32)
    wall_c = torch.as_tensor(Wall.center_position, dtype=torch.float32)

    T = positions.shape[0]
    M = T - 1 - h  # number of timestep samples (matches range(h, T-1))
    if M <= 0:
        return None

    # ---- Velocity history features (vectorized over t) ----
    # For t in [h, T-2], v_k = noisy[t-k] - noisy[t-k-1] for k in [0, h-1]
    #Computes the finite-difference velocity history for each node over the past h timesteps.
    v_fd_list = []
    for k in range(h):
        v_k = noisy_positions[h-k : T-1-k] - noisy_positions[h-k-1 : T-2-k]  # (M, N, 3)
        v_fd_list.append(v_k)
    v_fd_all = torch.cat(v_fd_list, dim=-1)  # (M, N, 3h)

    # ---- Wall distance (vectorized over t) ----
    rel_pos = noisy_positions[h : T-1] - wall_c                        # (M, N, 3)
    dist_all = torch.sum(rel_pos * wall_n, dim=-1, keepdim=True).clamp(-0.05, 0.5)  # (M, N, 1)

    #Adds the current velocity, relative wind (if used), and wall distance to the node features.
    v_curr = v_fd_list[0]
    node_parts = [v_fd_all]
    if use_wind:
        u, u_norm = relative_wind(wind_vector.view(1, 1, 3), v_curr)
        node_parts += [u, u_norm]
    node_parts.append(dist_all)
    x_node_all = torch.cat(node_parts, dim=-1)

    # Each edge contains the current displacement vector and its magnitude.
    pos_at_t = noisy_positions[h : T-1]                                # (M, N, 3)
    d_all = pos_at_t[:, sender] - pos_at_t[:, receiver]                # (M, E, 3)
    d_norm_all = torch.norm(d_all, dim=-1, keepdim=True)               # (M, E, 1)
    e_attr_all = torch.cat([d_all, d_norm_all], dim=-1)  # (M, E, 4)

    #Computes the acceleration targets for each node based on the finite-difference of the clean positions, corrected for noise.
    # These targets are used for training the model to predict node accelerations.
    accel_clean = positions[h+1 : T] - 2.0 * positions[h : T-1] + positions[h-1 : T-2]
    accel_corrected = accel_clean - noise[h-1 : T-2]                   # (M, N, 3)

    #Returns the node features, edge features, and acceleration targets for the current timestep.
    return (x_node_all.reshape(-1, x_node_all.shape[-1]),
            e_attr_all.reshape(-1, e_attr_all.shape[-1]),
            accel_corrected.reshape(-1, 3))

#Per-feature mean and std of the node features, edge features, and
#acceleration targets over every (noisy) training timestep. Returns
# (x_mean, x_std, e_mean, e_std, acc_mean, acc_std).
def _normalization_stats(dataset, rest_nodes, edge_index, Wall, h, noise_scale, use_wind):
    
    xs, es, ys = [], [], []

    #Iterates over the dataset to collect all node features, edge features, and acceleration targets.
    for d in dataset:
        positions = d["com"].unsqueeze(1) + torch.einsum('tij,nj->tni', d["R"], rest_nodes)
        sample = _noisy_features_and_targets(positions, d["wind"], edge_index,
                                             Wall, h, noise_scale, use_wind)
        if sample is None:
            continue
        xs.append(sample[0])
        es.append(sample[1])
        ys.append(sample[2])

    #Takes the collected node, edge, and acceleration features and computes their mean and standard deviation.
    def mean_std(parts):
        a = torch.cat(parts, dim=0)
        return a.mean(dim=0), a.std(dim=0).clamp_min(1e-8)

    #Computes the mean and standard deviation for each feature across all collected samples.
    x_mean, x_std = mean_std(xs)
    e_mean, e_std = mean_std(es)
    acc_mean, acc_std = mean_std(ys)
    return x_mean, x_std, e_mean, e_std, acc_mean, acc_std


#This function computes empirical per-step^2 angular-acceleration statistics for the training dataset.
# Returns a tensor of shape (3,) containing the standard deviation of angular acceleration in x, y, and
# z directions.
def _compute_angular_stats(dataset):

    alphas = []

    #Iterates over the dataset to compute angular acceleration for each sequence.
    for d in dataset:
        R = d["R"]
        w = so3_log(R[1:] @ R[:-1].transpose(-1, -2))           # (T-1, 3) rad/step
        if w.shape[0] >= 2:
            alphas.append(w[1:] - w[:-1])                       # (T-2, 3) rad/step^2

    #Concatenates all angular acceleration samples into a single tensor for computing statistics.
    a = torch.cat(alphas, dim=0)
    std = a.std(dim=0)

    #Computes the mean standard deviation for the x and y components, and keeps the z component separate.
    s_xy = float(std[:2].mean())

    #Returns the symmetrized angular acceleration standard deviation tensor.
    return torch.tensor([s_xy, s_xy, float(std[2])]).clamp_min(1e-8)


# ======================================================================
# Small helpers
# ======================================================================

# Check if the Triton library is available for optimized GPU operations.
def _triton_available():
    try:
        import triton  # noqa: F401
        return True
    except Exception:
        return False

#Function to prune old model checkpoints, keeping only the most recent ones.
#This makes it easier to manage disk space by removing outdated checkpoints automatically.
def prune_old_checkpoints(save_model_path, keep_last_n):

    if keep_last_n is None or keep_last_n <= 0:
        return
    stem = os.path.splitext(save_model_path)[0]
    folder = os.path.dirname(stem) or "."
    base = os.path.basename(stem)
    pattern = re.compile(r"^" + re.escape(base) + r"_epoch(\d+)\.pt$")

    found = []
    for fn in os.listdir(folder):
        m = pattern.match(fn)
        if m:
            found.append((int(m.group(1)), os.path.join(folder, fn)))
    found.sort()                                   # oldest epoch first
    for _, path in found[:-keep_last_n]:
        try:
            os.remove(path)
        except OSError as e:
            print(f"  (could not remove {os.path.basename(path)}: {e})")

# Generate a random rotation matrix about the z-axis.
# Used for data augmentation in the force model training.
def _random_z_rotation():
    th = torch.rand(()) * 2.0 * math.pi
    c, s = torch.cos(th), torch.sin(th)
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# Functions for building and iterating over chains of rigid states
# Used to make training batches for the model during training.
# Allows the model to have the nessesary state history for calculating the velocity history,
# and also gives the model access to future target states for trianing when using multistep prediction.
def build_chain_index(dataset, h, multistep):
    """List of (traj_idx, start_frame) for every valid chain window."""
    span = h + 1 + multistep
    index = []
    for ti, d in enumerate(dataset):
        last_start = d["T"] - span
        for s in range(last_start + 1):
            index.append((ti, s))
    return index

# Iterate over chains of rigid states to create training batches.
# Yields batches of chains with optional rigid random-walk noise applied to the input window.
def iterate_force_chains(dataset, chain_index, batch_size, h, multistep,
                         device, noise_scale):
    """
    Yields shuffled chain batches of rigid state. Rigid random-walk noise is
    applied to the INPUT window only (frame 0 clean, targets clean), mirroring
    the acceleration model's chain-noise convention but in (COM, rotation)
    space:
      COM:      i.i.d. velocity noise per transition (std noise_scale), cumsum'd
      rotation: i.i.d. rotation-vector noise per transition, cumsum'd and
                applied as a left perturbation  R_noisy = exp(w_cum) R.
                Its std is noise_scale / BLOCK_HALF_WIDTH: the rotation that
                moves a corner about as far as the COM noise does.
    """

    # Shuffle the chain index to create random batches.
    order = torch.randperm(len(chain_index))

    # Compute the rotation noise scale based on the block half-width.
    rot_noise_scale = noise_scale / BLOCK_HALF_WIDTH

    # Iterate over the chain index in batches of the specified size.
    for start in range(0, len(chain_index), batch_size):

        # Select the current batch of chain indices.
        sel = order[start:start + batch_size].tolist()
        B = len(sel)

        # Allocate tensors for the input and target windows, as well as the wind vectors.
        com_win = torch.empty(B, h + 1, 3)
        R_win = torch.empty(B, h + 1, 3, 3)
        tgt_com = torch.empty(B, multistep, 3)
        tgt_R = torch.empty(B, multistep, 3, 3)
        winds = torch.empty(B, 3)


        # Populate the tensors with data from the selected chains.
        for b, idx in enumerate(sel):
            ti, s = chain_index[idx]
            d = dataset[ti]
            com_win[b] = d["com"][s: s + h + 1]
            R_win[b] = d["R"][s: s + h + 1]
            tgt_com[b] = d["com"][s + h + 1: s + h + 1 + multistep]
            tgt_R[b] = d["R"][s + h + 1: s + h + 1 + multistep]
            winds[b] = d["wind"]

        # Move the tensors to the specified device.
        com_win = com_win.to(device)
        R_win = R_win.to(device)
        tgt_com = tgt_com.to(device)
        tgt_R = tgt_R.to(device)
        winds = winds.to(device)

        # Apply random-walk noise to the input window if the noise scale is greater than zero.
        if noise_scale > 0:
            vel_noise = torch.randn(B, h, 3, device=device) * noise_scale
            com_win[:, 1:] = com_win[:, 1:] + torch.cumsum(vel_noise, dim=1)
            w_noise = torch.randn(B, h, 3, device=device) * rot_noise_scale
            w_cum = torch.cumsum(w_noise, dim=1).reshape(B * h, 3)
            R_win[:, 1:] = so3_exp(w_cum).reshape(B, h, 3, 3) @ R_win[:, 1:]


        # Yield the current batch as a dictionary of tensors.
        # The batch dictionary contains the input and target windows, wind vectors, and batch size.
        yield {"com_win": com_win, "R_win": R_win, "tgt_com": tgt_com,
               "tgt_R": tgt_R, "wind": winds, "B": B}


# This function applies a random z-rotation augmentation to the rigid state in the batch.
def rotate_force_chain(batch):

    # Generate a random z-rotation matrix and apply it to the COM, rotation matrices, 
    # and wind vectors in the batch.
    Rz = _random_z_rotation().to(batch["com_win"].device)
    batch["com_win"] = batch["com_win"] @ Rz.T
    batch["tgt_com"] = batch["tgt_com"] @ Rz.T
    batch["wind"] = batch["wind"] @ Rz.T
    batch["R_win"] = Rz @ batch["R_win"]
    batch["tgt_R"] = Rz @ batch["tgt_R"]
    return batch


# Multistep unroll loss through the force decoder + dynamics layer.
# Basically unroll the model for multiple steps, compute the predicted next states, 
# and keep track of the loss over multiple steps.
def _unroll_force_loss(model, batch, multistep, Wall, h, rest_nodes,
                       edge_index_b, N,
                       x_mean, x_std, e_mean, e_std, scale_vec, ang_scale_vec,
                       acc_mean, acc_std, g_step, dt,
                       use_wind, use_drag_baseline, k_over_m, k_learnable,
                       contact_d0, contact_tau, loss_mode,
                       phys, phys_weights, report_slip):

    # Extract the relevant tensors from the batch and set up initial variables.
    com_win, R_win = batch["com_win"], batch["R_win"]
    tgt_com, tgt_R = batch["tgt_com"], batch["tgt_R"]
    wind = batch["wind"]
    B = batch["B"]
    device = com_win.device

    # Convert the wall's normal and center position to tensors and normalize the normal vector.
    wall_n = torch.as_tensor(Wall.normal, dtype=torch.float32, device=device)
    wall_n = wall_n / wall_n.norm().clamp_min(1e-12)
    wall_c = torch.as_tensor(Wall.center_position, dtype=torch.float32, device=device)

    # Build the initial position window for the unroll, and extract the previous and current
    # COM and rotation matrices.
    pos_window = [nodes_from_state(com_win[:, j], R_win[:, j], rest_nodes)
                  for j in range(h + 1)]

    # Extract the previous and current COM and rotation matrices for the rigid step.
    com_prev, com_curr = com_win[:, -2], com_win[:, -1]
    R_prev, R_curr = R_win[:, -2], R_win[:, -1]

    # Check if any physics loss terms are active and initialize accumulators for raw losses and series data.
    any_phys = any(v > 0 for v in phys_weights.values())

    raw_accum = {}
    fluid_series = []      # total fluid accel per step, for temporal smoothness
    torque_series = []     # fluid angular accel per step, same purpose

    step_losses = []

    # Unroll the model for the specified number of steps, computing predictions and losses at each step.
    for k in range(multistep):

        # Build the input features for the current unroll step
        x_node, e_attr = _build_features_for_unroll(
            pos_window, edge_index_b, Wall, wind,
            x_mean, x_std, e_mean, e_std, B, N, use_wind)

        # Runs the model to obtain raw contact and fluid predictions.
        contact_raw, fluid_raw = model(x_node, edge_index_b, e_attr, B)

        # Extract the current node positions from the position window.
        cur_nodes = pos_window[-1]

        # Compute the distance to the wall and the corresponding contact weight.
        dist = ((cur_nodes - wall_c) * wall_n).sum(-1, keepdim=True) 

        # Compute the contact weight based on the distance to the wall.
        c_w = contact_weight(dist, d0=contact_d0, tau=contact_tau)

        # Assemble the contact forces and compute the fluid accelerations from the raw predictions.
        phi_c = assemble_contact_forces(contact_raw, c_w, wall_n, scale_vec)
        a_fluid, alpha_fluid = fluid_wrench_from_raw(fluid_raw, scale_vec,
                                                     ang_scale_vec)

        # Optionally add the drag baseline to the fluid acceleration.
        # This will basically make the model learn the residual fluid acceleration
        # on top of the drag baseline.
        extra_accel = a_fluid
        if use_drag_baseline:
            extra_accel = extra_accel + drag_accel_step(
                wind, com_curr - com_prev, dt, k_over_m)

        # Perform the rigid body step to obtain the next center of mass and rotation,
        # using the computed contact and fluid forces.
        com_next, R_next = rigid_step(phi_c, com_prev, com_curr, R_prev, R_curr,
                                      rest_nodes, g_step,
                                      extra_accel=extra_accel,
                                      extra_alpha=alpha_fluid)

        # Compute the predicted and true node positions based on the updated and target states.
        pred_nodes = nodes_from_state(com_next, R_next, rest_nodes)
        true_nodes = nodes_from_state(tgt_com[:, k], tgt_R[:, k], rest_nodes)

        # Compute the loss for this step based on the chosen loss mode.
        if loss_mode == "accel":
            # Accel loss for the current step

            # Compute the predicted and true accelerations for the current step.
            a_pred = pred_nodes - 2.0 * pos_window[-1] + pos_window[-2]
            a_true = true_nodes - 2.0 * pos_window[-1] + pos_window[-2]

            # Normalize the predicted and true accelerations using the mean and standard deviation.
            a_pred_norm = (a_pred - acc_mean) / acc_std
            a_true_norm = (a_true - acc_mean) / acc_std

            # Compute the step loss as the mean squared error between the normalized 
            # predicted and true accelerations.
            step_losses.append((a_pred_norm - a_true_norm).pow(2).mean())
        else:
            # Position loss for the current step (used when loss_mode is not "accel").
            step_losses.append(((pred_nodes - true_nodes)
                                / BLOCK_WIDTH).pow(2).mean())

        # Compute the node velocity for the current step.
        v_node = pos_window[-1] - pos_window[-2]              # m/step

        # Report slip information for the first step if requested.
        if report_slip and k == 0:
            phys.last_slip_report = phys.slip_gate_report(
                phi_c, c_w, v_node, wall_n, dt=dt)


        # Check if any physics-based losses need to be computed for the current step.
        if any_phys:

            # Compute the theoretical drag acceleration for the current step using quadratic drag.
            _drag = drag_accel_step(
                wind, (com_curr - com_prev).detach(), dt, k_over_m)

            # Determine the target drag acceleration
            drag_target = _drag if k_learnable else _drag.detach()

            # Compute the total fluid acceleration for the physics terms, including the drag baseline if used.
            # Basically if drag baseline is used, include the drag target in the total fluid acceleration.
            # This will result in the a_fluid being a learned residual on top of the drag baseline.
            fluid_total_phys = (a_fluid + drag_target if use_drag_baseline
                                else a_fluid)
            fluid_series.append(fluid_total_phys)
            torque_series.append(alpha_fluid)

            # Compute all of the physics loss terms for the current step.
            step_raws = phys.compute_step_terms(
                phi_c, c_w, v_node, wall_n,
                fluid_total_phys, drag_target, phys_weights)

            # Accumulate the raw physics loss terms for later aggregation.
            for kname, v in step_raws.items():
                raw_accum[kname] = raw_accum.get(kname, 0.0) + v


        # Update the position and orientation windows for the next step.
        pos_window = pos_window[1:] + [pred_nodes]
        com_prev, com_curr = com_curr, com_next
        R_prev, R_curr = R_curr, R_next

    # Compute the mean position loss over all steps.
    pos_loss = torch.stack(step_losses).mean()

    # Normalize the accumulated raw physics loss terms by the number of steps.
    raw_terms = {k: v / multistep for k, v in raw_accum.items()}

    # Optionally add the fluid temporal smoothness term if its weight is positive.
    if phys_weights["w_fluid_smooth"] > 0:
        raw_terms["fluid_smooth"] = phys.h_fluid_temporal_smooth(
            fluid_series, torque_series)

    # Compute the total loss as the sum of the position loss and the weighted physics loss terms.
    total = pos_loss + PhysicsLosses.weighted_total(raw_terms, phys_weights)

    # Return the total loss and the detached raw physics terms for logging.
    return total, {k: float(v.detach()) for k, v in raw_terms.items()}


def _resolve_curriculum(multistep, curriculum_epochs, curriculum_schedule):
    """Return the same validated rollout-length schedule for planning and training."""
    if multistep < 1 or curriculum_epochs < 0:
        raise ValueError("multistep must be positive and curriculum_epochs nonnegative")
    if curriculum_epochs == 0 or multistep == 1:
        return [multistep]
    if curriculum_schedule is None:
        schedule = []
        k = 1
        while k < multistep:
            schedule.append(k)
            k *= 2
        return schedule + [multistep]
    schedule = list(curriculum_schedule)
    if (not schedule or schedule[-1] != multistep
            or any(not isinstance(k, int) or not 1 <= k <= multistep for k in schedule)
            or any(a > b for a, b in zip(schedule, schedule[1:]))):
        raise ValueError("curriculum_schedule must be nondecreasing positive integers ending at multistep")
    return schedule


def epochs_for_optimizer_steps(trajectory_lengths, target_optimizer_steps, batch_size,
                              h, multistep, accumulation_steps=1,
                              curriculum_epochs=0, curriculum_schedule=None):
    """Return (epochs, planned_steps), rounding the target up to a complete epoch.

    Each length T supplies max(T - h - K, 0) training windows. Both the final
    partial batch and final partial gradient-accumulation group count. Earlier
    curriculum phases last curriculum_epochs; the final phase continues until
    the target is reached. Early stopping can shorten the actual run.
    """
    for name, value in (("target_optimizer_steps", target_optimizer_steps),
                        ("batch_size", batch_size), ("h", h),
                        ("multistep", multistep), ("accumulation_steps", accumulation_steps)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not isinstance(curriculum_epochs, int) or curriculum_epochs < 0:
        raise ValueError("curriculum_epochs must be a nonnegative integer")
    lengths = list(trajectory_lengths)
    schedule = _resolve_curriculum(multistep, curriculum_epochs, curriculum_schedule)
    epochs = steps = 0
    for phase, k in enumerate(schedule):
        windows = sum(max(int(length) - h - k, 0) for length in lengths)
        batches = (windows + batch_size - 1) // batch_size
        steps_per_epoch = (batches + accumulation_steps - 1) // accumulation_steps
        if steps_per_epoch == 0:
            raise ValueError(f"No training windows available for history={h}, rollout K={k}")
        needed = (target_optimizer_steps - steps + steps_per_epoch - 1) // steps_per_epoch
        phase_epochs = needed if phase == len(schedule) - 1 else min(needed, curriculum_epochs)
        epochs += phase_epochs
        steps += phase_epochs * steps_per_epoch
        if steps >= target_optimizer_steps:
            return epochs, steps
    raise AssertionError("Final curriculum phase should exhaust the step target")


#This function is the main training loop for the force GNN model.
def train_force_gnn(Wall,
                    train_range,
                    val_range,
                    save_model_path,
                    trajectory_folder,
                    epochs,
                    batch_size,
                    accumulation_steps,
                    lr,
                    nodes_per_edge,
                    nearest_neighbors,
                    h,
                    message_passing_layers,
                    repeat_blocks,
                    latent_dim,
                    weights_only,
                    unscale_data,
                    noise_scale,
                    multistep,
                    curriculum_epochs,             # 0 = off; else epochs per ramp phase
                    curriculum_schedule,           # e.g. [1,2,4,8]; None -> powers of 2
                    Learning_Rate_Scheduler,       # "decay", "cosine", or None
                    use_wind,
                    dt,
                    gravity,                       # None -> read from replica_physics, else 9.615
                    mass,
                    use_drag_baseline,
                    k_over_m,              # k/m init (the fixed value when learn_k=False)
                    learn_k,               # recover k/m from data, like mu
                    contact_d0,
                    contact_tau,
                    loss_mode,             # "accel" (parity) or "position"
                    # --- physics-informed loss weights (proposal Eq. 6 gammas;
                    #     see physics_losses.py for each term) ---
                    w_fric_dir,            # gamma_1a: Coulomb DIRECTION half
                    w_fric_mag,            # gamma_1b: Coulomb MAGNITUDE half
                                           #   (mu's only gradient path)
                    w_fric_cone,           # gamma_1c: cone bound, static too
                    w_fluid_anchor,        # gamma_3a: fluid force == analytic drag law
                    w_fluid_smooth,        # gamma_3b: fluid smooth in time (K>=2)
                    mu_init,               # friction coefficient init (the fixed value when learn_mu=False)
                    learn_mu,              # recover mu from data
                    validation_check_interval,
                    epoch_checkpoint_interval,
                    keep_last_n_checkpoints,   # rotate; 0/None = keep all
                    slip_v0=1e-3, slip_tau=1e-4,   # slip gate (m/step)
                    compile_model=True,
                    early_stopping_patience=None):

    # Patience counts completed epochs, not the number of validation checks.
    if epochs < 1 or batch_size < 1 or accumulation_steps < 1:
        raise ValueError("epochs, batch_size, and accumulation_steps must be positive")
    if validation_check_interval < 1 or epoch_checkpoint_interval < 1:
        raise ValueError("validation and checkpoint intervals must be positive")
    if early_stopping_patience is not None and (
            not isinstance(early_stopping_patience, int) or early_stopping_patience < 1):
        raise ValueError("early_stopping_patience must be a positive integer or None")
    schedule = _resolve_curriculum(multistep, curriculum_epochs, curriculum_schedule)

    # Set the device for training (GPU if available, otherwise CPU).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Extract the stem of the save model path (filename without extension).
    stem = os.path.splitext(save_model_path)[0]

    # ---------------- data ----------------
    # build the training and validation datasets
    print("Building force training dataset (COM + rotation state)...")
    dataset_train, meta = build_force_dataset(train_range, trajectory_folder,
                                              weights_only=weights_only,
                                              unscale_data=unscale_data)
    print("Building force validation dataset...")
    dataset_val, _ = build_force_dataset(val_range, trajectory_folder,
                                         weights_only=weights_only,
                                         unscale_data=unscale_data)

    # If gravity is not specified, use the value from the dataset metadata or default to 9.615 m/s^2.
    if gravity is None:
        gravity = float(meta.get("g", 9.615))

    # Print the physics configuration for verification.
    print("=" * 70)
    print("FORCE-MODEL PHYSICS (must match the data generator):")
    print(f"  gravity = {gravity:.4f} m/s^2   dt = {dt:.6f} s   mass = {mass:.3f} kg")
    print(f"  drag baseline = {use_drag_baseline} (k/m init = {k_over_m}"
          f"{', LEARNABLE' if learn_k else ''})   "
          f"contact gate d0/tau = {contact_d0}/{contact_tau} m")
    print(f"  loss_mode = {loss_mode}"
          + ("   (per-node acceleration MSE, same objective as "
             "_unroll_chain_loss_accel)" if loss_mode == "accel"
             else "   (position MSE in block widths)"))
    print(f"  physics-loss weights: fric_dir={w_fric_dir} "
          f"fric_mag={w_fric_mag} fric_cone={w_fric_cone} "
          f"fluid_anchor={w_fluid_anchor} "
          f"fluid_smooth={w_fluid_smooth}")
    if meta:
        print(f"  replica_physics found in data: {meta}")
    print("=" * 70)

    # set up the rest node positions and adjacency graph for the mesh cube surface
    rest_nodes = torch.tensor(mesh_cube_surface(BLOCK_HALF_WIDTH * 2, nodes_per_edge),
                              dtype=torch.float32)

    # Number of nodes in the mesh cube surface.
    N = rest_nodes.shape[0]

    # Build the k-nearest neighbors adjacency graph for the mesh cube surface.
    edge_index = torch.tensor(knn_adjacency(rest_nodes.numpy(), k=nearest_neighbors),
                              dtype=torch.long)

    # Assign the adjacency graph to each validation dataset entry.
    for d in dataset_val:
        d["edge_index"] = edge_index          # rollout helper reads it from here

    # ---------------- normalization stats ----------------
    x_mean, x_std, e_mean, e_std, acc_mean, acc_std = _normalization_stats(
        dataset_train, rest_nodes, edge_index, Wall, h, noise_scale, use_wind)

    # Output scales, both [s_xy, s_xy, s_z] (equal x/y keeps them z-rotation
    # equivariant, so the rotation augmentation stays valid):
    #   scale_vec     - linear, from the acceleration-target stats
    #   ang_scale_vec - angular, from the empirical angular-acceleration stats

    # Compute the output scales for the linear and angular acceleration targets.
    s_xy = float(acc_std[:2].mean())
    scale_vec = torch.tensor([s_xy, s_xy, float(acc_std[2])])
    ang_scale_vec = _compute_angular_stats(dataset_train)
    print(f"  output scales: linear {scale_vec.tolist()} m/step^2 | "
          f"angular {ang_scale_vec.tolist()} rad/step^2")

    # ---------------- physics-informed loss module ----------------
    # Save the physics loss weights and initialize the physics-informed loss module.
    phys_weights = dict(w_fric_dir=w_fric_dir, w_fric_mag=w_fric_mag,
                        w_fric_cone=w_fric_cone,
                        w_fluid_anchor=w_fluid_anchor,
                        w_fluid_smooth=w_fluid_smooth)
    phys = PhysicsLosses(phi_g=gravity * dt * dt, ang_scale_vec=ang_scale_vec,
                         mu_init=mu_init, learn_mu=learn_mu,
                         k_init=k_over_m, learn_k=learn_k,
                         slip_v0=slip_v0, slip_tau=slip_tau)


    # Function to get the current value of k/m, either as a learnable tensor or a fixed float.
    def k_now():
        return phys.k_over_m if learn_k else k_over_m

    # Warn if learning k while using the drag baseline, as it affects identifiability.
    if learn_k and use_drag_baseline:
        print("  WARNING: learn_k with use_drag_baseline=True. The anchor then\n"
              "           reduces to ||residual||^2, which k cancels out of, so\n"
              "           k is identified only through the prediction loss and\n"
              "           only insofar as the anchor suppresses the residual.\n"
              "           use_drag_baseline=False is the cleanly identified case.")

    # Check if any physics loss is active and print relevant information.
    any_phys = any(v > 0 for v in phys_weights.values())
    if any_phys:
        print(f"  physics losses ON: {[k for k, v in phys_weights.items() if v > 0]}"
              f"   mu: " + (f"learnable, init {mu_init}" if learn_mu else
                            f"frozen at {mu_init}"))
        if phys_weights["w_fluid_smooth"] > 0 and multistep < 2:
            print("  WARNING: w_fluid_smooth needs multistep >= 2 for "
                  "consecutive predictions - it will contribute ZERO at K=1.")

    # Save the force configuration and normalization statistics for later use.
    force_cfg = dict(dt=dt, gravity=gravity, mass=mass,
                     use_drag_baseline=use_drag_baseline, k_over_m=k_over_m,
                     learn_k=learn_k,
                     contact_d0=contact_d0, contact_tau=contact_tau,
                     h=h, use_wind=use_wind, latent_dim=latent_dim,
                     L=message_passing_layers, K=repeat_blocks,
                     nodes_per_edge=nodes_per_edge,
                     nearest_neighbors=nearest_neighbors,
                     multistep=multistep, epochs=epochs,
                     early_stopping_patience=early_stopping_patience,
                     edge_features="displacement_and_magnitude",
                     scale_vec=scale_vec, ang_scale_vec=ang_scale_vec,
                     loss_mode=loss_mode,
                     w_fric_dir=w_fric_dir, w_fric_mag=w_fric_mag,
                     w_fric_cone=w_fric_cone,
                     w_fluid_anchor=w_fluid_anchor,
                     w_fluid_smooth=w_fluid_smooth,
                     mu_init=mu_init, learn_mu=learn_mu,
                     slip_v0=slip_v0, slip_tau=slip_tau,
                     noise_scale=noise_scale)

    # Define the path for saving normalization statistics and force configuration.
    norm_stats_path = stem + "_norms.pt"

    # Save the normalization statistics and force configuration to the defined path.
    torch.save({"x_mean": x_mean, "x_std": x_std, "e_mean": e_mean, "e_std": e_std,
                "acc_mean": acc_mean, "acc_std": acc_std, "force_cfg": force_cfg},
               norm_stats_path)
    print(f"Saved normalization stats + force config to {norm_stats_path}")

    # Move normalization statistics and other relevant tensors to the device for training.
    x_mean_g, x_std_g = x_mean.to(device), x_std.to(device)
    e_mean_g, e_std_g = e_mean.to(device), e_std.to(device)
    scale_vec_g = scale_vec.to(device)
    ang_scale_vec_g = ang_scale_vec.to(device)
    acc_mean_g, acc_std_g = acc_mean.to(device), acc_std.to(device)
    g_step = torch.tensor([0.0, 0.0, -gravity]) * dt * dt
    g_step_g = g_step.to(device)
    rest_g = rest_nodes.to(device)
    ei_g = edge_index.to(device)

    # Initialize the model with the given node and edge dimensions, latent dimension, and 
    # message passing layers.
    node_dim = x_mean.shape[0]
    edge_dim = e_mean.shape[0]
    model = ForceGNSModel(node_dim, edge_dim, latent_dim=latent_dim,
                          L=message_passing_layers, K=repeat_blocks).to(device)

    # Optionally compile the model for improved performance on supported hardware.
    if compile_model and torch.cuda.is_available() and _triton_available():
        model = torch.compile(model)

    # Move the physics parameters to the device for training.
    phys = phys.to(device)

    # Sets the optimizer and learning rate scheduler for training.
    optimizer = optim.Adam(list(model.parameters()) + list(phys.parameters()), lr=lr)
    scheduler = None
    if Learning_Rate_Scheduler == "decay":
        scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=(0.1) ** (1.0 / max(epochs, 1)))   # 10x decay over the run
    elif Learning_Rate_Scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Use the same curriculum as the optimizer-step budget calculation.
    curriculum_schedule = schedule if len(schedule) > 1 else None
    if curriculum_schedule is not None:
        print(f"Curriculum: {schedule} x {curriculum_epochs} epochs/phase")

    def _K_for_epoch(ep):
        if curriculum_schedule is None:
            return multistep
        return schedule[min(ep // curriculum_epochs, len(schedule) - 1)]

    train_loss_epochs, train_loss_values = [], []
    val_loss_epochs, val_loss_values = [], []
    mu_trace = []
    k_trace = []
    best_val_loss, best_val_epoch = float("inf"), -1
    loss_history_path = stem + "_loss_history.pt"
    global_step = 0         # optimizer steps taken, logged in the loss history
    chain_index = None
    chain_index_k = None
    eligible_since = None
    stopped_early = False

    # ---------------- epochs ----------------
    # Main training loop over the specified number of epochs.
    for epoch in range(epochs):

        # Record the start time of the epoch for timing purposes.
        t0 = time.time()

        # Determine the current value of multistep for this epoch based on the curriculum schedule.
        _K_now = _K_for_epoch(epoch)
        best_eligible = _K_now == multistep
        first_eligible_epoch = best_eligible and eligible_since is None
        if first_eligible_epoch:
            eligible_since = epoch + 1

        # Build the chain index only when K changes. The index is deterministic
        # for a fixed dataset, h, and K; iterate_force_chains() independently
        # shuffles its entries at the start of every epoch.
        if chain_index is None or _K_now != chain_index_k:
            chain_index = build_chain_index(dataset_train, h, _K_now)
            chain_index_k = _K_now
            if not chain_index:
                raise ValueError(f"No training windows available for history={h}, rollout K={_K_now}")

        # Record the time after building the chain index to measure the overhead of this operation.
        t1 = time.time()

        # Set the model to training mode and initialize accumulators for this epoch.
        model.train()
        total_loss = 0.0
        phys_accum = {}
        num_batches = 0
        optimizer.zero_grad(set_to_none=True)

        # Iterate over the force chains in the training dataset for this epoch.
        for bi, batch in enumerate(iterate_force_chains(
                dataset_train, chain_index, batch_size, h, _K_now, device,
                noise_scale=noise_scale)):

            # Apply a random rotation to the force chain to augment the training data.
            batch = rotate_force_chain(batch)
            # Extract the batch size from the current batch.
            B = batch["B"]
            # Construct the edge index for the current batch by offsetting the base edge index for each graph in the batch.
            edge_index_b = torch.cat([ei_g + b * N for b in range(B)], dim=1)

            # Compute the loss and raw terms for the current batch using the unrolled force loss function.
            loss, raw_terms = _unroll_force_loss(
                model, batch, _K_now, Wall, h, rest_g, edge_index_b, N,
                x_mean_g, x_std_g, e_mean_g, e_std_g, scale_vec_g,
                ang_scale_vec_g, acc_mean_g, acc_std_g, g_step_g, dt,
                use_wind=use_wind, use_drag_baseline=use_drag_baseline,
                k_over_m=k_now(), k_learnable=learn_k,
                contact_d0=contact_d0, contact_tau=contact_tau,
                loss_mode=loss_mode,
                phys=phys, phys_weights=phys_weights,
                report_slip=(bi == 0))

            # Backpropagate the loss for the current batch, taking into account gradient accumulation.
            (loss / accumulation_steps).backward()

            # Perform an optimizer step and reset gradients if the accumulation step threshold is reached.
            if (bi + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            # Accumulate the total loss and physics terms for reporting after the epoch.
            total_loss += float(loss.detach())
            for key, v in raw_terms.items():
                phys_accum[key] = phys_accum.get(key, 0.0) + v
            num_batches += 1

        # Flush the remainder if the number of batches is not a multiple of the accumulation steps.
        if num_batches % accumulation_steps != 0:      # flush the remainder
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1


        # Compute the average training loss for the epoch and record it.
        avg_train_loss = total_loss / max(num_batches, 1)
        epoch_num = epoch + 1
        train_loss_epochs.append(epoch_num)
        train_loss_values.append(float(avg_train_loss))

        # Record the time after completing the training for this epoch.
        t2 = time.time()

        #If a learning rate scheduler is being used, step it after each epoch.
        if scheduler is not None:
            scheduler.step()

        # Print the slip gate report if available.
        if getattr(phys, "last_slip_report", None) is not None:
            print(PhysicsLosses.fmt_slip_gate_report(phys.last_slip_report))

        # Print the accumulated physics terms and any learned parameters if applicable.
        if any_phys and phys_accum:
            nb = max(num_batches, 1)
            line = " | ".join(f"{k}: {v/nb:.3e}" for k, v in sorted(phys_accum.items()))
            print(f"  Physics terms (raw) | {line}")
            if learn_mu:
                mu_trace.append((epoch + 1, float(phys.mu.detach())))
                tag = ("" if w_fric_mag > 0 else
                       "  [NO GRADIENT PATH - frozen at mu_init; "
                       "needs w_fric_mag]")
                print(f"  recovered mu = {float(phys.mu.detach()):.4f}"
                      f"   (data generator used 0.198 for the replica sets)"
                      f"{tag}")
            if learn_k:
                k_trace.append((epoch + 1, float(phys.k_over_m.detach())))
                print(f"  recovered k/m = {float(phys.k_over_m.detach()):.5f}"
                      f"   (wind_error_analysis.py calibrated 0.0285)")

        # Perform validation at the specified interval, rolling out the model on the validation 
        # dataset and recording the results.
        # Validate at the patience deadline, even between regular checks, so a
        # last-minute improvement can reset the counter before stopping.
        last_improvement = best_val_epoch if best_val_epoch >= 0 else eligible_since
        patience_due = (early_stopping_patience is not None and best_eligible
                        and epoch_num - last_improvement >= early_stopping_patience)
        if (epoch % validation_check_interval == 0 or epoch_num == epochs
                or first_eligible_epoch or patience_due):

            # Determine the value of k/m to use for validation, either the learned value or the initial one.
            k_val = float(phys.k_over_m.detach()) if learn_k else k_over_m

            # Roll out the model on the validation dataset using the determined k/m value.
            rollout_center, rollout_angle = rollout_force_batched(
                model, dataset_val, Wall, h, rest_g,
                x_mean_g, x_std_g, e_mean_g, e_std_g, scale_vec_g,
                ang_scale_vec_g, g_step_g, dt, device,
                use_wind=use_wind, use_drag_baseline=use_drag_baseline,
                k_over_m=k_val,
                contact_d0=contact_d0, contact_tau=contact_tau,
                mass=mass)

            # Print the results of the validation rollout.
            print(f"  Rollout val | center: {rollout_center:.4f} | angle: {rollout_angle:.2f}")

            # Record the validation loss for this epoch.
            avg_val_loss = rollout_center
            val_loss_epochs.append(epoch_num)
            val_loss_values.append(float(avg_val_loss))

            # Check if the current validation loss is the best so far and if it is eligible to be 
            # considered the best.
            # A budget shorter than warm-up still needs a usable saved model.
            save_eligible = best_eligible or (epoch_num == epochs and best_val_epoch < 0)

            # Update the best validation loss and save the model if the current loss is the best and eligible.
            if avg_val_loss < best_val_loss and save_eligible:
                best_val_loss = float(avg_val_loss)
                best_val_epoch = epoch_num
                best_model_path = stem + "_best_model.pt"
                torch.save({"model_state_dict": model.state_dict(),
                            "k_over_m": k_val}, best_model_path)
                print(f"Best model saved to {best_model_path} at epoch {best_val_epoch}")

            # If the model is the best, but we are still in the warming up the curriculum, do not save it.
            elif avg_val_loss < best_val_loss:
                print(f"  (val {avg_val_loss:.6f} beats best, but curriculum K={_K_now} "
                      f"< final K={multistep} -- not saved)")

            last_improvement = best_val_epoch if best_val_epoch >= 0 else eligible_since
            stopped_early = (early_stopping_patience is not None and best_eligible
                             and epoch_num - last_improvement >= early_stopping_patience)

            # Print the summary of the current epoch, including training and validation losses.
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.9f} | "
                  f"Val Loss: {avg_val_loss:.9f}")

            
            # Save the current training and validation loss history, along with other relevant information.
            torch.save({"train_loss_epochs": train_loss_epochs,
                        "train_loss_values": train_loss_values,
                        "val_loss_epochs": val_loss_epochs,
                        "val_loss_values": val_loss_values,
                        "validation_check_interval": validation_check_interval,
                        "best_val_loss": best_val_loss,
                        "best_val_epoch": best_val_epoch,
                        "mu_trace": mu_trace,
                        # k_trace belongs in the PERIODIC save too, not only
                        # the final one: a run that is killed, crashes, or is
                        # read mid-flight otherwise loses the k history
                        # entirely and the report has no k drift.
                        "k_trace": k_trace,
                        "global_step": global_step,
                        "epochs_completed": epoch_num,
                        "early_stopping_patience": early_stopping_patience,
                        "stopped_early": stopped_early}, loss_history_path)
        else:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.9f}")

        # Print the timing information for the current epoch, including build and training times.
        print(f"Epoch {epoch+1}: build={t1-t0:.1f}s, train={t2-t1:.1f}s (K={_K_now})",
              flush=True)

        # Save a checkpoint of the model at regular intervals, including the current state of the model,
        # optimizer, and loss history.
        if (epoch + 1) % epoch_checkpoint_interval == 0:
            # Keep _physics.pt current too, so recovered_mu / recovered_k_over_m
            # survive an interrupted run.
            torch.save({"state_dict": phys.state_dict(),
                        "recovered_mu": float(phys.mu.detach()),
                        "recovered_k_over_m": float(phys.k_over_m.detach()),
                        "mu_mode": "learnable" if learn_mu else "frozen",
                        "k_mode": "learnable" if learn_k else "frozen",
                        "epoch": epoch + 1},
                       stem + "_physics.pt")

            checkpoint_path = stem + f"_epoch{epoch+1}.pt"
            torch.save({"epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "train_loss_epochs": train_loss_epochs,
                        "train_loss_values": train_loss_values,
                        "val_loss_epochs": val_loss_epochs,
                        "val_loss_values": val_loss_values,
                        "best_val_loss": best_val_loss,
                        "best_val_epoch": best_val_epoch}, checkpoint_path)
            prune_old_checkpoints(save_model_path, keep_last_n_checkpoints)
            kept = ("all" if not keep_last_n_checkpoints
                    else f"last {keep_last_n_checkpoints}")
            print(f"Checkpoint saved to {checkpoint_path}  (keeping {kept})")

        if stopped_early:
            print(f"Early stopping at epoch {epoch_num}: no new best validation "
                  f"center error for {early_stopping_patience} epochs "
                  f"(best epoch: {best_val_epoch}, best loss: {best_val_loss:.6f}).")
            break

    # Save the final model and physics parameters after training is complete.
    final_path = stem + "_final.pt"
    torch.save({"model_state_dict": model.state_dict(),
                "k_over_m": float(phys.k_over_m.detach()) if learn_k else k_over_m},
               final_path)
    print(f"Model saved to {final_path}")
    phys_path = stem + "_physics.pt"
    torch.save({"state_dict": phys.state_dict(),
                "recovered_mu": float(phys.mu),
                "recovered_k_over_m": float(phys.k_over_m.detach()),
                "k_mode": "learnable" if learn_k else "frozen",
                "mu_mode": "learnable" if learn_mu else "frozen",
                "weights": phys_weights}, phys_path)
    if any_phys and learn_mu:
        print(f"Recovered friction coefficient mu = {float(phys.mu):.4f} "
              f"(saved to {phys_path})")
    if learn_k:
        print(f"Recovered drag coefficient k/m = {float(phys.k_over_m.detach()):.5f} "
              f"(init / calibrated value was {k_over_m})")
    torch.save({"train_loss_epochs": train_loss_epochs,
                "train_loss_values": train_loss_values,
                "val_loss_epochs": val_loss_epochs,
                "val_loss_values": val_loss_values,
                "validation_check_interval": validation_check_interval,
                "best_val_loss": best_val_loss,
                "best_val_epoch": best_val_epoch,
                "mu_trace": mu_trace,
                "k_trace": k_trace,
                "global_step": global_step,
                "epochs_completed": epoch_num,
                "early_stopping_patience": early_stopping_patience,
                "stopped_early": stopped_early}, loss_history_path)
    print(f"Loss history saved to {loss_history_path} "
          f"(total optimizer steps: {global_step})")

    # Return the trained model to the caller.
    return model
