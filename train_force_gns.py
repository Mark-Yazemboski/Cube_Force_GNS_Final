"""
train_force_gns.py

Training and rollout utilities for the force-based Graph Neural Simulator.

The module prepares rigid-body trajectory data, builds node and edge features,
computes normalization statistics, and trains a force-prediction model. The
model predicts per-node contact forces and a body-level fluid wrench, which are
converted by the rigid dynamics layer into the cube's next position and
orientation. Rigid-state noise, multistep rollout training, curriculum
scheduling, rotation augmentation, and rollout-based validation are supported.

Optional physics-informed losses constrain friction direction and magnitude,
the Coulomb friction cone, analytic fluid drag, and temporal fluid smoothness.
The module also handles checkpointing, loss tracking, validation, and best-model
selection. No force labels are required; the primary supervision comes from
the observed trajectory states.

Run configuration and execution are provided by run_force_multi_step.py.
"""

import os
import re
import time
import math
import numpy as np
import torch
import torch.optim as optim

from generate_node_states import (mesh_cube_surface, knn_adjacency,
                                  unscale_position_velocity, relative_wind, add_random_walk_noise,
                                  BLOCK_HALF_WIDTH, BLOCK_WIDTH)
from evaluate_metrics import compute_metrics

# ---- the force model + dynamics layer ----
from force_gns import (ForceGNSModel, quat_wxyz_to_R, so3_exp, so3_log,
                       rigid_step, nodes_from_state, contact_weight,
                       assemble_contact_forces, fluid_wrench_from_raw,
                       drag_accel_step, I_OVER_M)

from physics_losses import PhysicsLosses


#This function takes one trajectory's node positions, applies random-walk noise to them,
#and returns the node features, edge features, and target accelerations for every timestep.
#It is only used to compute the normalization stats.
def _noisy_features_and_targets(positions, wind_vector, nodes_body, edge_index, Wall, h,
                                noise_scale, use_wind):
    """
    positions: (T, N, 3) clean node positions of one trajectory.
    Returns (x (M*N, node_dim), e (M*E, 8), y (M*N, 3)) stacked over the M
    timestep samples
    """

    # Add random-walk noise to the clean positions to simulate realistic perturbations.
    noisy_positions, noise = add_random_walk_noise(positions, noise_scale=noise_scale)

    #compute the relative displacement between sender and receiver nodes in the body frame
    sender = edge_index[0]
    receiver = edge_index[1]
    dU = nodes_body[sender] - nodes_body[receiver]
    dU_norm = torch.norm(dU, dim=1, keepdim=True)

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

    #Computes the edge features for each edge over the past h timesteps, including relative 
    #positions and displacements.
    pos_at_t = noisy_positions[h : T-1]                                # (M, N, 3)
    d_all = pos_at_t[:, sender] - pos_at_t[:, receiver]                # (M, E, 3)
    d_norm_all = torch.norm(d_all, dim=-1, keepdim=True)               # (M, E, 1)
    dU_broadcast = dU.unsqueeze(0).expand(M, -1, -1)
    dU_norm_broadcast = dU_norm.unsqueeze(0).expand(M, -1, -1)
    e_attr_all = torch.cat([d_all, d_norm_all, dU_broadcast, dU_norm_broadcast], dim=-1)  # (M, E, 8)

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
        sample = _noisy_features_and_targets(positions, d["wind"], rest_nodes, edge_index,
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
# Feature map used inside the unroll / rollout
# ======================================================================
# Builds the normalized node and edge features used during model rollouts.
# Unlike _noisy_features_and_targets(), this function operates on the current
# rolling state window, adds no training noise, and does not create targets.
# It is called at each unroll/rollout step to prepare the batched features that
# the force model consumes; _noisy_features_and_targets() is used only while
# computing normalization statistics from trajectory data.
def _build_features_for_unroll(pos_window, edge_index, nodes_body, Wall, wind,
                               x_mean, x_std, e_mean, e_std, B, N, use_wind):

    # Get the device from the most recent position tensor.
    device = pos_window[0].device

    # Compute finite difference velocities for the position window, most recent first.
    v_fd_list = []
    for k in range(len(pos_window) - 1):
        v_fd_list.append(pos_window[-(k+1)] - pos_window[-(k+2)])
    v_fd = torch.cat(v_fd_list, dim=-1)
    v_curr = v_fd_list[0]                                   # (B, N, 3)

    # Compute the current node features, including position, distance to wall
    x_t = pos_window[-1]
    wall_n = torch.as_tensor(Wall.normal,           dtype=x_t.dtype, device=device)
    wall_c = torch.as_tensor(Wall.center_position,  dtype=x_t.dtype, device=device)
    dist = torch.sum((x_t - wall_c) * wall_n, dim=-1, keepdim=True).clamp(-0.05, 0.5)  # (B, N, 1)

    #Starts a list to collect the different parts of the node features.
    node_parts = [v_fd]

    #If wind information is used, compute relative wind and its norm.
    if use_wind:
        u, u_norm = relative_wind(wind.unsqueeze(1), v_curr)   # (B,3)->(B,1,3) broadcasts
        node_parts += [u, u_norm]

    # Append the distance to the wall to the node features.
    node_parts.append(dist)

    # Concatenate all parts to form the final node feature matrix.
    x_node = torch.cat(node_parts, dim=-1).reshape(B * N, -1)

    # Compute edge features based on the current positions and body node positions.
    x_t_flat = x_t.reshape(B * N, 3)
    nodes_body_flat = nodes_body.reshape(B * N, 3)
    src, dst = edge_index[0], edge_index[1]

    # Compute the relative position vectors and their norms for the edges.
    d  = x_t_flat[src]        - x_t_flat[dst]
    dU = nodes_body_flat[src] - nodes_body_flat[dst]
    e_attr = torch.cat([d, torch.norm(d, dim=-1, keepdim=True),
                        dU, torch.norm(dU, dim=-1, keepdim=True)], dim=-1)

    # Normalize the node and edge features using the provided means and standard deviations.
    x_node = (x_node - x_mean) / x_std
    e_attr = (e_attr - e_mean) / e_std

    # Return the normalized node and edge features ready for the model.
    return x_node, e_attr


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


# ======================================================================
# Dataset: keep the rigid state, not just node positions
# ======================================================================

def build_force_dataset(traj_range, trajectory_folder, weights_only, unscale_data,
                        verbose_every=200):
    """
    Loads raw trajectory files, keeping per-frame COM and rotation matrix (the
    state the force model integrates) plus the wind vector. Also pulls the
    replica_physics dict (gravity, mu, ...) out of the first file that has one,
    so training can use the SAME gravity the data was generated with - a wrong
    g would be silently absorbed into the learned normal forces otherwise.

    Returns (dataset, meta) where dataset is a list of dicts
    {com (T,3), R (T,3,3), wind (3,), T} and meta holds replica_physics if found.
    """
    dataset, meta = [], {}

    # Iterate over the specified trajectory range and load each trajectory file.
    for n, throw_number in enumerate(traj_range):
        path = os.path.join(trajectory_folder, f"{throw_number}.pt")
        raw = torch.load(path, weights_only=weights_only)
        states = raw[0].float()

        # Unscale the position and velocity data if requested.
        # Used to revert any scaling applied to the raw trajectory data.
        # Only used when the trajectories were the papers trajectories
        if unscale_data:
            states = unscale_position_velocity(states)

        # Extract the center of mass (COM) and quaternion from the state.
        com = states[:, 0:3].contiguous()
        quat = states[:, 3:7]
        # Convert the quaternion to a rotation matrix.
        R = quat_wxyz_to_R(quat)


        # Extract the wind vector from the raw data if available.
        wind = torch.zeros(3)
        if len(raw) > 1:
            try:
                wind = torch.as_tensor(raw[1], dtype=torch.float32).reshape(3)
            except Exception:
                pass

        
        # Extract the replica_physics dictionary from the raw data if available.
        if not meta and len(raw) > 3 and isinstance(raw[3], dict):
            rp = raw[3].get("replica_physics", None)
            if isinstance(rp, dict):
                meta = dict(rp)

        # Append the processed trajectory data to the dataset list.
        dataset.append({"com": com, "R": R, "wind": wind, "T": com.shape[0]})

        # Print progress if verbose mode is enabled.
        if verbose_every and (n + 1) % verbose_every == 0:
            print(f"  loaded {n + 1} trajectories...", flush=True)

    # Return the complete dataset and any extracted metadata.
    return dataset, meta


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
    # Expand the rest nodes to match the batch size.
    rest_b = rest_nodes.unsqueeze(0).expand(B, -1, -1)

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
            pos_window, edge_index_b, rest_b, Wall, wind,
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


# This function performs a batched rollout of the force prediction model, 
# padding each trajectory to the maximum length and rolling in lockstep. 
# It returns the predicted forces and optionally per-trajectory results.
# Used for validation rollouts of the force prediction model.
def rollout_force_batched(model, trajs, Wall, h, rest_nodes,
                          x_mean, x_std, e_mean, e_std, scale_vec, ang_scale_vec,
                          g_step, dt, device,
                          use_wind, use_drag_baseline, k_over_m,
                          contact_d0, contact_tau, mass,
                          return_forces=False, return_per_traj=False):
    # Set the model to evaluation mode.
    model.eval()

    # Get the batch size and number of nodes.
    B = len(trajs)
    N = rest_nodes.shape[0]

    # Extract the lengths of each trajectory and determine the maximum length.
    lengths = [t["T"] for t in trajs]
    T_max = max(lengths)

    # Define a helper function to pad a tensor to a specified length along the first dimension.
    def _pad(x, T):
        if x.shape[0] == T:
            return x
        tail = x[-1:].expand(T - x.shape[0], *([-1] * (x.dim() - 1)))
        return torch.cat([x, tail], dim=0)

    # Pad the center of mass and rotation matrices for all trajectories to the maximum length.
    com_all = torch.stack([_pad(t["com"], T_max) for t in trajs]).to(device)
    R_all = torch.stack([_pad(t["R"], T_max) for t in trajs]).to(device)

    # gets the wind vectors for all trajectories and moves them to the device.
    wind = torch.stack([t["wind"] for t in trajs]).to(device)

    # Move the rest nodes to the device and create a batch-expanded version.
    rest_nodes = rest_nodes.to(device)
    rest_b = rest_nodes.unsqueeze(0).expand(B, -1, -1)

    # Prepare the batched edge index for all trajectories.
    ei = trajs[0]["edge_index"].to(device)
    edge_index_b = torch.cat([ei + b * N for b in range(B)], dim=1)

    # Prepare the wall normal and center position tensors and move them to the device.
    wall_n = torch.as_tensor(Wall.normal, dtype=torch.float32, device=device)
    wall_n = wall_n / wall_n.norm().clamp_min(1e-12)
    wall_c = torch.as_tensor(Wall.center_position, dtype=torch.float32, device=device)

    # Move the normalization and scaling tensors to the device.
    x_mean, x_std = x_mean.to(device), x_std.to(device)
    e_mean, e_std = e_mean.to(device), e_std.to(device)
    scale_vec = scale_vec.to(device)
    ang_scale_vec = ang_scale_vec.to(device)
    g_step = g_step.to(device)

    # Work from frame h: seed the window with the first h+1 true frames of
    # that view.
    com_fh = com_all[:, h:]
    R_fh = R_all[:, h:]
    L_max = com_fh.shape[1]

    # Extract the true node positions for the current frame window.
    true_nodes_fh = com_fh.unsqueeze(2) + torch.einsum('btij,nj->btni', R_fh, rest_nodes)

    # Initialize the predicted node positions and the position window with the first h+1 true frames.
    pred_nodes = [true_nodes_fh[:, i].clone() for i in range(h + 1)]
    pos_window = [true_nodes_fh[:, i] for i in range(h + 1)]

    # Initialize the previous and current center of mass and rotation matrices for the unrolling loop.
    com_prev, com_curr = com_fh[:, h - 1].clone(), com_fh[:, h].clone()
    R_prev, R_curr = R_fh[:, h - 1].clone(), R_fh[:, h].clone()

    # Initialize the dictionary to store forces if required.
    forces = {"F_contact": [], "F_fluid": [], "tau_contact": [], "tau_fluid": [],
              # per-node, split for visualization (Newtons)
              "node_normal": [], "node_tangent": []} if return_forces else None
    
    # Precompute dt squared and the scaled inertia for efficiency.
    dt2 = dt * dt
    m_I = mass * I_OVER_M                            # = I, the cube's inertia

    # Begin the unrolling loop without tracking gradients.
    with torch.no_grad():

        # Loop over the frames in the current window, starting from frame h.
        for _ in range(h, L_max - 1):

            # Build the input features for the current unroll step.
            x_node, e_attr = _build_features_for_unroll(
                pos_window, edge_index_b, rest_b, Wall, wind,
                x_mean, x_std, e_mean, e_std, B, N, use_wind)

            # Forward pass through the model to obtain raw contact and fluid forces.
            contact_raw, fluid_raw = model(x_node, edge_index_b, e_attr, B)

            # Extract the current node positions from the position window.
            cur_nodes = pos_window[-1]
            # Compute the distance from the current nodes to the wall along the wall normal.
            dist = ((cur_nodes - wall_c) * wall_n).sum(-1, keepdim=True)

            # Compute the contact weight based on the distance to the wall.
            c_w = contact_weight(dist, d0=contact_d0, tau=contact_tau)

            # Assemble the contact forces from the raw contact predictions.
            phi_c = assemble_contact_forces(contact_raw, c_w, wall_n, scale_vec)

            # Compute the fluid acceleration and angular acceleration from the raw fluid forces.
            a_fluid, alpha_fluid = fluid_wrench_from_raw(fluid_raw, scale_vec,
                                                         ang_scale_vec)

            # Add any extra accelerations, such as drag, to the fluid acceleration.
            extra_accel = a_fluid
            if use_drag_baseline:
                extra_accel = extra_accel + drag_accel_step(
                    wind, com_curr - com_prev, dt, k_over_m)



            if return_forces:
                # Convert the predicted contact and fluid accelerations into
                # physical forces: F = m*a/dt^2.
                F_c = mass * phi_c.sum(dim=1) / dt2
                F_f = mass * extra_accel / dt2

                # Compute each node's lever arm from the center of mass in
                # world coordinates for the contact torque calculation.
                r = torch.einsum('bij,nj->bni', R_curr, rest_nodes)

                # Compute contact and fluid torques: tau = I*alpha/dt^2.
                tau_c = mass * torch.cross(r, phi_c, dim=-1).sum(dim=1) / dt2
                tau_f = m_I * alpha_fluid / dt2

                # Split each node's contact acceleration into normal and
                # tangential components relative to the wall.
                phi_n = (phi_c * wall_n).sum(-1, keepdim=True) * wall_n

                # Convert the per-node normal and tangential components to
                # Newtons and store them for visualization.
                forces["node_normal"].append(mass * phi_n / dt2)
                forces["node_tangent"].append(mass * (phi_c - phi_n) / dt2)

                # Store the total contact/fluid forces and torques for this
                # rollout step.
                forces["F_contact"].append(F_c)
                forces["F_fluid"].append(F_f)
                forces["tau_contact"].append(tau_c)
                forces["tau_fluid"].append(tau_f)

            # Perform the rigid body step using the computed accelerations and angular accelerations.
            com_next, R_next = rigid_step(phi_c, com_prev, com_curr, R_prev,
                                          R_curr, rest_nodes, g_step,
                                          extra_accel=extra_accel,
                                          extra_alpha=alpha_fluid)

            # Compute the next node positions from the updated center of mass and rotation matrix.
            nxt = nodes_from_state(com_next, R_next, rest_nodes)

            # Append the predicted next node positions to the list and update the position window.
            pred_nodes.append(nxt)

            # Update the position window and the previous/current center of mass and rotation matrices.
            pos_window = pos_window[1:] + [nxt]
            com_prev, com_curr = com_curr, com_next
            R_prev, R_curr = R_curr, R_next

    # Stack the predicted node positions along the time dimension to form the final output tensor.
    pred_nodes = torch.stack(pred_nodes, dim=1)                # (B, L_max, N, 3)

    center_errors, angle_errors, per_traj = [], [], []
    rest_cpu = rest_nodes.cpu()

    # Compute metrics for each trajectory in the batch.
    for b in range(B):
        Lb = lengths[b] - h

        # Compute the metrics for the current trajectory up to the valid length Lb.
        m = compute_metrics(pred_nodes[b, :Lb].cpu(), true_nodes_fh[b, :Lb].cpu(), rest_cpu)

        # Append the computed metrics to the respective lists.
        center_errors.append(m["center_error"])
        angle_errors.append(m["angle_error_deg"])

        # Optionally store the per-trajectory metrics if requested.
        if return_per_traj:
            per_traj.append(m)

    # Compute the mean of the collected metrics and prepare the output.
    out = [float(np.mean(center_errors)), float(np.mean(angle_errors))]

    if return_forces:
        # Stack the predicted forces along the time dimension and move them to the CPU.
        forces = {k: torch.stack(v, dim=1).cpu() for k, v in forces.items()}

        # Append the stacked forces to the output list.
        out.append(forces)


    if return_per_traj:
        # Append the per-trajectory metrics along with the predicted and true node positions
        # and lengths to the output list.
        out.append((per_traj, pred_nodes.cpu(), true_nodes_fh.cpu(), lengths))

    # Return the collected output as a tuple.
    return tuple(out)

# Load a trained model from the specified folder and return the model along with its associated data.
def load_trained_model(model_folder, device, prefix=None, checkpoint="best"):

    # Auto-detect the model prefix if not provided.
    if prefix is None:
        prefixes = [f[:-len("_norms.pt")] for f in os.listdir(model_folder)
                    if f.endswith("_norms.pt")]
        assert len(prefixes) == 1, f"set the model prefix explicitly, found: {prefixes}"
        prefix = prefixes[0]

    # Load the normalization statistics for the model.
    norms = torch.load(os.path.join(model_folder, prefix + "_norms.pt"), weights_only=False)
    # Extract the force configuration from the loaded normalization statistics.
    cfg = norms["force_cfg"]

    # Rebuild the rest node positions and the adjacency graph based on the force configuration.
    rest_nodes = torch.tensor(
        mesh_cube_surface(BLOCK_HALF_WIDTH * 2, cfg["nodes_per_edge"]), dtype=torch.float32)

    # Build the adjacency graph using k-nearest neighbors based on the rest node positions.
    edge_index = torch.tensor(
        knn_adjacency(rest_nodes.numpy(), k=cfg["nearest_neighbors"]), dtype=torch.long)

    # Initialize the ForceGNS model with the loaded configuration and normalization statistics.
    model = ForceGNSModel(norms["x_mean"].shape[0], norms["e_mean"].shape[0],
                          latent_dim=cfg["latent_dim"], L=cfg["L"], K=cfg["K"])

    # Determine the checkpoint file name based on the specified checkpoint type.
    ckpt_name = prefix + ("_best_model.pt" if checkpoint == "best" else "_final.pt")

    # Load the checkpoint file containing the model's weights and other relevant information.
    ckpt = torch.load(os.path.join(model_folder, ckpt_name),
                      map_location=device, weights_only=False)

    # Extract the model state dictionary and the k/m value from the checkpoint.
    if "model_state_dict" in ckpt:
        sd, k_over_m = ckpt["model_state_dict"], ckpt["k_over_m"]
    else:
        # If the checkpoint does not contain a separate k/m value, use the one from the configuration.
        sd, k_over_m = ckpt, cfg["k_over_m"]
        if cfg.get("learn_k") and cfg["use_drag_baseline"]:
            print(f"WARNING: {ckpt_name} predates saving the learned k/m; rolling "
                  f"out with the init k/m = {k_over_m} instead of the learned value.")
            
    # Handle the case where the model was compiled with torch.compile and the state dict 
    # keys have an "_orig_mod." prefix.
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}

    # Load the state dictionary into the model and move it to the specified device for evaluation.
    model.load_state_dict(sd)

    # Move the model to the specified device and set it to evaluation mode.
    model.to(device).eval()

    # Return the initialized model, normalization statistics, configuration, rest node positions,
    # adjacency graph, checkpoint prefix, and k/m value.
    return model, norms, cfg, rest_nodes, edge_index, prefix, k_over_m



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
                    compile_model=True):

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

    #This will set up the curriculum learning schedule if applicable.
    #Curriculum basically means gradually increasing the multistep parameter over the course of training.
    if curriculum_epochs > 0 and multistep > 1:
        if curriculum_schedule is None:
            curriculum_schedule = []
            k = 1
            while k < multistep:
                curriculum_schedule.append(k)
                k *= 2
            curriculum_schedule.append(multistep)
        print(f"Curriculum: {curriculum_schedule} x {curriculum_epochs} epochs/phase")
    else:
        curriculum_schedule = None

    # Helper function to determine the current value of K based on the epoch and curriculum schedule.
    def _K_for_epoch(ep):
        if curriculum_schedule is None:
            return multistep
        phase = min(ep // curriculum_epochs, len(curriculum_schedule) - 1)
        return curriculum_schedule[phase]

    train_loss_epochs, train_loss_values = [], []
    val_loss_epochs, val_loss_values = [], []
    mu_trace = []
    k_trace = []
    best_val_loss, best_val_epoch = float("inf"), -1
    loss_history_path = stem + "_loss_history.pt"
    global_step = 0         # optimizer steps taken, logged in the loss history
    chain_index = None
    chain_index_k = None

    # ---------------- epochs ----------------
    # Main training loop over the specified number of epochs.
    for epoch in range(epochs):

        # Record the start time of the epoch for timing purposes.
        t0 = time.time()

        # Determine the current value of multistep for this epoch based on the curriculum schedule.
        _K_now = _K_for_epoch(epoch)

        # Build the chain index only when K changes. The index is deterministic
        # for a fixed dataset, h, and K; iterate_force_chains() independently
        # shuffles its entries at the start of every epoch.
        if chain_index is None or _K_now != chain_index_k:
            chain_index = build_chain_index(dataset_train, h, _K_now)
            chain_index_k = _K_now

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
        if epoch % validation_check_interval == 0:

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
            best_eligible = (multistep <= 1) or (curriculum_schedule is None) or (_K_now == multistep)

            # Update the best validation loss and save the model if the current loss is the best and eligible.
            if avg_val_loss < best_val_loss and best_eligible:
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
                        "global_step": global_step}, loss_history_path)
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
                "global_step": global_step}, loss_history_path)
    print(f"Loss history saved to {loss_history_path} "
          f"(total optimizer steps: {global_step})")

    # Return the trained model to the caller.
    return model
