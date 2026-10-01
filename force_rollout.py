"""Provide the shared machinery for running a force GNS model forward through a
trajectory. This module constructs normalized node and edge features from a
rolling position history, loads trained weights and their saved configuration,
and reconstructs the model's cube graph. Its batched rollout starts from
observed states and repeatedly feeds predicted states back into the model,
handling trajectories of different lengths and computing motion metrics.
It can also return predicted positions and physical forces and torques for
evaluation and visualization; training uses the same feature builder and
rollout function for consistency.
"""

import os

import numpy as np
import torch

from force_data import mesh_cube_surface, knn_adjacency, relative_wind, BLOCK_HALF_WIDTH
from evaluate_metrics import compute_metrics
from force_gns import (ForceGNSModel, rigid_step, nodes_from_state, contact_weight,
                       assemble_contact_forces, fluid_wrench_from_raw,
                       drag_accel_step, I_OVER_M)


# ======================================================================
# Feature map used inside the unroll / rollout
# ======================================================================
# Builds the normalized node and edge features used during model rollouts.
# Unlike _noisy_features_and_targets(), this function operates on the current
# rolling state window, adds no training noise, and does not create targets.
# It is called at each unroll/rollout step to prepare the batched features that
# the force model consumes; _noisy_features_and_targets() is used only while
# computing normalization statistics from trajectory data.
def _build_features_for_unroll(pos_window, edge_index, Wall, wind,
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

    # Each edge contains the current displacement vector and its magnitude.
    x_t_flat = x_t.reshape(B * N, 3)
    src, dst = edge_index[0], edge_index[1]

    # Compute the relative position vectors and their norms for the edges.
    d  = x_t_flat[src]        - x_t_flat[dst]
    e_attr = torch.cat([d, torch.norm(d, dim=-1, keepdim=True)], dim=-1)

    # Normalize the node and edge features using the provided means and standard deviations.
    x_node = (x_node - x_mean) / x_std
    e_attr = (e_attr - e_mean) / e_std

    # Return the normalized node and edge features ready for the model.
    return x_node, e_attr


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

    # Move the rest nodes to the rollout device.
    rest_nodes = rest_nodes.to(device)

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
                pos_window, edge_index_b, Wall, wind,
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
    if norms["e_mean"].numel() != 4 or norms["e_std"].numel() != 4:
        raise ValueError(
            "This checkpoint uses the old edge features. Retrain with the four "
            "displacement-and-magnitude features, or evaluate it using the old code.")
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
