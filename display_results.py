import os

import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import torch_geometric
from generate_node_states import get_gns_features, knn_adjacency, relative_wind

#This file contains functions for visualizing the results of the GNN model's predictions 

#Set device for PyTorch (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



#This function performs shape matching to align the predicted positions with the rest positions,
#which helps to correct for drift in the predictions over time.
def shape_match(pred_positions, rest_positions, masses=None, alpha=1.0):

    # pred_positions: (N, 3) tensor of current predicted positions
    # rest_positions: (N, 3) tensor of rest positions (e.g. initial configuration)
    # masses: optional (N,) tensor of node masses for weighted shape matching
    # alpha: blending factor [0,1] for how much to move toward shape-matched positions

    #gets the number of nodes from the predicted positions tensor, which is needed for the shape matching calculations.
    N = pred_positions.shape[0]

    #If masses are not provided, it defaults to uniform masses for all nodes. 
    #This is used in the shape matching calculations to compute the centers of mass and the Apq matrix.
    if masses is None:
        masses = torch.ones(N, device=pred_positions.device)

    #Compute centers of mass for the rest positions and the predicted positions. 
    #This is done by taking a weighted average of the positions using the masses, 
    #which gives the center of mass for each configuration.
    x0_cm = (rest_positions * masses[:, None]).sum(0) / masses.sum()
    x_cm = (pred_positions * masses[:, None]).sum(0) / masses.sum()

    #Finds the relative positions of the nodes to the center of mass for both the rest positions and the predicted positions.
    q = rest_positions - x0_cm   
    p = pred_positions - x_cm  

    #Computes the Apq matrix, which is a weighted covariance matrix between the rest positions relative to their
    #center of mass and the predicted positions relative to their center of mass.
    Apq = (masses[:, None, None] * p[:, :, None] @ q[:, None, :]).sum(0)  # (3,3)

    #Performs Singular Value Decomposition (SVD) on the Apq matrix to find the optimal rotation
    #that aligns the predicted positions with the rest positions.
    U, S, Vh = torch.linalg.svd(Apq)


    d = torch.linalg.det(U @ Vh)
    D = torch.diag(torch.tensor([1.0, 1.0, d], device=U.device))
    R = U @ D @ Vh

    #Applies the rotation to the predicted positions and translates them 
    #to align with the center of mass of the predicted positions.
    g = (R @ q.T).T + x_cm  # (N,3)

    #Blends the original predicted positions with the shape-matched positions using the alpha parameter,
    #which controls the influence of the shape matching on the final positions.
    #If alpha is 1, it fully applies the shape matching correction. If alpha is 0, it leaves the predicted positions unchanged.
    new_positions = pred_positions + alpha * (g - pred_positions)

    #Returns the new positions after applying shape matching.
    return new_positions


#This function performs a closed-loop rollout of the GNN model over a trajectory,
#Aswell as applies shape matching at each step to correct for drift.
def rollout_trajectory_feedback_shape_match(
    trajectory_folder,
    model,
    Wall,
    throw_number,
    nodes_per_edge=2,
    nearest_neighbors=4,
    h=2,
    rest_positions=None,
    accel_std=None,
    accel_mean=None,
    x_mean=None,
    x_std=None,
    e_mean=None,
    e_std=None,
    do_shape_match=True,
    shape_alpha=1.0,
    return_edge_info=False,
    weights_only_load=False,
    unscale_trajectory_data=False,
    warn = True,
    use_wind=False
):
    
    #Loads the trajectory data for the specified throw number from the dataset, and extracts the wind vector
    data = torch.load(os.path.join(trajectory_folder, f"{throw_number}.pt"), weights_only=False)
    try:
        wind_vector = data[1]  
    except:
        wind_vector = torch.zeros(3)
        if warn:
            print("Warning: Wind vector not found in data file. Setting wind vector to zero.")


    #First generates the node features, edge features, edge indices, and true positions for the specified trajectory in the dataset.
    node_feat, edge_feat, edge_index, true_positions = get_gns_features(
        Wall,
        throw_number,
        nodes_per_edge=nodes_per_edge,
        nearest_neighbors=nearest_neighbors,
        h=h,
        data_folder=trajectory_folder,
        weights_only=weights_only_load,
        unscale_data=unscale_trajectory_data
    )

    #Converts the edge indices and true positions to the appropriate device (CPU or GPU) for computation.
    edge_index = edge_index.long().to(device)
    true_positions = true_positions.to(device)


    #If rest positions are not provided, it defaults to using the first frame of the true positions as the rest configuration
    #for shape matching. Normally the rest positions would be passed in from the "mesh_cube_surface" function.
    if rest_positions is None:
        rest_positions = true_positions[0].clone()
    rest_positions = rest_positions.to(device)
    
    #Moves the normalization statistics to the appropriate device if they are provided, which will be used to denormalize
    #the model's predictions during the rollout.
    if accel_std is not None and accel_mean is not None:
        accel_std = accel_std.to(device)
        accel_mean = accel_mean.to(device)
    if x_mean is not None and x_std is not None:
        x_mean = x_mean.to(device)
        x_std = x_std.to(device)
    if e_mean is not None and e_std is not None:
        e_mean = e_mean.to(device)
        e_std = e_std.to(device)

    #Checks that there are at least 3 frames in the true positions, which is necessary for
    #the feedback rollout with a history of h=2.
    if true_positions.shape[0] < 3:
        raise ValueError("Need at least 3 frames for feedback rollout with h=2.")

    #Sets the initial predicted positions to be the same as the true positions for the first three frames,
    #which are used as the initial conditions for the rollout.
    pred_positions = [true_positions[t].clone() for t in range(h + 1)]

    #Performs the closed-loop rollout for the remaining frames in the trajectory, starting from frame 3.
    for _ in range(h, true_positions.shape[0] - 1):

        recent = pred_positions[-(h+1):]

        #This will take in the current predicted positions and the previous two predicted positions,
        #along with the edge indices, rest positions, and wall information, (Aswell as the normalization statistics if provided) 
        #and will output the node features and edge features that are used as input to the GNN model
        #for the current step of the rollout.
        x_node, e_attr = _build_feedback_features(
            recent, edge_index, rest_positions, Wall,
            x_mean=x_mean, x_std=x_std,
            e_mean=e_mean, e_std=e_std,
            wind_vector=wind_vector, use_wind=use_wind
        )

        #Constructs a PyTorch Geometric Data object with the node features, edge indices, and edge features,
        #and moves it to the appropriate device.
        data = torch_geometric.data.Data(
            x=x_node,
            edge_index=edge_index,
            edge_attr=e_attr
        ).to(device)

        #Runs a forward pass of the GNN model to predict the accelerations (a_t) for the current step, 
        #using the constructed Data object as input.
        with torch.no_grad():
            a_t = model(data)

        #Denormalizes the predicted accelerations if the normalization statistics are provided,
        if accel_std is not None and accel_mean is not None:
            a_t = a_t * accel_std + accel_mean

        #Integrates the predicted accelerations to get the next predicted positions (x_next) 
        #using a simple finite difference scheme,
        x_next = a_t + 2.0 * recent[-1] - recent[-2]

        #If shape matching is enabled, it applies the shape matching function to the predicted positions to correct for drift,
        if do_shape_match:  
            x_next = shape_match(x_next, rest_positions, alpha=shape_alpha)

        #Appends the new predicted positions to the list of predicted positions for the trajectory.
        pred_positions.append(x_next)

    #After the rollout is complete, it stacks the list of predicted positions into a single tensor
    #and moves it to the CPU for further processing or visualization.
    pred_positions = torch.stack(pred_positions, dim=0).cpu()

    #Moves the true positions to the CPU as well
    true_positions_cpu = true_positions.cpu()

    #If return_edge_info is True, it constructs a dictionary containing the edge indices 
    #and edge features (moved to CPU) and returns it along with the predicted positions and true positions.
    if return_edge_info:
        edge_info = {
            "edge_index": edge_index.cpu(),
            "edge_feat": edge_feat.cpu(),
        }
        return pred_positions, true_positions_cpu, edge_info

    return pred_positions, true_positions_cpu


#This function given the predicted and true positions, will show both of them in an amimated 3D plot 
#with the edges drawn between the nodes, which is useful for visualizing how well the model's predictions 
#match the true trajectory over time.
def animate_cube(
    pred_positions,
    true_positions=None,
    edge_info=None,
    interval=50,
    save_path=None
):
    #Sets the figure and 3D axis for the animation.
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    #Moves pred and true positions to CPU if they are not already, which is necessary for plotting with Matplotlib.
    pred_positions = pred_positions.cpu()
    if true_positions is not None:
        true_positions = true_positions.cpu()

    #Gets the edge indices from the edge_info dictionary if it is provided, and moves it to CPU if it is a tensor.
    edge_index = edge_info["edge_index"]
    if torch.is_tensor(edge_index):
        edge_index = edge_index.detach().cpu().numpy()


    #This block computes the limits for the 3D plot based on the range of the predicted and true positions,
    all_pos = pred_positions
    if true_positions is not None:
        all_pos = torch.cat([pred_positions, true_positions], dim=0)

    x_min, x_max = all_pos[:, :, 0].min(), all_pos[:, :, 0].max()
    y_min, y_max = all_pos[:, :, 1].min(), all_pos[:, :, 1].max()
    z_min, z_max = all_pos[:, :, 2].min(), all_pos[:, :, 2].max()
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)



    #Sets the axis labels and title for the plot, and initializes the scatter plots for the predicted positions (in red)
    #and true positions (in blue), as well as the line objects for the edges.
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Red = Prediction | Blue = Ground Truth')
    pred_scatter = ax.scatter([], [], [], c='r', label='Pred')

    if true_positions is not None:
        gt_scatter = ax.scatter([], [], [], c='b', alpha=0.5, label='GT')
        ax.legend()
    else:
        gt_scatter = None

    pred_edge_lines = [
        ax.plot([], [], [], c='r', alpha=0.25, linewidth=1.0)[0]
        for _ in range(edge_index.shape[1])
    ]

    if true_positions is not None:
        gt_edge_lines = [
            ax.plot([], [], [], c='b', alpha=0.2, linewidth=1.0)[0]
            for _ in range(edge_index.shape[1])
        ]
    else:
        gt_edge_lines = []


    #This function updates the positions of the predicted and true nodes, as well as the edges, for each frame of the animation.
    def update(frame):

        #Updates the scatter plot for the predicted positions with the new positions for the current frame.
        pred_scatter._offsets3d = (
            pred_positions[frame, :, 0].numpy(),
            pred_positions[frame, :, 1].numpy(),
            pred_positions[frame, :, 2].numpy()
        )

        #Updates the line objects for the edges based on the current predicted positions and the edge indices.
        for i in range(edge_index.shape[1]):
            src = int(edge_index[0, i])
            dst = int(edge_index[1, i])

            pred_edge_lines[i].set_data(
                [pred_positions[frame, src, 0].item(), pred_positions[frame, dst, 0].item()],
                [pred_positions[frame, src, 1].item(), pred_positions[frame, dst, 1].item()]
            )
            pred_edge_lines[i].set_3d_properties(
                [pred_positions[frame, src, 2].item(), pred_positions[frame, dst, 2].item()]
            )


        #If the true positions are provided, it updates the scatter plot for the true positions and the line objects
        #for the edges based on the true positions as well.
        if true_positions is not None:
            gt_scatter._offsets3d = (
                true_positions[frame, :, 0].numpy(),
                true_positions[frame, :, 1].numpy(),
                true_positions[frame, :, 2].numpy()
            )

            for i in range(edge_index.shape[1]):
                src = int(edge_index[0, i])
                dst = int(edge_index[1, i])

                gt_edge_lines[i].set_data(
                    [true_positions[frame, src, 0].item(), true_positions[frame, dst, 0].item()],
                    [true_positions[frame, src, 1].item(), true_positions[frame, dst, 1].item()]
                )
                gt_edge_lines[i].set_3d_properties(
                    [true_positions[frame, src, 2].item(), true_positions[frame, dst, 2].item()]
                )

        #Returns the artists that were updated for the current frame, which is necessary for Matplotlib's animation 
        #to know which objects to redraw.
        artists = [pred_scatter] + pred_edge_lines
        if gt_scatter is not None:
            artists = artists + [gt_scatter] + gt_edge_lines


        return tuple(artists)

    #This creates the animation using Matplotlib's FuncAnimation, which calls the update function for each frame 
    #of the animation to update the plot with the new positions and edges.
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=pred_positions.shape[0],
        interval=interval,
        blit=False
    )

    #Sets the axis equal so the scales of each axis are the same.
    ax.set_aspect('equal')

    
    #Saves the animation as a gif
    if save_path is not None:
        print(f"Saving animation to {save_path}...")
        ani.save(save_path, writer='pillow', fps=1000 // interval)
        print("Saved successfully.")
    
    plt.show()


#This function builds the node features and edge features for a given time step in the rollout, 
#using the current predicted positions, the previous two predicted positions, the edge indices,
#rest positions, wall information, and normalization statistics if provided.
def _build_feedback_features(positions_history, edge_index, rest_positions, Wall,
                             wind_vector=None, x_mean=None, x_std=None, e_mean=None, e_std=None, use_wind=False):
    # positions_history is a list of h+1 position tensors, most recent last
    x_t = positions_history[-1]

    # Build h velocity vectors via finite differences
    v_fd_list = []
    for k in range(len(positions_history) - 1):
        v_fd_list.append(positions_history[-(k+1)] - positions_history[-(k+2)])
    v_fd = torch.cat(v_fd_list, dim=-1)
    v_curr = v_fd_list[0]                                   # (N, 3)

    if Wall is not None:
        wall_n = torch.as_tensor(Wall.normal, dtype=x_t.dtype, device=x_t.device)
        wall_c = torch.as_tensor(Wall.center_position, dtype=x_t.dtype, device=x_t.device)
        b = torch.sum((x_t - wall_c) * wall_n, dim=-1, keepdim=True).clamp(-0.05, 0.5)

    N = x_t.shape[0]
    node_parts = [v_fd]
    if use_wind:
        u, u_norm = relative_wind(wind_vector.to(x_t.device), v_curr)
        node_parts += [u, u_norm]
    node_parts.append(b)
    x_node = torch.cat(node_parts, dim=-1)

    src, dst = edge_index[0], edge_index[1]
    d = x_t[src] - x_t[dst]
    d_norm = torch.norm(d, dim=-1, keepdim=True)

    if rest_positions is not None:
        d_u = rest_positions[src] - rest_positions[dst]
        d_u_norm = torch.norm(d_u, dim=-1, keepdim=True)
    else:
        d_u = torch.zeros_like(d)
        d_u_norm = torch.zeros_like(d_norm)

    e_attr = torch.cat([d, d_norm, d_u, d_u_norm], dim=-1)

    if x_mean is not None and x_std is not None:
        x_node = (x_node - x_mean) / x_std
    if e_mean is not None and e_std is not None:
        e_attr = (e_attr - e_mean) / e_std

    return x_node, e_attr


#This function displays the meshed cube with the initial node positions and edges, 
#which is useful for visualizing how the nodes are arranged on the cube and how the edges connect them.
def display_meshed_cube(points, edge_index=None, nearest_neighbors=4):

    #Converts the input points and edge indices to numpy arrays if they are PyTorch tensors, 
    #which is necessary for plotting with Matplotlib.
    if torch.is_tensor(points):
        points_np = points.detach().cpu().numpy()
    else:
        points_np = points

    if edge_index is None:
        edge_index = knn_adjacency(points_np, k=nearest_neighbors)
    elif torch.is_tensor(edge_index):
        edge_index = edge_index.detach().cpu().numpy()


    #Sets the figure and 3D axis for the animation.
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    #Plots the points of the meshed cube as red dots in the 3D space.
    ax.scatter(points_np[:, 0], points_np[:, 1], points_np[:, 2], c='r')

    #Draw directed edges in edge_index.
    for i in range(edge_index.shape[1]):
        src = int(edge_index[0, i])
        dst = int(edge_index[1, i])

        ax.plot(
            [points_np[src, 0], points_np[dst, 0]],
            [points_np[src, 1], points_np[dst, 1]],
            [points_np[src, 2], points_np[dst, 2]],
            c='k',
            alpha=0.35,
            linewidth=1.0
        )

    #Sets the axis labels and title for the plot
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Meshed Cube Surface Points + Edges')

    plt.show()





#This function displays an origional trajectory aswell as a randomly rotated version of the same trajectory, 
#which is a common data augmentation technique used in training GNNs for physical systems.
#This visualization is useful in determining if the augmentation is working properly, and also gives a visual 
#intuition for how the trajectories change with different rotations.
def animate_augmented_data(
    Wall,
    throw_number,
    nodes_per_edge=2,
    nearest_neighbors=4,
    h=2,
    interval=50,
    save_path=None
):

    #Generates the node features, edge features, edge indices, and positions for the specified trajectory in the dataset,
    node_feat, edge_feat, edge_index, positions = get_gns_features(
        Wall,
        throw_number,
        nodes_per_edge=nodes_per_edge,
        nearest_neighbors=nearest_neighbors,
        h=h
    )

    #Moves the positions, edge indices, node features, and edge features to the CPU for plotting with Matplotlib.
    positions = positions.cpu()
    edge_index = edge_index.cpu()
    node_feat = node_feat.cpu()
    edge_feat = edge_feat.cpu()

    
    #Extracts the velocities from the node features, which are assumed to be in the first 6 dimensions
    #of the node features (3 for velocity and 3 for previous velocity).
    velocities = node_feat[:, :, 0:6].reshape(
        node_feat.shape[0],
        node_feat.shape[1],
        2,
        3
    ).sum(dim=2)

    #Applies a random rotation around the Z-axis to the positions and velocities to create an augmented version of the trajectory.
    theta = torch.rand(1) * 2 * torch.pi
    c = torch.cos(theta)
    s = torch.sin(theta)

    R = torch.tensor([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ]).squeeze()

    #Rotates the positions and velocities using the rotation matrix R, which is applied to each position and velocity vector
    #in the trajectory.
    pos_rot = positions @ R.T
    vel_rot = velocities @ R.T

    
    #Extracts the edge features for the distance vectors and their norms, which are assumed to be in the first 8 dimensions
    d = edge_feat[:, :, 0:3]
    d_norm = edge_feat[:, :, 3:4]
    d_u = edge_feat[:, :, 4:7]
    d_u_norm = edge_feat[:, :, 7:8]

    #Applies the same rotation to the edge features for the distance vectors, which ensures that the edges are consistent 
    #with the rotated positions.
    d_rot = d @ R.T
    d_u_rot = d_u @ R.T

    #Combines the rotated edge features back into a single tensor, which will be used for drawing the edges in the animation.
    edge_feat_rot = torch.cat([d_rot, d_norm, d_u_rot, d_u_norm], dim=-1)

    #Extracts the distance vectors from the edge features for both the original and rotated trajectories, 
    #which will be used to draw the edges in the animation.
    edge_d = edge_feat[:, :, 0:3] 
    edge_d_rot = edge_feat_rot[:, :, 0:3]

    #Converts the positions, velocities, and edge features to numpy arrays for plotting with Matplotlib.
    positions = positions.numpy()
    pos_rot = pos_rot.numpy()
    velocities = velocities.numpy()
    vel_rot = vel_rot.numpy()
    edge_d = edge_d.numpy()
    edge_d_rot = edge_d_rot.numpy()

    #Sets up the figure and 3D axis for the animation
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    #Combines the original and rotated positions into a single array to compute the limits for the 3D plot, 
    #which ensures that both trajectories are fully visible in the plot.
    all_pos = torch.cat([torch.tensor(positions),
                         torch.tensor(pos_rot)], dim=0)

    ax.set_xlim(all_pos[:, :, 0].min(), all_pos[:, :, 0].max())
    ax.set_ylim(all_pos[:, :, 1].min(), all_pos[:, :, 1].max())
    ax.set_zlim(all_pos[:, :, 2].min(), all_pos[:, :, 2].max())



    #Sets axis labels and title for the plot
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Blue = Original | Red = Rotated")

    #This function updates the positions of the nodes, the edges, and the velocity vectors for both the original 
    #and rotated trajectories for each frame of the animation.
    def update(frame):

        #Clears the current plot to prepare for drawing the new positions and edges for the current frame.
        ax.cla() 

        #Re-set limits after clearing
        ax.set_xlim(all_pos[:, :, 0].min(), all_pos[:, :, 0].max())
        ax.set_ylim(all_pos[:, :, 1].min(), all_pos[:, :, 1].max())
        ax.set_zlim(all_pos[:, :, 2].min(), all_pos[:, :, 2].max())

        #Plots nodes for original trajectory in blue and rotated trajectory in red for the current frame of the animation.
        ax.scatter(
            positions[frame, :, 0],
            positions[frame, :, 1],
            positions[frame, :, 2],
            c='b'
        )

        ax.scatter(
            pos_rot[frame, :, 0],
            pos_rot[frame, :, 1],
            pos_rot[frame, :, 2],
            c='r'
        )

        #Plots edges from edge_feat displacement d using sender anchor.
        #For each edge: d = x_sender - x_receiver, so receiver = sender - d.
        if edge_d is not None and edge_d_rot is not None:
            for i in range(edge_index.shape[1]):
                src = int(edge_index[0, i])

                src_pos = positions[frame, src]
                dst_from_feat = src_pos - edge_d[frame, i]

                src_pos_rot = pos_rot[frame, src]
                dst_from_feat_rot = src_pos_rot - edge_d_rot[frame, i]

                ax.plot(
                    [src_pos[0], dst_from_feat[0]],
                    [src_pos[1], dst_from_feat[1]],
                    [src_pos[2], dst_from_feat[2]],
                    c='b',
                    alpha=0.3
                )

                ax.plot(
                    [src_pos_rot[0], dst_from_feat_rot[0]],
                    [src_pos_rot[1], dst_from_feat_rot[1]],
                    [src_pos_rot[2], dst_from_feat_rot[2]],
                    c='r',
                    alpha=0.3
                )

        #Plots velocity vectors for the original and rotated trajectories using quiver, 
        #which draws arrows representing the velocity at each node.
        ax.quiver(
            positions[frame, :, 0],
            positions[frame, :, 1],
            positions[frame, :, 2],
            velocities[frame, :, 0],
            velocities[frame, :, 1],
            velocities[frame, :, 2],
            color='b',
            length=0.05,
            normalize=True
        )

        ax.quiver(
            pos_rot[frame, :, 0],
            pos_rot[frame, :, 1],
            pos_rot[frame, :, 2],
            vel_rot[frame, :, 0],
            vel_rot[frame, :, 1],
            vel_rot[frame, :, 2],
            color='r',
            length=0.05,
            normalize=True
        )

    #Creates the animation using Matplotlib's FuncAnimation, which calls the update function for each 
    #frame of the animation to update the plot with the new positions, edges, and velocity vectors.
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=positions.shape[0],
        interval=interval,
        blit=False
    )

    #Sets the axis equal so the scales of each axis are the same, which ensures that the trajectories are not distorted in the plot.
    ax.set_aspect('equal')


    #Saves the animation as a gif
    if save_path is not None:
        print(f"Saving animation to {save_path}...")
        ani.save(save_path, writer='pillow', fps=1000 // interval)
        print("Saved successfully.")
    
    plt.show()

#This function plots the training and validation loss curves over epochs
def plot_loss_curves(
    train_loss_epochs,
    train_loss_values,
    val_loss_epochs=None,
    val_loss_values=None,
    title="Training and Validation Loss",
    save_path=None,
    show_plot=True,
):
    if len(train_loss_epochs) != len(train_loss_values):
        raise ValueError("train_loss_epochs and train_loss_values must have same length")

    if val_loss_epochs is not None and val_loss_values is not None:
        if len(val_loss_epochs) != len(val_loss_values):
            raise ValueError("val_loss_epochs and val_loss_values must have same length")

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(train_loss_epochs, train_loss_values, label="Train Loss", linewidth=2.0)

    if val_loss_epochs is not None and val_loss_values is not None and len(val_loss_epochs) > 0:
        ax.plot(
            val_loss_epochs,
            val_loss_values,
            label="Validation Loss",
            linewidth=2.0,
            marker="o",
            markersize=4,
        )

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved loss curve to {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax