import torch
import numpy as np

#The half width of the block used in the making of the dataset
BLOCK_HALF_WIDTH = 0.0524
BLOCK_WIDTH = 2.0 * BLOCK_HALF_WIDTH

#THIS NEEDS TO BE CHANGED WHEN WE MOVE AWAY FROM MOJOCO
TIMESTEP = 0.0001348
SUBSTEPS = 50
DT_RECORD = TIMESTEP * SUBSTEPS

def relative_wind(wind_ms, v_curr):
    """wind_ms (m/s) broadcastable to v_curr; v_curr is the most-recent FD velocity (m/step).
       Returns (u, ||u||) in meters-per-recorded-step."""
    u = wind_ms * DT_RECORD - v_curr
    return u, torch.norm(u, dim=-1, keepdim=True)

#takes in the raw data and converts it back to SI units using the conversion described in the paper
def unscale_position_velocity(scaled_tensor):
    unscaled_tensor = scaled_tensor.clone()
    unscaled_tensor[..., :3] *= BLOCK_HALF_WIDTH     # position
    unscaled_tensor[..., 7:10] *= BLOCK_HALF_WIDTH   # velocity
    return unscaled_tensor


#Creates a cube mesh surface centered at the origin with specified side length and number of nodes per edge
def mesh_cube_surface(side_length, nodes_per_edge):


    L = side_length / 2.0
    lin = np.linspace(-L, L, nodes_per_edge)

    nodes = []

    # ±X faces
    for x in [-L, L]:
        for y in lin:
            for z in lin:
                nodes.append([x, y, z])

    # ±Y faces
    for y in [-L, L]:
        for x in lin:
            for z in lin:
                nodes.append([x, y, z])

    # ±Z faces
    for z in [-L, L]:
        for x in lin:
            for y in lin:
                nodes.append([x, y, z])

    #removes all of the duplicates points, and returns the unique set of nodes
    return np.unique(np.array(nodes), axis=0)

#Finds the k-nearest neighbors of each node to create an adjacency list
def knn_adjacency(nodes, k):
    N = nodes.shape[0]
    diff = nodes[:, np.newaxis, :] - nodes[np.newaxis, :, :]  # (N,N,3)
    dist = np.linalg.norm(diff, axis=2)
    edge_list = []
    for i in range(N):
        knn_idx = np.argsort(dist[i])[1:k+1]  # exclude self
        for j in knn_idx:
            edge_list.append([i, j])
    edge_index = np.array(edge_list).T  # shape (2, num_edges)
    return edge_index


def add_random_walk_noise(positions, noise_scale):
    velocities = positions[1:] - positions[:-1]            # (T, N, 3)
    T, N, _ = velocities.shape

    # Generate all noise in one shot
    noise = torch.randn(T, N, 3, device=positions.device) * noise_scale

    # Add noise to velocities (vectorized)
    noisy_velocities = velocities + noise

    # Reconstruct positions: cumulative sum of noisy velocities + initial position
    new_positions = torch.empty_like(positions)
    new_positions[0] = positions[0]
    new_positions[1:] = positions[0:1] + torch.cumsum(noisy_velocities, dim=0)

    return new_positions, noise
