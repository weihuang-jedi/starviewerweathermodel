import os
import numpy as np
import torch


def generate_or_load_edge_index(num_nodes: int, edge_file: str = "") -> torch.Tensor:
    """Loads existing edge connectivity or constructs a synthetic k-NN edge graph."""
    if edge_file and os.path.exists(edge_file):
        print(f"[GRAPH] Loading precomputed edge topology from '{edge_file}'...")
        edge_index = torch.load(edge_file)
        if isinstance(edge_index, dict) and "edge_index" in edge_index:
            edge_index = edge_index["edge_index"]
        return edge_index.to(torch.long)

    print(f"[GRAPH] Generating synthetic icosahedral mesh graph for {num_nodes} nodes...")
    phi = np.linspace(0, np.pi, int(np.sqrt(num_nodes)))
    theta = np.linspace(0, 2 * np.pi, int(np.sqrt(num_nodes)))
    phi_m, theta_m = np.meshgrid(phi, theta)

    x = np.sin(phi_m) * np.cos(theta_m)
    y = np.sin(phi_m) * np.sin(theta_m)
    z = np.cos(phi_m)
    coords = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T[:num_nodes]

    from scipy.spatial import cKDTree
    tree = cKDTree(coords)
    _, indices = tree.query(coords, k=7)

    src_list, dst_list = [], []
    for i, neighbors in enumerate(indices):
        for n in neighbors[1:]:
            src_list.append(i)
            dst_list.append(n)

    return torch.tensor([src_list, dst_list], dtype=torch.long)

def generate_vertical_edge_index(num_nodes: int, num_levels: int) -> torch.Tensor:
    """
    Generates bidirectional vertical edges for a 3D grid [Levels * Nodes].
    Connects (level k, node i) with (level k+1, node i).
    Returns edge_index of shape [2, 2 * num_nodes * (num_levels - 1)].
    """
    edges_src = []
    edges_dst = []

    for k in range(num_levels - 1):
        level_k_offset = k * num_nodes
        level_k_plus_1_offset = (k + 1) * num_nodes

        nodes_k = torch.arange(num_nodes, dtype=torch.long) + level_k_offset
        nodes_k_plus_1 = torch.arange(num_nodes, dtype=torch.long) + level_k_plus_1_offset

        # Upward edges (k -> k+1)
        edges_src.append(nodes_k)
        edges_dst.append(nodes_k_plus_1)

        # Downward edges (k+1 -> k)
        edges_src.append(nodes_k_plus_1)
        edges_dst.append(nodes_k)

    edge_index_vert = torch.stack([
        torch.cat(edges_src),
        torch.cat(edges_dst)
    ], dim=0)

    return edge_index_vert
