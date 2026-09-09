# models/grid_filters.py
import torch
import torch.nn as nn

class IcosahedralPentagonFilter(nn.Module):
    """
    Localized anti-aliasing filter for icosahedral dual grids.
    Identifies 12 pentagon nodes (degree 5) and applies localized
    smoothing to them and their 1-ring neighbors while leaving
    hexagonal nodes untouched.
    """
    def __init__(self, edge_index: torch.Tensor, num_nodes: int, alpha: float = 0.35):
        super().__init__()
        self.num_nodes = num_nodes
        self.alpha = alpha

        # 1. Compute node degrees (incoming edges)
        src, dst = edge_index
        degrees = torch.zeros(num_nodes, dtype=torch.long, device=edge_index.device)
        degrees.scatter_add_(0, dst, torch.ones_like(dst, dtype=torch.long))

        # 2. Identify the 12 pentagon nodes (degree == 5)
        pentagon_mask = (degrees == 5)
        pentagon_indices = torch.where(pentagon_mask)[0]

        # 3. Find 1-ring neighbors of pentagon nodes
        neighbor_mask = torch.zeros(num_nodes, dtype=torch.bool, device=edge_index.device)
        for p in pentagon_indices:
            neighbors_out = dst[src == p]
            neighbors_in = src[dst == p]
            neighbor_mask[neighbors_out] = True
            neighbor_mask[neighbors_in] = True

        # Full affected region (pentagons + 1-ring neighbors)
        affected_mask = pentagon_mask | neighbor_mask
        self.register_buffer("affected_mask", affected_mask)

        # 4. Precompute sparse normalized adjacency matrix
        adj = self._build_normalized_adj(edge_index, num_nodes)
        self.register_buffer("adj_matrix", adj)

    def _build_normalized_adj(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Builds a row-normalized sparse adjacency matrix with self-loops."""
        device = edge_index.device
        src, dst = edge_index

        self_loops = torch.arange(num_nodes, device=device).unsqueeze(0).repeat(2, 1)
        full_edges = torch.cat([edge_index, self_loops], dim=1)

        row, col = full_edges[1], full_edges[0]

        in_degree = torch.zeros(num_nodes, device=device)
        in_degree.scatter_add_(0, row, torch.ones_like(row, dtype=torch.float))

        edge_weight = 1.0 / in_degree[row]

        indices = torch.stack([row, col], dim=0)
        sparse_adj = torch.sparse_coo_tensor(
            indices, edge_weight, (num_nodes, num_nodes), device=device
        ).coalesce()

        return sparse_adj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Supports 2D [N, C], 3D [B, N, C], or 4D [B, Vars, Levels, N] tensors.
        """
        orig_shape = x.shape
        
        # Handle 4D weather model outputs: [B, Vars, Levels, N]
        if x.dim() == 4:
            B, V, L, N = x.shape
            # Reshape to [B, N, Vars * Levels]
            x_3d = x.permute(0, 3, 1, 2).reshape(B, N, V * L)
            smoothed_3d = self._apply_smoothing(x_3d)
            # Reshape back to [B, Vars, Levels, N]
            return smoothed_3d.view(B, N, V, L).permute(0, 2, 3, 1)

        if x.dim() == 2:
            x = x.unsqueeze(0)

        return self._apply_smoothing(x).squeeze(0) if len(orig_shape) == 2 else self._apply_smoothing(x)

    def _apply_smoothing(self, x: torch.Tensor) -> torch.Tensor:
        """Applies spatial aggregation to [B, N, C] tensors."""
        B, N, C = x.shape
        x_flat = x.permute(1, 0, 2).reshape(N, B * C)

        smoothed_flat = torch.sparse.mm(self.adj_matrix, x_flat)
        smoothed = smoothed_flat.reshape(N, B, C).permute(1, 0, 2)

        smoothed = (1.0 - self.alpha) * x + self.alpha * smoothed

        mask = self.affected_mask.view(1, N, 1).expand(B, N, C)
        return torch.where(mask, smoothed, x)
