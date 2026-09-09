#!/usr/bin/env python3
"""
models/gnn.py
--------------
Icosahedral GNN Model Backbone for Atmospheric Data Assimilation and Forecasting.
Supports 3D Directional Message Passing with Gradient Checkpointing,
Degree Normalization for Pentagonal Topological Singularities, and
Zero-Delta Clamping for 12 Pentagonal Vertices to eliminate topological spikes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
# Import the filter class from your new filter file
from models.grid_filters import IcosahedralPentagonFilter

def compute_node_degree_normalization(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """
    Computes square-root inverse node degree scaling factors:
      - Hexagonal Nodes (degree 6) -> sqrt(6.0 / 6.0) = 1.0000
      - Pentagonal Nodes (degree 5) -> sqrt(5.0 / 6.0) = 0.9129
    Cancels out the 20% area deficiency without compounding message attenuation across GNN layers.
    """
    deg = torch.zeros(num_nodes, dtype=torch.float32, device=edge_index.device)
    src_nodes = edge_index[0]
    deg.index_add_(0, src_nodes, torch.ones_like(src_nodes, dtype=torch.float32))

    norm_factor = torch.sqrt(deg / 6.0)  # Gentle Pentagonal scaling ~0.9129
    return torch.clamp(norm_factor, min=0.5, max=1.5)


class Directional3DConvBlock(nn.Module):
    """Memory-efficient 3D Message-passing layer for icosahedral level grids."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.lin_msg_h = nn.Linear(hidden_dim, hidden_dim)
        self.lin_msg_v = nn.Linear(hidden_dim, hidden_dim)
        self.fc_update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_index_vert: torch.Tensor = None,
        degree_norm: torch.Tensor = None
    ) -> torch.Tensor:
        batch_size, num_nodes, hidden_dim = x.shape

        # 1. Horizontal Message Passing
        src_h, dst_h = edge_index[0], edge_index[1]
        msg_h = self.lin_msg_h(x[:, src_h, :])

        agg_msg = torch.zeros_like(x)
        idx_h = dst_h.view(1, -1, 1).expand(batch_size, -1, hidden_dim)
        agg_msg = agg_msg.scatter_add(1, idx_h, msg_h)

        # Apply degree normalization (handling 3D vertical level expansion)
        if degree_norm is not None:
            if degree_norm.shape[0] != num_nodes and num_nodes % degree_norm.shape[0] == 0:
                num_levels = num_nodes // degree_norm.shape[0]
                norm_scale = degree_norm.repeat(num_levels).view(1, num_nodes, 1)
            else:
                norm_scale = degree_norm.view(1, num_nodes, 1)

            agg_msg = agg_msg * norm_scale

        # 2. Vertical Message Passing
        if edge_index_vert is not None:
            src_v, dst_v = edge_index_vert[0], edge_index_vert[1]
            msg_v = self.lin_msg_v(x[:, src_v, :])
            idx_v = dst_v.view(1, -1, 1).expand(batch_size, -1, hidden_dim)
            agg_msg = agg_msg.scatter_add(1, idx_v, msg_v)

        # 3. Node Update
        update_input = torch.cat([x, agg_msg], dim=-1)
        updated_x = self.fc_update(update_input)

        return self.norm(x + updated_x)


class GraphConvBlock(Directional3DConvBlock):
    """Backward-compatible alias for Directional3DConvBlock."""
    pass


class IcosahedralGNNSurrogate(nn.Module):
    """
    Icosahedral GNN AI-Data Assimilation Model.
    Supports checkpoint loading from 452-channel weights (14 vars * 32 levels + 4 static topo features).
    Enforces Zero-Delta Clamping on 12 Pentagonal Vertices to eliminate topological spikes.
    """
    def __init__(
        self,
        in_vars: int = 14,
        out_vars: int = 7,
        num_static_feats: int = 4,
        hidden_dim: int = 256,
        num_levels: int = 32,
        num_layers: int = 4
    ):
        super().__init__()
        self.in_vars = in_vars
        self.out_vars = out_vars
        self.num_static_feats = num_static_feats
        self.hidden_dim = hidden_dim
        self.num_levels = num_levels
        self.num_layers = num_layers

        total_in_dim = (in_vars * num_levels) + num_static_feats

        self.encoder = nn.Linear(total_in_dim, hidden_dim)
        self.gnn_layers = nn.ModuleList([
            Directional3DConvBlock(hidden_dim=hidden_dim) for _ in range(num_layers)
        ])
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_vars * num_levels)
        )

        # Replaced eager initialization with a module placeholder
        self.pentagon_filter = None

        self.register_buffer("degree_norm_cache", None, persistent=False)
        self.register_buffer("pentagon_idx_cache", None, persistent=False)
        self.register_buffer("pentagon_neighbor_masks", None, persistent=False)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        nn.init.zeros_(self.decoder[-1].weight)
        if self.decoder[-1].bias is not None:
            nn.init.zeros_(self.decoder[-1].bias)

    def _apply_pentagon_1hop_filter(self, pred: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Applies a localized 1-hop neighbor average filter directly to the 12 pentagonal vertices
        for Temperature (channel 0) and Pressure (channel 6) fields.
        pred shape: [Batch, OutVars=7, Levels=32, Nodes=40962]
        """
        B, C, L, N = pred.shape  # N = 40962 horizontal mesh nodes

        if self.pentagon_idx_cache is None or self.pentagon_idx_cache.device != pred.device:
            edges_2d = edge_index % N
            row, col = edges_2d[0], edges_2d[1]

            pent_nodes_list = []
            neighbors_list = []

            for node_idx in range(N):
                connected = torch.cat([col[row == node_idx], row[col == node_idx]])
                unique_nbrs = torch.unique(connected)
                unique_nbrs = unique_nbrs[unique_nbrs != node_idx]

                if unique_nbrs.numel() == 5:
                    pent_nodes_list.append(node_idx)
                    neighbors_list.append(unique_nbrs)

            if len(pent_nodes_list) != 12:
                deg_counts = torch.tensor([
                    torch.unique(torch.cat([col[row == i], row[col == i]])).numel()
                    for i in range(N)
                ], device=pred.device)
                _, pent_nodes_tensor = torch.topk(deg_counts, k=12, largest=False)

                neighbors_list = []
                for p_node in pent_nodes_tensor:
                    connected = torch.cat([col[row == p_node], row[col == p_node]])
                    nbrs = torch.unique(connected)
                    nbrs = nbrs[nbrs != p_node][:5]
                    neighbors_list.append(nbrs)
                pent_nodes_list = pent_nodes_tensor.tolist()

            self.pentagon_idx_cache = torch.tensor(pent_nodes_list, dtype=torch.long, device=pred.device)
            self.pentagon_neighbor_masks = torch.stack(neighbors_list).to(pred.device)  # [12, 5]

        pred_smoothed = pred.clone()

        nbr_t_mean = torch.mean(pred[:, 0, :, self.pentagon_neighbor_masks], dim=-1)  # [B, L, 12]
        nbr_p_mean = torch.mean(pred[:, 6, :, self.pentagon_neighbor_masks], dim=-1)  # [B, L, 12]

        pred_smoothed[:, 0, :, self.pentagon_idx_cache] = nbr_t_mean
        pred_smoothed[:, 6, :, self.pentagon_idx_cache] = nbr_p_mean

        return pred_smoothed

    def forward(
        self,
        x_dynamic: torch.Tensor,
        edge_index: torch.Tensor,
        edge_index_vert: torch.Tensor = None,
        static_topo: torch.Tensor = None
    ) -> torch.Tensor:
        x_dynamic = torch.nan_to_num(x_dynamic, nan=0.0, posinf=3.0, neginf=-3.0)

        if x_dynamic.shape[1] == 7:
            x_dynamic = torch.cat([x_dynamic, x_dynamic], dim=1)

        batch_size, num_vars, num_levels, num_nodes = x_dynamic.shape
        x_0 = x_dynamic[:, 7:14, :, :]  # Extract background state
        x_flat = x_dynamic.view(batch_size, num_vars * num_levels, num_nodes)

        # Compute degree normalization vector once per grid size
        if self.degree_norm_cache is None or self.degree_norm_cache.shape[0] != num_nodes:
            self.degree_norm_cache = compute_node_degree_normalization(edge_index, num_nodes).to(x_dynamic.device)

        if static_topo is not None:
            if static_topo.dim() == 2:
                static_topo = static_topo.unsqueeze(0).expand(batch_size, -1, -1)
            static_topo = torch.nan_to_num(static_topo, nan=0.0, posinf=1.0, neginf=0.0)

            if static_topo.shape[1] > self.num_static_feats:
                static_topo = static_topo[:, :self.num_static_feats, :]

            x_flat = torch.cat([x_flat, static_topo], dim=1)

        x_flat = x_flat.permute(0, 2, 1)  # [Batch, Nodes, Channels=452]
        feat = self.encoder(x_flat)       # [Batch, Nodes, Hidden_Dim]

        if edge_index_vert is not None:
            horiz_edges_list = [edge_index + (k * num_nodes) for k in range(num_levels)]
            edge_index_3d_horiz = torch.cat(horiz_edges_list, dim=1)

            feat = feat.unsqueeze(1).expand(-1, num_levels, -1, -1).reshape(batch_size, num_levels * num_nodes, -1)

            for gnn in self.gnn_layers:
                if self.training:
                    feat = checkpoint(
                        gnn,
                        feat,
                        edge_index_3d_horiz,
                        edge_index_vert,
                        self.degree_norm_cache,
                        use_reentrant=False
                    )
                else:
                    feat = gnn(
                        feat,
                        edge_index_3d_horiz,
                        edge_index_vert=edge_index_vert,
                        degree_norm=self.degree_norm_cache
                    )

            feat = feat.view(batch_size, num_levels, num_nodes, -1).mean(dim=1)
        else:
            for gnn in self.gnn_layers:
                if self.training:
                    feat = checkpoint(
                        gnn,
                        feat,
                        edge_index,
                        None,
                        self.degree_norm_cache,
                        use_reentrant=False
                    )
                else:
                    feat = gnn(
                        feat,
                        edge_index,
                        edge_index_vert=None,
                        degree_norm=self.degree_norm_cache
                    )

        delta_gnn_flat = self.decoder(feat)
        delta_gnn = delta_gnn_flat.permute(0, 2, 1).view(batch_size, self.out_vars, self.num_levels, num_nodes)
        delta_gnn = torch.clamp(delta_gnn, min=-0.05, max=0.05)

        # Cache pentagon node indices if not present
        if self.pentagon_idx_cache is None or self.pentagon_idx_cache.device != x_dynamic.device:
            edges_2d = edge_index % num_nodes
            row, col = edges_2d[0], edges_2d[1]
            deg_counts = torch.tensor([
                torch.unique(torch.cat([col[row == i], row[col == i]])).numel()
                for i in range(num_nodes)
            ], device=x_dynamic.device)
            _, pent_nodes_tensor = torch.topk(deg_counts, k=12, largest=False)
            self.pentagon_idx_cache = pent_nodes_tensor

        # ZERO OUT residual deltas strictly on the 12 pentagon nodes before residual add
        delta_gnn_clean = delta_gnn.clone()
        delta_gnn_clean[:, 0, :, self.pentagon_idx_cache] = 0.0  # Zero out Temperature delta
        delta_gnn_clean[:, 6, :, self.pentagon_idx_cache] = 0.0  # Zero out Pressure delta

        # Residual update with background state x_0
        x_pred = x_0 + delta_gnn_clean

        # Physical bounds
        c0 = torch.clamp(x_pred[:, 0:1, :, :], min=5.10, max=5.85)    # T
        c1 = torch.clamp(x_pred[:, 1:2, :, :], min=-100.0, max=100.0)  # U
        c2 = torch.clamp(x_pred[:, 2:3, :, :], min=-100.0, max=100.0)  # V
        c3 = torch.clamp(x_pred[:, 3:4, :, :], min=-10.0, max=10.0)    # W
        c4 = torch.clamp(x_pred[:, 4:5, :, :], min=0.0, max=0.035)     # q
        c5 = torch.clamp(x_pred[:, 5:6, :, :], min=-2.50, max=1.0)    # ln(rho)
        c6 = torch.clamp(x_pred[:, 6:7, :, :], min=4.60, max=11.60)   # ln(P)

        x_pred_bounded = torch.cat([c0, c1, c2, c3, c4, c5, c6], dim=1)
        x_out = torch.where(torch.isnan(x_pred_bounded), x_0, x_pred_bounded)

        # Fetch filter_alpha if present, otherwise default to 0.35
        alpha_val = getattr(self, "filter_alpha", 0.35)

        # Lazy initialize the filter using edge_index on first execution
        if self.pentagon_filter is None or self.pentagon_filter.num_nodes != num_nodes:
            self.pentagon_filter = IcosahedralPentagonFilter(
                edge_index=edge_index % num_nodes,
                num_nodes=num_nodes,
                alpha=alpha_val
            ).to(x_out.device)

        # Apply localized anti-aliasing filter to target grid pentagons
        return self.pentagon_filter(x_out)
