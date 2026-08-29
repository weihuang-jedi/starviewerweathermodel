#!/usr/bin/env python3
"""
models/gnn.py
-------------
Icosahedral GNN Surrogate Model Backbone for Atmospheric Data Assimilation and Forecasting.
Supports 3D Directional Message Passing with Gradient Checkpointing and Out-of-Place Tensor Bounding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


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
        edge_index_vert: torch.Tensor = None
    ) -> torch.Tensor:
        batch_size, num_nodes, hidden_dim = x.shape

        # 1. Horizontal Message Passing
        src_h, dst_h = edge_index[0], edge_index[1]
        msg_h = self.lin_msg_h(x[:, src_h, :])
        
        agg_msg = torch.zeros_like(x)
        idx_h = dst_h.view(1, -1, 1).expand(batch_size, -1, hidden_dim)
        agg_msg = agg_msg.scatter_add(1, idx_h, msg_h)

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
    def __init__(
        self,
        in_vars: int = 14,          
        out_vars: int = 7,          
        num_static_feats: int = 4,  
        hidden_dim: int = 256,
        num_levels: int = 32,
        num_layers: int = 6
    ):
        super().__init__()
        self.in_vars = in_vars
        self.out_vars = out_vars
        self.num_static_feats = num_static_feats
        self.hidden_dim = hidden_dim
        self.num_levels = num_levels
        self.num_layers = num_layers

        if in_vars > 50:
            total_in_dim = in_vars
        else:
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

    def forward(
        self,
        x_dynamic: torch.Tensor,
        edge_index: torch.Tensor,
        edge_index_vert: torch.Tensor = None,
        static_topo: torch.Tensor = None
    ) -> torch.Tensor:
        x_dynamic = torch.nan_to_num(x_dynamic, nan=0.0, posinf=3.0, neginf=-3.0)
        batch_size, num_vars, num_levels, num_nodes = x_dynamic.shape

        x_0 = x_dynamic[:, 7:14, :, :]

        x_flat = x_dynamic.view(batch_size, num_vars * num_levels, num_nodes)

        if static_topo is not None:
            if static_topo.dim() == 2:
                static_topo = static_topo.unsqueeze(0).expand(batch_size, -1, -1)
            static_topo = torch.nan_to_num(static_topo, nan=0.0, posinf=1.0, neginf=0.0)
            x_flat = torch.cat([x_flat, static_topo], dim=1)

        x_flat = x_flat.permute(0, 2, 1)  # [Batch, Nodes, Channels]

        feat = self.encoder(x_flat)       # [Batch, Nodes, Hidden_Dim]

        if edge_index_vert is not None:
            horiz_edges_list = [edge_index + (k * num_nodes) for k in range(num_levels)]
            edge_index_3d_horiz = torch.cat(horiz_edges_list, dim=1)

            feat = feat.unsqueeze(1).expand(-1, num_levels, -1, -1).reshape(batch_size, num_levels * num_nodes, -1)

            for gnn in self.gnn_layers:
                if self.training:
                    feat = checkpoint(gnn, feat, edge_index_3d_horiz, edge_index_vert, use_reentrant=False)
                else:
                    feat = gnn(feat, edge_index_3d_horiz, edge_index_vert=edge_index_vert)

            feat = feat.view(batch_size, num_levels, num_nodes, -1).mean(dim=1)
        else:
            for gnn in self.gnn_layers:
                if self.training:
                    feat = checkpoint(gnn, feat, edge_index, None, use_reentrant=False)
                else:
                    feat = gnn(feat, edge_index, edge_index_vert=None)

        delta_gnn_flat = self.decoder(feat)
        delta_gnn = delta_gnn_flat.permute(0, 2, 1).view(batch_size, self.out_vars, self.num_levels, num_nodes)

        delta_gnn = torch.clamp(delta_gnn, min=-0.05, max=0.05)
        x_pred = x_0 + delta_gnn

        # Out-of-place physical bounds
        c0 = torch.clamp(x_pred[:, 0:1, :, :], min=5.10, max=5.85)   # T
        c1 = torch.clamp(x_pred[:, 1:2, :, :], min=-100.0, max=100.0)  # U
        c2 = torch.clamp(x_pred[:, 2:3, :, :], min=-100.0, max=100.0)  # V
        c3 = torch.clamp(x_pred[:, 3:4, :, :], min=-10.0, max=10.0)    # W
        c4 = torch.clamp(x_pred[:, 4:5, :, :], min=0.0, max=0.035)     # q
        c5 = torch.clamp(x_pred[:, 5:6, :, :], min=-10.0, max=1.0)    # ln(rho)
        c6 = torch.clamp(x_pred[:, 6:7, :, :], min=4.60, max=11.60)   # ln(P)

        x_pred_bounded = torch.cat([c0, c1, c2, c3, c4, c5, c6], dim=1)

        return torch.where(torch.isnan(x_pred_bounded), x_0, x_pred_bounded)
