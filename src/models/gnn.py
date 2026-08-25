#!/usr/bin/env python3
"""
models/gnn.py
-------------
Icosahedral GNN Surrogate Model Backbone for Atmospheric Data Assimilation and Forecasting.
Implements Extrapolation Baseline Scheme:
    X_pred(t+6h) = X(t0) + (X(t0) - X(t-6h)) + delta_X_GNN
Guarantees physically non-zero initializations and realistic atmospheric state profiles.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvBlock(nn.Module):
    """Message-passing graph convolution layer operating over icosahedral mesh topologies."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.fc_msg = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.fc_update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, hidden_dim = x.shape
        src_nodes, dst_nodes = edge_index[0], edge_index[1]

        x_src = x[:, src_nodes, :]
        x_dst = x[:, dst_nodes, :]

        msg_input = torch.cat([x_src, x_dst], dim=-1)
        messages = self.fc_msg(msg_input)

        aggregated_msg = torch.zeros_like(x)
        index = dst_nodes.view(1, -1, 1).expand(batch_size, -1, hidden_dim)
        aggregated_msg.scatter_add_(1, index, messages)

        update_input = torch.cat([x, aggregated_msg], dim=-1)
        updated_x = self.fc_update(update_input)

        return self.norm(x + updated_x)


class IcosahedralGNNSurrogate(nn.Module):
    """
    Icosahedral GNN Network using Linear Trend Extrapolation + GNN Variational Residuals:
        X_pred = X_0 + (X_0 - X_m6) + delta_X_GNN
    """
    def __init__(
        self,
        in_vars: int = 14,          # Input variables (7 vars @ t-6h, 7 vars @ t0)
        out_vars: int = 7,          # Output variables (ln_t, u, v, w, q, ln_rho, ln_p)
        num_static_feats: int = 4,  # Static terrain features (Elevation + Land-Sea Mask)
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
            GraphConvBlock(hidden_dim=hidden_dim) for _ in range(num_layers)
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
        
        # Zero-initialize output projection layer so delta_X_GNN starts at 0.0
        nn.init.zeros_(self.decoder[-1].weight)
        if self.decoder[-1].bias is not None:
            nn.init.zeros_(self.decoder[-1].bias)

    def forward(
        self,
        x_dynamic: torch.Tensor,
        edge_index: torch.Tensor,
        static_topo: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            x_dynamic: Dynamic trajectory [Batch, 14, 32, Nodes]
                       Channels 0..6: X_-6h | Channels 7..13: X_0
            edge_index: Edge graph topology [2, Num_Edges]
            static_topo: Surface features [Batch, Static_Feats (3 or 4), Nodes]
        """
        x_dynamic = torch.nan_to_num(x_dynamic, nan=0.0, posinf=10.0, neginf=-10.0)
        if static_topo is not None:
            static_topo = torch.nan_to_num(static_topo, nan=0.0, posinf=1.0, neginf=0.0)

        batch_size, num_vars, num_levels, num_nodes = x_dynamic.shape

        # 1. Extract X_-6h and X_0 states from input trajectory
        x_m6 = x_dynamic[:, 0:7, :, :]
        x_0  = x_dynamic[:, 7:14, :, :]

        # 2. Compute Linear Trend Baseline Extrapolation: X_trend = X_0 + (X_0 - X_m6)
        x_trend = x_0 + (x_0 - x_m6)

        # 3. Predict GNN Variational Delta Correction (delta_X_GNN)
        x_flat = x_dynamic.view(batch_size, num_vars * num_levels, num_nodes)

        if static_topo is not None:
            if static_topo.dim() == 2:
                static_topo = static_topo.unsqueeze(0).expand(batch_size, -1, -1)
            x_flat = torch.cat([x_flat, static_topo], dim=1)

        x_flat = x_flat.permute(0, 2, 1)  # [Batch, Nodes, Channels] -> [1, 40962, 452]

        feat = self.encoder(x_flat)
        for gnn in self.gnn_layers:
            feat = gnn(feat, edge_index)

        delta_gnn_flat = self.decoder(feat)  # [Batch, Nodes, Out_Vars * Levels]
        delta_gnn = delta_gnn_flat.permute(0, 2, 1).view(batch_size, self.out_vars, self.num_levels, num_nodes)

        # 4. Final Output: X_pred = X_trend + delta_X_GNN
        x_pred = x_trend + delta_gnn

        return torch.nan_to_num(x_pred, nan=0.0, posinf=15.0, neginf=-15.0)
