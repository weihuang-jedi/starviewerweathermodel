#!/usr/bin/env python3
"""
models/gnn.py
-------------
Icosahedral GNN Surrogate Model Backbone for Atmospheric Data Assimilation and Forecasting.
Implements residual trend prediction (x_{t+1} = x_t + delta_x), terrain-following 3D vertical state grids,
static topography feature conditioning (surface elevation + land-sea mask), flexible multi-step 4D
trajectory representations, Xavier weight initialization, and input/output sanitization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvBlock(nn.Module):
    """
    Message-passing graph convolution layer operating over icosahedral mesh topologies.
    Aggregates edge features from neighboring mesh nodes and updates node embeddings.
    """
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
        """
        Args:
            x: Node feature embeddings [Batch, Nodes, Hidden_Dim]
            edge_index: Graph connectivity edge list [2, Num_Edges]

        Returns:
            Updated node feature embeddings [Batch, Nodes, Hidden_Dim]
        """
        batch_size, num_nodes, hidden_dim = x.shape
        src_nodes, dst_nodes = edge_index[0], edge_index[1]

        # Gather node features for source and destination edge pairs
        x_src = x[:, src_nodes, :]  # [Batch, Num_Edges, Hidden_Dim]
        x_dst = x[:, dst_nodes, :]  # [Batch, Num_Edges, Hidden_Dim]

        # Compute pairwise edge messages
        msg_input = torch.cat([x_src, x_dst], dim=-1)  # [Batch, Num_Edges, Hidden_Dim * 2]
        messages = self.fc_msg(msg_input)              # [Batch, Num_Edges, Hidden_Dim]

        # Scatter-add messages back onto destination nodes
        aggregated_msg = torch.zeros_like(x)
        index = dst_nodes.view(1, -1, 1).expand(batch_size, -1, hidden_dim)
        aggregated_msg.scatter_add_(1, index, messages)

        # Update node states with residual connection
        update_input = torch.cat([x, aggregated_msg], dim=-1)  # [Batch, Nodes, Hidden_Dim * 2]
        updated_x = self.fc_update(update_input)

        return self.norm(x + updated_x)


class IcosahedralGNNSurrogate(nn.Module):
    """
    Global Icosahedral GNN Surrogate Network with Residual Learning Architecture.
    Predicts the 6-hour atmospheric state increment (delta_x) such that x(t+1) = x(t) + delta_x.
    """
    def __init__(
        self,
        in_vars: int = 14,          # Input dynamic variables (7 vars x 2 timesteps: t-6h, t0)
        out_vars: int = 7,          # Output dynamic variables (ln_t, u, v, w, q, ln_rho, ln_p)
        num_static_feats: int = 2,  # Static terrain features (Surface Elevation + Land-Sea Mask)
        hidden_dim: int = 256,      # Expanded hidden capacity
        num_levels: int = 32,
        num_layers: int = 6         # Expanded depth for multi-scale message passing
    ):
        super().__init__()
        self.in_vars = in_vars
        self.out_vars = out_vars
        self.num_static_feats = num_static_feats
        self.hidden_dim = hidden_dim
        self.num_levels = num_levels
        self.num_layers = num_layers

        # Total input dimension per node: (In_Vars * Levels) + Static_Topography_Features
        total_in_dim = (in_vars * num_levels) + num_static_feats

        # Node Feature Encoder
        self.encoder = nn.Linear(total_in_dim, hidden_dim)

        # Sequence of GNN Interaction Layers
        self.gnn_layers = nn.ModuleList([
            GraphConvBlock(hidden_dim=hidden_dim) for _ in range(num_layers)
        ])

        # State Tendency Decoder (Predicts delta_x)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_vars * num_levels)
        )

        # Initialize network weights
        self._init_weights()

    def _init_weights(self):
        """Applies Xavier Uniform initialization with small residual output weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Initialize last layer of decoder near zero to ensure small initial trend increments
        nn.init.uniform_(self.decoder[-1].weight, a=-1e-4, b=1e-4)
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
            x_dynamic: Dynamic state trajectory [Batch, In_Vars=14, Levels=32, Nodes]
            edge_index: Graph connectivity edge indices [2, Num_Edges]
            static_topo: Static surface features [Batch, Num_Static_Feats=2, Nodes]

        Returns:
            Predicted target atmospheric state x(t+1) = x(t) + delta_x [Batch, Out_Vars=7, Levels=32, Nodes]
        """
        # Scrub inputs directly
        x_dynamic = torch.nan_to_num(x_dynamic, nan=0.0, posinf=10.0, neginf=-10.0)
        if static_topo is not None:
            static_topo = torch.nan_to_num(static_topo, nan=0.0, posinf=1.0, neginf=0.0)

        batch_size, num_vars, num_levels, num_nodes = x_dynamic.shape

        # Extract current state x(t_0) from input trajectory [Batch, 7, 32, Nodes]
        # x_dynamic layout: channels 0..6 (t - 6h), channels 7..13 (t_0)
        if num_vars >= 14:
            x_curr = x_dynamic[:, 7:14, :, :]
        else:
            x_curr = x_dynamic[:, :7, :, :]

        # Flatten vertical levels into node feature dimension -> [Batch, In_Vars * Levels, Nodes]
        x_flat = x_dynamic.view(batch_size, num_vars * num_levels, num_nodes)

        # Concatenate static topography features if provided
        if static_topo is not None:
            if static_topo.dim() == 2:
                static_topo = static_topo.unsqueeze(0).expand(batch_size, -1, -1)
            x_flat = torch.cat([x_flat, static_topo], dim=1)

        # Permute to node-first layout -> [Batch, Nodes, Channels]
        x_flat = x_flat.permute(0, 2, 1)

        # Encode inputs into hidden embedding space
        feat = self.encoder(x_flat)  # [Batch, Nodes, Hidden_Dim]

        # Message Passing over Icosahedral Graph Topology
        for gnn in self.gnn_layers:
            feat = gnn(feat, edge_index)

        # Decode node embeddings into 6-hour residual tendency delta_x
        delta_x_flat = self.decoder(feat)  # [Batch, Nodes, Out_Vars * Levels]

        # Reshape delta_x to [Batch, Out_Vars=7, Levels=32, Nodes]
        delta_x = delta_x_flat.permute(0, 2, 1).view(batch_size, self.out_vars, self.num_levels, num_nodes)

        # Residual update: x(t+1) = x(t) + delta_x
        pred_next = x_curr + delta_x

        # Final output guard
        return torch.nan_to_num(pred_next, nan=0.0, posinf=15.0, neginf=-15.0)
