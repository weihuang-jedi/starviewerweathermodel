#!/usr/bin/env python3
"""
models/gnn.py
-------------
Icosahedral GNN Surrogate Model Backbone for Atmospheric Data Assimilation and Forecasting.
Supports terrain-following 3D vertical state grids, static topography feature conditioning
(surface elevation + land-sea mask), and flexible multi-step 4D trajectory representations.
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
    Global Icosahedral GNN Surrogate Network.
    Ingests dynamic multi-level state profiles [Batch, In_Vars, Levels, Nodes] and static
    topography features [Batch, Num_Static_Feats, Nodes] to predict next-state atmospheric fields.
    """
    def __init__(
        self,
        in_vars: int = 14,          # Input dynamic variables (e.g., 7 vars x 2 timesteps for 4D trajectory)
        out_vars: int = 7,          # Output dynamic variables predicted (ln_t, u, v, w, q, ln_rho, ln_p)
        num_static_feats: int = 2,  # Static terrain features (Surface Elevation + Land-Sea Mask)
        hidden_dim: int = 128,
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

        # Total input dimension per node: (In_Vars * Levels) + Static_Topography_Features
        total_in_dim = (in_vars * num_levels) + num_static_feats

        # Node Feature Encoder
        self.encoder = nn.Linear(total_in_dim, hidden_dim)

        # Sequence of GNN Interaction Layers
        self.gnn_layers = nn.ModuleList([
            GraphConvBlock(hidden_dim=hidden_dim) for _ in range(num_layers)
        ])

        # State Prediction Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_vars * num_levels)
        )

    def forward(
        self,
        x_dynamic: torch.Tensor,
        edge_index: torch.Tensor,
        static_topo: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            x_dynamic: Dynamic atmospheric states [Batch, In_Vars, Levels=32, Nodes]
            edge_index: Master icosahedral edge topology indices [2, Num_Edges]
            static_topo: Static surface features [Batch, Num_Static_Feats=2, Nodes] (Elevation + Mask)

        Returns:
            Predicted target atmospheric state fields [Batch, Out_Vars=7, Levels=32, Nodes]
        """
        batch_size, num_vars, num_levels, num_nodes = x_dynamic.shape

        # Flatten vertical levels into node feature dimension -> [Batch, In_Vars * Levels, Nodes]
        x_flat = x_dynamic.view(batch_size, num_vars * num_levels, num_nodes)

        # Concatenate static topography features if provided
        if static_topo is not None:
            if static_topo.dim() == 2:
                # Expand [Static_Feats, Nodes] -> [Batch, Static_Feats, Nodes]
                static_topo = static_topo.unsqueeze(0).expand(batch_size, -1, -1)
            
            x_flat = torch.cat([x_flat, static_topo], dim=1)  # [Batch, (In_Vars * Levels) + Num_Static, Nodes]

        # Permute to node-first layout for PyTorch linear layers -> [Batch, Nodes, Channels]
        x_flat = x_flat.permute(0, 2, 1)

        # Encode inputs into hidden embedding space
        feat = self.encoder(x_flat)  # [Batch, Nodes, Hidden_Dim]

        # Message Passing over Icosahedral Graph Topology
        for gnn in self.gnn_layers:
            feat = gnn(feat, edge_index)

        # Decode node embeddings into predicted vertical state profiles
        out_flat = self.decoder(feat)  # [Batch, Nodes, Out_Vars * Levels]

        # Permute back to standard AIDA tensor shape -> [Batch, Out_Vars, Levels, Nodes]
        out_flat = out_flat.permute(0, 2, 1).view(batch_size, self.out_vars, self.num_levels, num_nodes)

        return out_flat
