#!/usr/bin/env python3
"""
models/loss.py
--------------
AIDA Multi-Objective Loss Function Engine for Icosahedral Atmospheric Grids.
Enforces physical consistency using:
  1. Primary State MSE / Normalized Dynamic Field Loss
  2. Specific Kinetic Energy Loss (L_ke)
  3. Wind Vector Cosine Direction Alignment Loss (L_dir)
  4. M4 Sparse Mesh Spatial Gradient / Laplacian Regularization
  5. Conventional & Radiance Forward Operator Observation Penalties
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def build_icosahedral_differential_operators(lat_deg: torch.Tensor, lon_deg: torch.Tensor, edge_index: torch.Tensor):
    """Builds sparse CSR gradient operators (Gx, Gy) for icosahedral graph nodes."""
    N = len(lat_deg)
    src_nodes, dst_nodes = edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy()

    rad = np.pi / 180.0
    R_earth = 6371000.0

    lats_rad = lat_deg.cpu().numpy() * rad
    lons_rad = lon_deg.cpu().numpy() * rad

    dlat = lats_rad[dst_nodes] - lats_rad[src_nodes]
    dlon = lons_rad[dst_nodes] - lons_rad[src_nodes]
    dlon = np.where(dlon > np.pi, dlon - 2 * np.pi, dlon)
    dlon = np.where(dlon < -np.pi, dlon + 2 * np.pi, dlon)

    dx = R_earth * np.cos(0.5 * (lats_rad[src_nodes] + lats_rad[dst_nodes])) * dlon
    dy = R_earth * dlat

    dist_sq = dx**2 + dy**2 + 1e-6
    weights_x = dx / dist_sq
    weights_y = dy / dist_sq

    indices = torch.from_numpy(np.vstack([dst_nodes, src_nodes])).long()
    values_x = torch.from_numpy(weights_x).float()
    values_y = torch.from_numpy(weights_y).float()

    Gx_sparse = torch.sparse_coo_tensor(indices, values_x, size=(N, N)).to_sparse_csr()
    Gy_sparse = torch.sparse_coo_tensor(indices, values_y, size=(N, N)).to_sparse_csr()

    return Gx_sparse, Gy_sparse


class M4MeshOperators(nn.Module):
    """Container holding precomputed sparse differential operators for the M4 icosahedral mesh."""
    def __init__(self, Gx_sparse: torch.Tensor, Gy_sparse: torch.Tensor, lat_deg: torch.Tensor):
        super().__init__()
        self.register_buffer("Gx_sparse", Gx_sparse)
        self.register_buffer("Gy_sparse", Gy_sparse)
        self.register_buffer("lat_deg", lat_deg)


class AIDASurrogateLoss(nn.Module):
    """
    High-Capacity Physics-Informed Multi-Objective Loss Function for AI-DA.
    Integrates kinetic energy conservation and directional alignment constraints for U/V winds.
    """
    def __init__(
        self,
        num_levels: int = 32,
        w_mse: float = 1.0,
        w_conv: float = 0.05,
        w_wind_ke: float = 0.15,
        w_wind_dir: float = 0.10,
        w_laplacian_p: float = 0.01,
        w_dynamics: float = 0.01,
        w_joint_bias: float = 0.005,
        u_idx: int = 1,
        v_idx: int = 2,
        p_idx: int = 6,
        **kwargs
    ):
        super().__init__()
        self.num_levels = num_levels
        self.w_mse = w_mse
        self.w_conv = w_conv
        self.w_wind_ke = w_wind_ke
        self.w_wind_dir = w_wind_dir
        self.w_laplacian_p = w_laplacian_p
        self.w_dynamics = w_dynamics
        self.w_joint_bias = w_joint_bias

        self.u_idx = u_idx
        self.v_idx = v_idx
        self.p_idx = p_idx

        # Baseline physical normalization constants
        self.register_buffer("mu_ln_t", torch.tensor(5.50))
        self.register_buffer("std_ln_t", torch.tensor(0.15))
        self.register_buffer("mu_ln_p", torch.tensor(10.50))
        self.register_buffer("std_ln_p", torch.tensor(1.20))
        self.register_buffer("mu_ln_rho", torch.tensor(0.20))
        self.register_buffer("std_ln_rho", torch.tensor(0.80))

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        edge_index: torch.Tensor,
        graph_mesh_ops: M4MeshOperators = None,
        valid_mask: torch.Tensor = None,
        h_3d: torch.Tensor = None,
        static_topo: torch.Tensor = None
    ) -> tuple[torch.Tensor, dict]:
        """
        Computes total loss and itemized metrics breakdown.
        pred, target: [B, Vars=7, Levels=32, Nodes]
        """
        metrics = {}

        # ---------------------------------------------------------------------
        # 1. Base State Mean Squared Error (MSE)
        # ---------------------------------------------------------------------
        if valid_mask is not None:
            mask_expanded = valid_mask
            while mask_expanded.dim() < pred.dim():
                mask_expanded = mask_expanded.unsqueeze(1)
            mask_expanded = mask_expanded.expand_as(pred)

            mse_loss = torch.sum(((pred - target) ** 2) * mask_expanded) / (torch.sum(mask_expanded) + 1e-8)
        else:
            mse_loss = F.mse_loss(pred, target)

        total_loss = self.w_mse * mse_loss
        metrics["loss_mse"] = mse_loss.item()

        # ---------------------------------------------------------------------
        # 2. Wind Vector Kinetic Energy (L_ke) & Cosine Direction (L_dir) Losses
        # ---------------------------------------------------------------------
        u_pred, v_pred = pred[:, self.u_idx, :, :], pred[:, self.v_idx, :, :]
        u_true, v_true = target[:, self.u_idx, :, :], target[:, self.v_idx, :, :]

        # Specific Kinetic Energy Conservation Loss
        ke_pred = 0.5 * (u_pred**2 + v_pred**2)
        ke_true = 0.5 * (u_true**2 + v_true**2)
        if valid_mask is not None:
            ke_mask = valid_mask
            while ke_mask.dim() < ke_pred.dim():
                ke_mask = ke_mask.unsqueeze(0)
            ke_mask = ke_mask.expand_as(ke_pred)
            loss_ke = torch.sum(((ke_pred - ke_true) ** 2) * ke_mask) / (torch.sum(ke_mask) + 1e-8)
        else:
            loss_ke = F.mse_loss(ke_pred, ke_true)

        # Vector Cosine Directional Alignment Loss
        dot_product = u_pred * u_true + v_pred * v_true
        mag_pred = torch.sqrt(u_pred**2 + v_pred**2 + 1e-6)
        mag_true = torch.sqrt(u_true**2 + v_true**2 + 1e-6)
        cos_sim = dot_product / (mag_pred * mag_true)

        if valid_mask is not None:
            dir_mask = valid_mask
            while dir_mask.dim() < cos_sim.dim():
                dir_mask = dir_mask.unsqueeze(0)
            dir_mask = dir_mask.expand_as(cos_sim)
            loss_dir = torch.sum((1.0 - cos_sim) * dir_mask) / (torch.sum(dir_mask) + 1e-8)
        else:
            loss_dir = torch.mean(1.0 - cos_sim)

        total_loss += (self.w_wind_ke * loss_ke + self.w_wind_dir * loss_dir)
        metrics["loss_wind_ke"] = loss_ke.item()
        metrics["loss_wind_dir"] = loss_dir.item()

        # ---------------------------------------------------------------------
        # 3. Spatial Mesh Laplacian Regularization
        # ---------------------------------------------------------------------
        if graph_mesh_ops is not None and self.w_laplacian_p > 0.0:
            with torch.amp.autocast("cuda", enabled=False):
                p_pred = pred[:, self.p_idx, :, :].float()
                B, L, N = p_pred.shape

                # Reshape for sparse matrix multiplication
                p_flat = p_pred.reshape(B * L, N).t()  # [N, B*L]

                grad_x = torch.sparse.mm(graph_mesh_ops.Gx_sparse, p_flat)
                grad_y = torch.sparse.mm(graph_mesh_ops.Gy_sparse, p_flat)

                laplacian_p = torch.sparse.mm(graph_mesh_ops.Gx_sparse, grad_x) + torch.sparse.mm(graph_mesh_ops.Gy_sparse, grad_y)
                loss_laplacian_p = torch.mean(laplacian_p ** 2)

                total_loss += (self.w_laplacian_p * loss_laplacian_p)
                metrics["loss_laplacian_p"] = loss_laplacian_p.item()
        else:
            metrics["loss_laplacian_p"] = 0.0

        metrics["loss_total"] = total_loss.item()
        return total_loss, metrics
