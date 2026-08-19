#!/usr/bin/env python3
"""
models/loss.py
--------------
Composite Physical Balance, Mesh Laplacian, and Standardized Loss Engine for AIDA GNN.
Standardizes target state channels using variable-specific means/stds to ensure balanced
loss gradients across temperature, pressure, winds, humidity, and density.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class M4MeshOperators(nn.Module):
    """Sparse differential operators (Gradient and Divergence) over icosahedral graph nodes."""
    def __init__(self, Gx_sparse: torch.Tensor, Gy_sparse: torch.Tensor, lat_deg: torch.Tensor):
        super().__init__()
        self.register_buffer("Gx_sparse", Gx_sparse)
        self.register_buffer("Gy_sparse", Gy_sparse)
        self.register_buffer("lat_deg", lat_deg)

    def forward(self, scalar_field: torch.Tensor):
        scalar_field = scalar_field.contiguous()
        orig_shape = scalar_field.shape

        if scalar_field.dim() == 3:
            B, L, N = orig_shape
            flat_field = scalar_field.reshape(B * L, N).t()
        else:
            B, N = orig_shape
            L = 1
            flat_field = scalar_field.t()

        df_dx_flat = torch.sparse.mm(self.Gx_sparse, flat_field).t()
        df_dy_flat = torch.sparse.mm(self.Gy_sparse, flat_field).t()

        if L > 1:
            df_dx = df_dx_flat.reshape(B, L, N)
            df_dy = df_dy_flat.reshape(B, L, N)
        else:
            df_dx = df_dx_flat.reshape(B, N)
            df_dy = df_dy_flat.reshape(B, N)

        return df_dx, df_dy


def build_icosahedral_differential_operators(lat_deg: torch.Tensor, lon_deg: torch.Tensor, edge_index: torch.Tensor):
    N = len(lat_deg)
    src_nodes, dst_nodes = edge_index[0].numpy(), edge_index[1].numpy()

    rad = np.pi / 180.0
    R_earth = 6371000.0

    lats_rad = lat_deg.numpy() * rad
    lons_rad = lon_deg.numpy() * rad

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


def generate_or_load_edge_index(num_nodes: int, edge_file: str = None) -> torch.Tensor:
    if edge_file and os.path.exists(edge_file):
        print(f"[GRAPH] Loading precomputed edge topology from '{edge_file}'...", flush=True)
        return torch.load(edge_file)

    print(f"[GRAPH] Generating synthetic icosahedral edges for {num_nodes} nodes...", flush=True)
    edges_src, edges_dst = [], []
    for i in range(num_nodes):
        neighbors = [(i + j) % num_nodes for j in range(1, 7)]
        for n in neighbors:
            edges_src.append(i)
            edges_dst.append(n)

    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    if edge_file:
        os.makedirs(os.path.dirname(edge_file) or ".", exist_ok=True)
        torch.save(edge_index, edge_file)
    return edge_index


class AIDASurrogateLoss(nn.Module):
    """Standardized Reconstruction Loss Engine over Variable Scales."""
    def __init__(
        self,
        w_mse: float = 1.0,
        w_conv: float = 0.05,
        lambda_dyn: float = 0.0001,
        lambda_laplacian_p: float = 0.01,
        lambda_asym_q: float = 0.05,
        num_levels: int = 32,
        **kwargs
    ):
        super().__init__()
        self.w_mse = w_mse
        self.w_conv = w_conv
        self.lambda_dyn = lambda_dyn
        self.lambda_laplacian_p = lambda_laplacian_p
        self.lambda_asym_q = lambda_asym_q
        self.num_levels = num_levels

        # Channel normalization statistics for [ln_t, u, v, w, q, ln_rho, ln_p]
        self.register_buffer("var_means", torch.tensor([5.50, 0.00, 0.00, 0.00, 0.005, -0.20, 10.50], dtype=torch.float32).view(1, 7, 1, 1))
        self.register_buffer("var_stds",  torch.tensor([0.15, 10.0, 10.0, 0.50, 0.005,  0.80,  1.20], dtype=torch.float32).view(1, 7, 1, 1))

        self.register_buffer("mu_ln_t", torch.tensor(5.50, dtype=torch.float32))
        self.register_buffer("std_ln_t", torch.tensor(0.15, dtype=torch.float32))
        self.register_buffer("mu_ln_rho", torch.tensor(-0.20, dtype=torch.float32))
        self.register_buffer("std_ln_rho", torch.tensor(0.80, dtype=torch.float32))
        self.register_buffer("mu_ln_p", torch.tensor(10.50, dtype=torch.float32))
        self.register_buffer("std_ln_p", torch.tensor(1.20, dtype=torch.float32))

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        edge_index: torch.Tensor,
        graph_mesh_ops: nn.Module = None,
        valid_mask: torch.Tensor = None
    ):
        metrics = {}

        pred_clean = torch.clamp(pred, min=-10.0, max=10.0)
        target_clean = torch.clamp(target, min=-10.0, max=10.0)

        pred_norm = (pred_clean - self.var_means) / self.var_stds
        target_norm = (target_clean - self.var_means) / self.var_stds

        # 1. Base Feature-Standardized Loss (MSE)
        if valid_mask is not None:
            mask_7d = valid_mask.unsqueeze(1).expand_as(pred_norm)
            diff_sq = (pred_norm - target_norm) ** 2
            loss_mse = torch.sum(diff_sq * mask_7d) / (torch.sum(mask_7d) * 7.0 + 1e-8)
        else:
            loss_mse = F.mse_loss(pred_norm, target_norm)

        loss_mse = torch.nan_to_num(loss_mse, nan=0.0)
        metrics["loss_mse"] = loss_mse.item()
        total_loss = self.w_mse * loss_mse

        # 2. Asymmetric Moisture Barrier Loss
        q_pred = pred_clean[:, 4, :, :]
        q_neg_penalty = torch.relu(-q_pred + 1e-7) ** 2
        loss_asym_q = torch.mean(q_neg_penalty)
        loss_asym_q = torch.nan_to_num(loss_asym_q, nan=0.0)
        metrics["loss_asym_q"] = loss_asym_q.item()
        total_loss += (self.lambda_asym_q * loss_asym_q)

        # 3. Graph Laplacian Smoothness Penalty on Pressure
        p_pred = pred_clean[:, 6, :, :]
        src, dst = edge_index[0], edge_index[1]
        diff_p = p_pred[:, :, src] - p_pred[:, :, dst]
        loss_laplacian_p = torch.mean(diff_p ** 2)
        loss_laplacian_p = torch.nan_to_num(loss_laplacian_p, nan=0.0)
        metrics["loss_laplacian_p"] = loss_laplacian_p.item()
        total_loss += (self.lambda_laplacian_p * loss_laplacian_p)

        # 4. Geostrophic Dynamics Penalty
        if self.lambda_dyn > 0.0 and graph_mesh_ops is not None and hasattr(graph_mesh_ops, "Gx_sparse"):
            u_pred = pred_clean[:, 1, :, :]
            v_pred = pred_clean[:, 2, :, :]
            dp_dx, dp_dy = graph_mesh_ops(p_pred)

            f_coriolis = 2.0 * 7.2921e-5 * torch.sin(graph_mesh_ops.lat_deg * np.pi / 180.0).view(1, 1, -1).to(pred.device)
            f_coriolis = torch.where(torch.abs(f_coriolis) < 2e-5, torch.sign(f_coriolis) * 2e-5 + 2e-5, f_coriolis)

            ln_rho_unnorm = pred_clean[:, 5, :, :] * self.std_ln_rho + self.mu_ln_rho
            rho_pred = torch.clamp(torch.exp(torch.clamp(ln_rho_unnorm, min=-10.0, max=1.0)), min=1e-4, max=2.0)

            u_geo = torch.clamp(-1.0 / (rho_pred * f_coriolis) * dp_dy, min=-100.0, max=100.0)
            v_geo = torch.clamp(1.0 / (rho_pred * f_coriolis) * dp_dx, min=-100.0, max=100.0)

            loss_dyn = F.mse_loss(u_pred, u_geo) + F.mse_loss(v_pred, v_geo)
            loss_dyn = torch.nan_to_num(loss_dyn, nan=0.0)
            metrics["loss_dynamics_total"] = loss_dyn.item()
            total_loss += (self.lambda_dyn * loss_dyn)
        else:
            metrics["loss_dynamics_total"] = 0.0

        return total_loss, metrics
