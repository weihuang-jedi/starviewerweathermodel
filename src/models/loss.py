#!/usr/bin/env python3
"""
models/loss.py
--------------
AIDA Multi-Objective Loss Function Engine for Icosahedral Atmospheric Grids.
Enforces physical consistency using:
  1. Primary State MSE / Normalized Dynamic Field Loss
  2. Specific Kinetic Energy Loss (L_ke)
  3. Wind Vector Cosine Direction Alignment Loss (L_dir)
  4. 3D Horizontal Navier-Stokes Momentum Residual Loss (L_momentum) via Sparse Graph Operators
  5. Non-Hydrostatic Vertical Momentum Equation Residual Loss (L_vert_dynamics)
  6. Log-State Mass Continuity Equation Residual Loss (L_continuity)
  7. M4 Sparse Mesh Spatial Gradient / Laplacian Regularization
  8. Conventional & Radiance Forward Operator Observation Penalties
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
    Integrates kinetic energy conservation, directional wind alignment,
    3D horizontal momentum, non-hydrostatic vertical momentum, and mass continuity.
    """
    def __init__(
        self,
        num_levels: int = 32,
        w_mse: float = 1.0,
        w_conv: float = 0.05,
        w_wind_ke: float = 0.15,
        w_wind_dir: float = 0.10,
        w_laplacian_p: float = 0.01,
        w_dynamics: float = 0.001,
        w_vert_dynamics: float = 0.0001,
        w_continuity: float = 0.0001,
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
        self.w_vert_dynamics = w_vert_dynamics
        self.w_continuity = w_continuity
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

    def compute_momentum_residual_loss(
        self,
        pred: torch.Tensor,
        graph_mesh_ops: M4MeshOperators,
        edge_index_vert: torch.Tensor = None,
        h_3d: torch.Tensor = None
    ) -> torch.Tensor:
        """Computes 3D horizontal Navier-Stokes momentum residuals."""
        R_d = 287.05
        B, C, L, N = pred.shape

        u = pred[:, self.u_idx, :, :]
        v = pred[:, self.v_idx, :, :]
        w = pred[:, 3, :, :]
        ln_T = pred[:, 0, :, :]
        ln_P = pred[:, self.p_idx, :, :]

        t_abs = torch.exp(ln_T * self.std_ln_t + self.mu_ln_t)

        Omega = 7.292115e-5
        f_coriolis = (2.0 * Omega * torch.sin(graph_mesh_ops.lat_deg * (np.pi / 180.0))).to(pred.device)
        f_coriolis = f_coriolis.view(1, 1, N).expand(B, L, N)

        u_flat = u.permute(2, 0, 1).reshape(N, B * L)
        v_flat = v.permute(2, 0, 1).reshape(N, B * L)
        p_flat = ln_P.permute(2, 0, 1).reshape(N, B * L)

        du_dx = torch.sparse.mm(graph_mesh_ops.Gx_sparse, u_flat).reshape(N, B, L).permute(1, 2, 0)
        du_dy = torch.sparse.mm(graph_mesh_ops.Gy_sparse, u_flat).reshape(N, B, L).permute(1, 2, 0)

        dv_dx = torch.sparse.mm(graph_mesh_ops.Gx_sparse, v_flat).reshape(N, B, L).permute(1, 2, 0)
        dv_dy = torch.sparse.mm(graph_mesh_ops.Gy_sparse, v_flat).reshape(N, B, L).permute(1, 2, 0)

        dp_dx = torch.sparse.mm(graph_mesh_ops.Gx_sparse, p_flat).reshape(N, B, L).permute(1, 2, 0)
        dp_dy = torch.sparse.mm(graph_mesh_ops.Gy_sparse, p_flat).reshape(N, B, L).permute(1, 2, 0)

        if h_3d is not None and h_3d.dim() == 4:
            dz = torch.abs(torch.diff(h_3d.squeeze(1), dim=1))
            dz = torch.clamp(dz, min=10.0)
            dz = torch.cat([dz, dz[:, -1:, :]], dim=1)

            du_dz = torch.diff(u, dim=1)
            du_dz = torch.cat([du_dz, du_dz[:, -1:, :]], dim=1) / dz

            dv_dz = torch.diff(v, dim=1)
            dv_dz = torch.cat([dv_dz, dv_dz[:, -1:, :]], dim=1) / dz
        else:
            du_dz = torch.diff(u, dim=1)
            du_dz = torch.cat([du_dz, du_dz[:, -1:, :]], dim=1)

            dv_dz = torch.diff(v, dim=1)
            dv_dz = torch.cat([dv_dz, dv_dz[:, -1:, :]], dim=1)

        advection_u = (u * du_dx) + (v * du_dy) + (w * du_dz)
        advection_v = (u * dv_dx) + (v * dv_dy) + (w * dv_dz)

        pgf_u = R_d * t_abs * dp_dx
        pgf_v = R_d * t_abs * dp_dy

        R_u = advection_u - (f_coriolis * v) + pgf_u
        R_v = advection_v + (f_coriolis * u) + pgf_v

        return torch.mean(R_u ** 2) + torch.mean(R_v ** 2)

    def compute_vertical_momentum_residual_loss(
        self,
        pred: torch.Tensor,
        graph_mesh_ops: M4MeshOperators,
        h_3d: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Computes robust non-dimensionalized vertical momentum residuals using 
        ideal gas thermodynamic log-pressure PGF scaling: (1/rho)*dP/dz = R_d * T * d(ln P)/dz.
        """
        g = 9.80665
        R_d = 287.05
        R_earth = 6371000.0
        Omega = 7.292115e-5

        B, C, L, N = pred.shape

        u = pred[:, self.u_idx, :, :]
        v = pred[:, self.v_idx, :, :]
        w = pred[:, 3, :, :]
        ln_T = pred[:, 0, :, :]
        ln_P = pred[:, self.p_idx, :, :]

        # Absolute temperature (Kelvin)
        t_abs = torch.exp(ln_T * self.std_ln_t + self.mu_ln_t)

        # 1. Level thickness dz (m)
        if h_3d is not None and h_3d.dim() == 4:
            dz = torch.abs(torch.diff(h_3d.squeeze(1), dim=1))
            dz = torch.clamp(dz, min=10.0)
            dz = torch.cat([dz, dz[:, -1:, :]], dim=1)
        else:
            dz = 250.0

        # 2. Thermodynamic Vertical Pressure Gradient Acceleration
        # d(ln P)/dz via physical scale factor
        dln_P_dz = torch.diff(ln_P * self.std_ln_p + self.mu_ln_p, dim=1)
        dln_P_dz = torch.cat([dln_P_dz, dln_P_dz[:, -1:, :]], dim=1) / dz

        # PGF Acceleration = - R_d * T * d(ln P)/dz
        pgf_w = -R_d * t_abs * dln_P_dz

        # Hydrostatic acceleration imbalance relative to g
        hydrostatic_imbalance = (pgf_w - g) / g

        # 3. Spatial Advection of vertical wind w
        w_flat = w.permute(2, 0, 1).reshape(N, B * L)
        dw_dx = torch.sparse.mm(graph_mesh_ops.Gx_sparse, w_flat).reshape(N, B, L).permute(1, 2, 0)
        dw_dy = torch.sparse.mm(graph_mesh_ops.Gy_sparse, w_flat).reshape(N, B, L).permute(1, 2, 0)

        dw_dz = torch.diff(w, dim=1)
        dw_dz = torch.cat([dw_dz, dw_dz[:, -1:, :]], dim=1) / dz

        advection_w = ((u * dw_dx) + (v * dw_dy) + (w * dw_dz)) / g

        # 4. Spherical Metric & Coriolis accelerations
        metric_centrifugal = (-(u**2 + v**2) / R_earth) / g

        cos_lat = torch.cos(graph_mesh_ops.lat_deg * (np.pi / 180.0)).to(pred.device)
        cos_lat = cos_lat.view(1, 1, N).expand(B, L, N)
        coriolis_w = (2.0 * Omega * u * cos_lat) / g

        # 5. Combined Non-Dimensional Vertical Residual
        residual_w = advection_w - hydrostatic_imbalance + metric_centrifugal + coriolis_w

        return torch.mean(residual_w ** 2)

    def compute_continuity_residual_loss(
        self,
        pred: torch.Tensor,
        graph_mesh_ops: M4MeshOperators,
        h_3d: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Computes non-dimensionalized log-state mass continuity residuals:
          R_cont = [ V·∇(ln rho) + ∇·V ] / (1e-4 s^-1)
        """
        B, C, L, N = pred.shape

        u = pred[:, self.u_idx, :, :]
        v = pred[:, self.v_idx, :, :]
        w = pred[:, 3, :, :]
        ln_rho = pred[:, 5, :, :]

        if h_3d is not None and h_3d.dim() == 4:
            dz = torch.abs(torch.diff(h_3d.squeeze(1), dim=1))
            dz = torch.clamp(dz, min=10.0)
            dz = torch.cat([dz, dz[:, -1:, :]], dim=1)
        else:
            dz = 250.0

        u_flat = u.permute(2, 0, 1).reshape(N, B * L)
        v_flat = v.permute(2, 0, 1).reshape(N, B * L)
        rho_flat = ln_rho.permute(2, 0, 1).reshape(N, B * L)

        drho_dx = torch.sparse.mm(graph_mesh_ops.Gx_sparse, rho_flat).reshape(N, B, L).permute(1, 2, 0)
        drho_dy = torch.sparse.mm(graph_mesh_ops.Gy_sparse, rho_flat).reshape(N, B, L).permute(1, 2, 0)

        du_dx = torch.sparse.mm(graph_mesh_ops.Gx_sparse, u_flat).reshape(N, B, L).permute(1, 2, 0)
        dv_dy = torch.sparse.mm(graph_mesh_ops.Gy_sparse, v_flat).reshape(N, B, L).permute(1, 2, 0)

        drho_dz = torch.diff(ln_rho, dim=1)
        drho_dz = torch.cat([drho_dz, drho_dz[:, -1:, :]], dim=1) / dz

        dw_dz = torch.diff(w, dim=1)
        dw_dz = torch.cat([dw_dz, dw_dz[:, -1:, :]], dim=1) / dz

        advection_rho = (u * drho_dx) + (v * drho_dy) + (w * drho_dz)
        div_V = du_dx + dv_dy + dw_dz

        # Normalize by typical synoptic divergence scale (1e-4 s^-1)
        residual_cont = (advection_rho + div_V) / 1e-4

        return torch.mean(residual_cont ** 2)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        edge_index: torch.Tensor,
        edge_index_vert: torch.Tensor = None,
        graph_mesh_ops: M4MeshOperators = None,
        valid_mask: torch.Tensor = None,
        h_3d: torch.Tensor = None,
        static_topo: torch.Tensor = None
    ) -> tuple[torch.Tensor, dict]:
        """Computes total loss and itemized metrics breakdown."""
        metrics = {}

        # 1. Base State Mean Squared Error (MSE)
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

        # 2. Wind Kinetic Energy & Direction
        u_pred, v_pred = pred[:, self.u_idx, :, :], pred[:, self.v_idx, :, :]
        u_true, v_true = target[:, self.u_idx, :, :], target[:, self.v_idx, :, :]

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

        # 3. Spatial Mesh Laplacian Regularization
        if graph_mesh_ops is not None and self.w_laplacian_p > 0.0:
            with torch.amp.autocast("cuda", enabled=False):
                p_pred = pred[:, self.p_idx, :, :].float()
                B, L, N = p_pred.shape

                p_flat = p_pred.reshape(B * L, N).t()

                grad_x = torch.sparse.mm(graph_mesh_ops.Gx_sparse, p_flat)
                grad_y = torch.sparse.mm(graph_mesh_ops.Gy_sparse, p_flat)

                laplacian_p = torch.sparse.mm(graph_mesh_ops.Gx_sparse, grad_x) + torch.sparse.mm(graph_mesh_ops.Gy_sparse, grad_y)
                loss_laplacian_p = torch.mean(laplacian_p ** 2)

                total_loss += (self.w_laplacian_p * loss_laplacian_p)
                metrics["loss_laplacian_p"] = loss_laplacian_p.item()
        else:
            metrics["loss_laplacian_p"] = 0.0

        # 4. Log-State Mass Continuity Residual Loss
        if graph_mesh_ops is not None and self.w_continuity > 0.0:
            loss_continuity = self.compute_continuity_residual_loss(
                pred=pred,
                graph_mesh_ops=graph_mesh_ops,
                h_3d=h_3d
            )
            total_loss += (self.w_continuity * loss_continuity)
            metrics["loss_continuity"] = loss_continuity.item()
        else:
            metrics["loss_continuity"] = 0.0

        # 5. Non-Hydrostatic Vertical Momentum Residual Loss
        if graph_mesh_ops is not None and self.w_vert_dynamics > 0.0:
            loss_vert_dynamics = self.compute_vertical_momentum_residual_loss(
                pred=pred,
                graph_mesh_ops=graph_mesh_ops,
                h_3d=h_3d
            )
            total_loss += (self.w_vert_dynamics * loss_vert_dynamics)
            metrics["loss_vert_dynamics"] = loss_vert_dynamics.item()
        else:
            metrics["loss_vert_dynamics"] = 0.0

        # 6. 3D Horizontal Navier-Stokes Momentum Residual Loss
        if graph_mesh_ops is not None and self.w_dynamics > 0.0:
            loss_momentum = self.compute_momentum_residual_loss(
                pred=pred,
                graph_mesh_ops=graph_mesh_ops,
                edge_index_vert=edge_index_vert,
                h_3d=h_3d
            )
            total_loss += (self.w_dynamics * loss_momentum)
            metrics["loss_momentum"] = loss_momentum.item()
        else:
            metrics["loss_momentum"] = 0.0

        metrics["loss_total"] = total_loss.item()
        return total_loss, metrics
