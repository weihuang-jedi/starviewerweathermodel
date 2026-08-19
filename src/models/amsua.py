#!/usr/bin/env python3
"""
models/amsua.py
---------------
Differentiable AMSU-A Forward Radiance Operator supporting 3D terrain-following heights (h_3d).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# AMSU-A 15 Channels: Peak altitudes (m) and weighting widths (m)
AMSUA_PEAK_ALTITUDES_M = np.array([
      100.0,   500.0,  1500.0,  3000.0,  5000.0,  7500.0, 10500.0, 14000.0,
    18000.0, 21000.0, 26000.0, 30000.0, 35000.0, 40000.0, 45000.0
], dtype=np.float32)

AMSUA_SIGMA_HEIGHTS_M = np.array([
    1500.0, 2000.0, 2200.0, 2500.0, 2800.0, 3000.0, 3200.0, 3500.0,
    3800.0, 4000.0, 4200.0, 4500.0, 5000.0, 5500.0, 6000.0
], dtype=np.float32)

AMSUA_PEAK_P_PA = np.array([
    100000.0, 95000.0, 85000.0, 70000.0, 50000.0, 40000.0, 25000.0, 15000.0,
     10000.0,  5000.0,  2000.0,  1000.0,   500.0,   200.0,   100.0
], dtype=np.float32)

AMSUA_SIGMA_LN_P = np.array([
    0.30, 0.35, 0.40, 0.40, 0.40, 0.45, 0.45, 0.50,
    0.50, 0.50, 0.55, 0.55, 0.60, 0.60, 0.65
], dtype=np.float32)


class DifferentiableAMSUAOperator(nn.Module):
    def __init__(self, num_levels: int = 32):
        super().__init__()
        self.num_levels = num_levels
        self.num_channels = 15

        self.register_buffer("peak_p", torch.from_numpy(AMSUA_PEAK_P_PA))
        self.register_buffer("sigma_ln_p", torch.from_numpy(AMSUA_SIGMA_LN_P))
        self.register_buffer("peak_z", torch.from_numpy(AMSUA_PEAK_ALTITUDES_M))
        self.register_buffer("sigma_z", torch.from_numpy(AMSUA_SIGMA_HEIGHTS_M))

    def forward(self, temp_k: torch.Tensor, press_hpa: torch.Tensor, h_3d: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            temp_k: [Batch, Levels=32, Nodes]
            press_hpa: [Batch, Levels=32, Nodes]
            h_3d: Dynamic 3D terrain-following height profiles [Batch, Levels=32, Nodes] (meters)
        """
        if h_3d is not None:
            # 3D Terrain-following height weighting kernel
            # h_3d: [B, 32, N] -> [B, 1, 32, N]
            h_exp = h_3d.unsqueeze(1)
            peak_z = self.peak_z.view(1, self.num_channels, 1, 1)
            sigma_z = self.sigma_z.view(1, self.num_channels, 1, 1)

            weight_logits = -0.5 * ((h_exp - peak_z) / sigma_z) ** 2
        else:
            # Fallback log-pressure weighting kernel
            press_pa = torch.clamp(press_hpa * 100.0, min=10.0)
            ln_press = torch.log(press_pa).unsqueeze(1)
            peak_p = torch.log(self.peak_p).view(1, self.num_channels, 1, 1)
            sigma_p = self.sigma_ln_p.view(1, self.num_channels, 1, 1)

            weight_logits = -0.5 * ((ln_press - peak_p) / sigma_p) ** 2

        weights = F.softmax(weight_logits, dim=2)  # [B, 15, 32, N]
        tb_sim = torch.sum(temp_k.unsqueeze(1) * weights, dim=2)  # [B, 15, N]
        return tb_sim
