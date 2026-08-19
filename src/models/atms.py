#!/usr/bin/env python3
"""
models/atms.py
--------------
Differentiable ATMS (Advanced Technology Microwave Sounder, 22 Channels) Forward Operator
supporting 3D terrain-following geometric heights (h_3d).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ATMS 22 Channels: Peak altitudes (m) and weighting widths (m)
ATMS_PEAK_ALTITUDES_M = np.array([
      100.0,   300.0,   800.0,  1800.0,  3500.0,  5500.0,  8500.0, 11500.0,
    15000.0, 18500.0, 22000.0, 26000.0, 31000.0, 37000.0, 42000.0,   100.0,
     2000.0,  4000.0,  6000.0,  7500.0,  9500.0,  2000.0
], dtype=np.float32)

ATMS_SIGMA_HEIGHTS_M = np.array([
    1200.0, 1500.0, 1800.0, 2200.0, 2500.0, 2800.0, 3200.0, 3500.0,
    3800.0, 4200.0, 4500.0, 4800.0, 5200.0, 5800.0, 6500.0, 1200.0,
    2000.0, 2400.0, 2800.0, 3200.0, 3600.0, 2000.0
], dtype=np.float32)

ATMS_PEAK_P_PA = np.array([
    100000.0, 95000.0, 90000.0, 80000.0, 65000.0, 50000.0, 35000.0, 22000.0,
     13000.0,  7000.0,  4000.0,  2000.0,  1000.0,   400.0,   150.0, 100000.0,
     80000.0, 60000.0, 45000.0, 35000.0, 27000.0, 80000.0
], dtype=np.float32)

ATMS_SIGMA_LN_P = np.array([
    0.30, 0.35, 0.38, 0.40, 0.42, 0.45, 0.45, 0.50,
    0.50, 0.52, 0.55, 0.58, 0.60, 0.65, 0.70, 0.30,
    0.40, 0.45, 0.48, 0.50, 0.52, 0.40
], dtype=np.float32)

ATMS_CHAN_ERRORS = np.array([
    2.50, 2.20, 1.20, 0.75, 0.55, 0.45, 0.45, 0.45,
    0.45, 0.55, 0.70, 0.90, 1.30, 2.00, 3.80, 2.50,
    1.80, 1.50, 1.30, 1.20, 1.10, 2.00
], dtype=np.float32)


class DifferentiableATMSOperator(nn.Module):
    def __init__(self, num_levels: int = 32):
        super().__init__()
        self.num_levels = num_levels
        self.num_channels = len(ATMS_PEAK_P_PA)

        self.register_buffer("peak_z", torch.from_numpy(ATMS_PEAK_ALTITUDES_M))
        self.register_buffer("sigma_z", torch.from_numpy(ATMS_SIGMA_HEIGHTS_M))
        self.register_buffer("peak_p", torch.from_numpy(ATMS_PEAK_P_PA))
        self.register_buffer("sigma_ln_p", torch.from_numpy(ATMS_SIGMA_LN_P))
        self.register_buffer("obs_errors", torch.from_numpy(ATMS_CHAN_ERRORS))

    def forward(self, temp_k: torch.Tensor, press_pa: torch.Tensor, h_3d: torch.Tensor = None) -> torch.Tensor:
        if h_3d is not None:
            h_exp = h_3d.unsqueeze(1)  # [B, 1, 32, N]
            peak_z = self.peak_z.view(1, self.num_channels, 1, 1)
            sigma_z = self.sigma_z.view(1, self.num_channels, 1, 1)
            weight_logits = -0.5 * ((h_exp - peak_z) / sigma_z) ** 2
        else:
            press_pa = torch.clamp(press_pa, min=10.0)
            ln_press = torch.log(press_pa).unsqueeze(1)
            peak_p = torch.log(self.peak_p).view(1, self.num_channels, 1, 1)
            sigma_p = self.sigma_ln_p.view(1, self.num_channels, 1, 1)
            weight_logits = -0.5 * ((ln_press - peak_p) / sigma_p) ** 2

        weights = F.softmax(weight_logits, dim=2)
        tb_sim = torch.sum(temp_k.unsqueeze(1) * weights, dim=2)
        return tb_sim
