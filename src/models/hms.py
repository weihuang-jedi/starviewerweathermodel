#!/usr/bin/env python3
"""
models/hms.py
-------------
Differentiable HMS (12 Channels) Forward Operator
supporting 3D terrain-following geometric heights (h_3d).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

HMS_PEAK_ALTITUDES_M = np.array([
    100.0, 500.0, 1500.0, 3000.0, 5000.0, 7500.0,
    100.0, 2000.0, 4000.0, 6000.0, 7500.0, 9500.0
], dtype=np.float32)

HMS_SIGMA_HEIGHTS_M = np.array([
    1200.0, 1600.0, 2000.0, 2400.0, 2800.0, 3200.0,
    1200.0, 2000.0, 2400.0, 2800.0, 3200.0, 3600.0
], dtype=np.float32)

HMS_PEAK_P_PA = np.array([
    100000.0, 95000.0, 85000.0, 70000.0, 50000.0, 40000.0,
    100000.0, 80000.0, 60000.0, 45000.0, 35000.0, 27000.0
], dtype=np.float32)

HMS_SIGMA_LN_P = np.array([
    0.30, 0.35, 0.40, 0.40, 0.40, 0.45,
    0.30, 0.40, 0.45, 0.48, 0.50, 0.52
], dtype=np.float32)

HMS_CHAN_ERRORS = np.array([
    2.50, 2.20, 1.20, 0.75, 0.55, 0.45,
    2.50, 1.80, 1.50, 1.30, 1.20, 1.10
], dtype=np.float32)


class DifferentiableHMSOperator(nn.Module):
    def __init__(self, num_levels: int = 32):
        super().__init__()
        self.num_levels = num_levels
        self.num_channels = len(HMS_PEAK_P_PA)

        self.register_buffer("peak_z", torch.from_numpy(HMS_PEAK_ALTITUDES_M))
        self.register_buffer("sigma_z", torch.from_numpy(HMS_SIGMA_HEIGHTS_M))
        self.register_buffer("peak_p", torch.from_numpy(HMS_PEAK_P_PA))
        self.register_buffer("sigma_ln_p", torch.from_numpy(HMS_SIGMA_LN_P))
        self.register_buffer("obs_errors", torch.from_numpy(HMS_CHAN_ERRORS))

    def forward(self, temp_k: torch.Tensor, press_pa: torch.Tensor, h_3d: torch.Tensor = None) -> torch.Tensor:
        if h_3d is not None:
            h_exp = h_3d.unsqueeze(1)
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
