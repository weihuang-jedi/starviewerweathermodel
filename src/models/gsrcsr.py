#!/usr/bin/env python3
"""
models/gsrcsr.py
----------------
Differentiable GSRCSR (GOES ABI Clear-Sky Radiance, 7 Channels) Forward Operator
supporting 3D terrain-following geometric heights (h_3d).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

GSRCSR_PEAK_ALTITUDES_M = np.array([
    9000.0, 6000.0, 4000.0, 10500.0, 100.0, 100.0, 500.0
], dtype=np.float32)

GSRCSR_SIGMA_HEIGHTS_M = np.array([
    2600.0, 2200.0, 1800.0, 2800.0, 1000.0, 1000.0, 1400.0
], dtype=np.float32)

GSRCSR_PEAK_P_PA = np.array([
    32000.0, 48000.0, 62000.0, 25000.0, 100000.0, 100000.0, 95000.0
], dtype=np.float32)

GSRCSR_SIGMA_LN_P = np.array([
    0.45, 0.45, 0.40, 0.35, 0.25, 0.25, 0.30
], dtype=np.float32)

GSRCSR_CHAN_ERRORS = np.array([
    1.10, 1.00, 0.90, 1.20, 0.95, 0.95, 1.10
], dtype=np.float32)


class DifferentiableGSRCSROperator(nn.Module):
    def __init__(self, num_levels: int = 32):
        super().__init__()
        self.num_levels = num_levels
        self.num_channels = len(GSRCSR_PEAK_P_PA)

        self.register_buffer("peak_z", torch.from_numpy(GSRCSR_PEAK_ALTITUDES_M))
        self.register_buffer("sigma_z", torch.from_numpy(GSRCSR_SIGMA_HEIGHTS_M))
        self.register_buffer("peak_p", torch.from_numpy(GSRCSR_PEAK_P_PA))
        self.register_buffer("sigma_ln_p", torch.from_numpy(GSRCSR_SIGMA_LN_P))
        self.register_buffer("obs_errors", torch.from_numpy(GSRCSR_CHAN_ERRORS))

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
