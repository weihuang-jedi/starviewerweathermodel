#!/usr/bin/env python3
"""
models/gsrasr.py
----------------
Differentiable GSRASR (GOES ABI All-Sky Radiance, 10 Channels) Forward Operator
supporting 3D terrain-following geometric heights (h_3d).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

GSRASR_PEAK_ALTITUDES_M = np.array([
    100.0, 9000.0, 6000.0, 4000.0, 100.0, 10500.0, 100.0, 100.0, 500.0, 2000.0
], dtype=np.float32)

GSRASR_SIGMA_HEIGHTS_M = np.array([
    1200.0, 2600.0, 2200.0, 1800.0, 1000.0, 2800.0, 1000.0, 1000.0, 1400.0, 1800.0
], dtype=np.float32)

GSRASR_PEAK_P_PA = np.array([
    100000.0, 32000.0, 48000.0, 62000.0, 100000.0, 25000.0, 100000.0, 100000.0, 95000.0, 80000.0
], dtype=np.float32)

GSRASR_SIGMA_LN_P = np.array([
    0.30, 0.45, 0.45, 0.40, 0.25, 0.35, 0.25, 0.25, 0.30, 0.35
], dtype=np.float32)

GSRASR_CHAN_ERRORS = np.array([
    1.40, 1.15, 1.05, 0.95, 1.10, 1.25, 1.00, 1.00, 1.15, 1.35
], dtype=np.float32)


class DifferentiableGSRASROperator(nn.Module):
    def __init__(self, num_levels: int = 32):
        super().__init__()
        self.num_levels = num_levels
        self.num_channels = len(GSRASR_PEAK_P_PA)

        self.register_buffer("peak_z", torch.from_numpy(GSRASR_PEAK_ALTITUDES_M))
        self.register_buffer("sigma_z", torch.from_numpy(GSRASR_SIGMA_HEIGHTS_M))
        self.register_buffer("peak_p", torch.from_numpy(GSRASR_PEAK_P_PA))
        self.register_buffer("sigma_ln_p", torch.from_numpy(GSRASR_SIGMA_LN_P))
        self.register_buffer("obs_errors", torch.from_numpy(GSRASR_CHAN_ERRORS))

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
