#!/usr/bin/env python3
"""
models/cris.py
--------------
Differentiable CrIS (Cross-track Infrared Sounder, 30 Channels) Forward Operator
supporting 3D terrain-following geometric heights (h_3d).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DifferentiableCrISOperator(nn.Module):
    def __init__(self, num_levels: int = 32):
        super().__init__()
        self.num_levels = num_levels
        self.num_channels = 30

        peak_z_arr = np.linspace(100.0, 42000.0, self.num_channels, dtype=np.float32)
        sigma_z_arr = np.linspace(1100.0, 4500.0, self.num_channels, dtype=np.float32)

        peak_p_arr = np.logspace(np.log10(100000.0), np.log10(150.0), self.num_channels, dtype=np.float32)
        sigma_p_arr = np.full(self.num_channels, 0.35, dtype=np.float32)
        errors_arr = np.linspace(0.7, 1.7, self.num_channels, dtype=np.float32)

        self.register_buffer("peak_z", torch.from_numpy(peak_z_arr))
        self.register_buffer("sigma_z", torch.from_numpy(sigma_z_arr))
        self.register_buffer("peak_p", torch.from_numpy(peak_p_arr))
        self.register_buffer("sigma_ln_p", torch.from_numpy(sigma_p_arr))
        self.register_buffer("obs_errors", torch.from_numpy(errors_arr))

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
