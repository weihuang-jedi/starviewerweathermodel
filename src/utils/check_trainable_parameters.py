#!/usr/bin/env python3
import os
import sys
import torch
import torch.nn as nn

# Ensure parent directory is in Python path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.gnn import IcosahedralGNNSurrogate


def count_parameters(model: nn.Module) -> int:
    """Returns the total number of trainable parameters in a PyTorch module."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate_processor_scaling():
    hidden_dims = [128, 256, 384, 512]

    # Common input/output configuration for M6 grid
    in_vars = 18       # 14 dynamic trajectory channels + 4 static topo channels
    out_vars = 7       # Predicted 7 atmospheric state variables
    num_levels = 32    # Vertical level count

    print("=" * 85)
    print(f"{'Hidden Dim (C)':<12} | {'Layers':<8} | {'Total Parameters':<20} | {'Scale':<10}")
    print("=" * 85)

    for hidden_dim in hidden_dims:
        # 1. Standard Depth (4 Layers)
        model_4l = IcosahedralGNNSurrogate(
            in_vars=in_vars,
            out_vars=out_vars,
            hidden_dim=hidden_dim,
            num_levels=num_levels,
            num_layers=4
        )
        params_4l = count_parameters(model_4l)

        # 2. Deeper Architecture (16 Layers)
        model_16l = IcosahedralGNNSurrogate(
            in_vars=in_vars,
            out_vars=out_vars,
            hidden_dim=hidden_dim,
            num_levels=num_levels,
            num_layers=16
        )
        params_16l = count_parameters(model_16l)

        print(f"{hidden_dim:<12} | {4:<8} | {params_4l:>15,} | {f'~{params_4l/1e6:.2f}M':<10}")
        print(f"{hidden_dim:<12} | {16:<8} | {params_16l:>15,} | {f'~{params_16l/1e6:.2f}M':<10}")
        print("-" * 85)

    print("=" * 85)


if __name__ == "__main__":
    evaluate_processor_scaling()
