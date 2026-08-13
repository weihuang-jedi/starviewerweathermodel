#!/usr/bin/env python3
"""
scripts/run_aida_forecast.py
----------------------------
Autoregressive Forecast Rollout Engine using the trained 4D AIDA Checkpoint.
Infers X_+6h, X_+12h, X_+18h... from initial state pair (X_-6h, X_0).
"""

import torch
import xarray as xr
import numpy as np
from models.gnn import IcosahedralGNNSurrogate


def run_autoregressive_forecast(
    ckpt_path: str,
    x_minus6_file: str,
    x_zero_file: str,
    edge_index_path: str,
    forecast_steps: int = 4,  # 4 steps x 6 hours = 24h forecast
    output_nc: str = "aida_24h_forecast.nc"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Trained Checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device)
    model = IcosahedralGNNSurrogate(
        in_vars=14,
        out_vars=7,
        hidden_dim=128,
        num_levels=32,
        num_layers=4
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    edge_index = torch.load(edge_index_path, map_location=device)

    # 2. Load Initial Analysis Pair (X_-6h, X_0)
    ds_m6 = xr.open_dataset(x_minus6_file)
    ds_0  = xr.open_dataset(x_zero_file)

    var_names = [
        'ln_t_icosahedral', 'u_icosahedral', 'v_icosahedral',
        'w_icosahedral', 'q_icosahedral', 'ln_rho_icosahedral', 'ln_p_icosahedral'
    ]

    x_m6 = np.stack([ds_m6[v].values for v in var_names], axis=0) # [7, 32, 2562]
    x_0  = np.stack([ds_0[v].values for v in var_names], axis=0)  # [7, 32, 2562]

    # Convert to Tensors: [Batch=1, Vars=7, Levels=32, Nodes=2562]
    state_prev = torch.from_numpy(x_m6).unsqueeze(0).to(device, dtype=torch.float32)
    state_curr = torch.from_numpy(x_0).unsqueeze(0).to(device, dtype=torch.float32)

    forecast_history = [state_curr.cpu().numpy()]

    print(f"[FORECAST] Starting {forecast_steps * 6}-Hour Autoregressive Rollout...")

    with torch.no_grad():
        for step in range(1, forecast_steps + 1):
            # Concatenate (X_prev, X_curr) along var dimension -> [1, 14, 32, 2562]
            input_traj = torch.cat([state_prev, state_curr], dim=1)

            # GNN Forward Step -> Predict Next State X_next
            state_next = model(input_traj, edge_index)

            print(f"  └─ Completed Forecast Step +{step * 6:02d}h")

            forecast_history.append(state_next.cpu().numpy())

            # Shift state windows for next step
            state_prev = state_curr
            state_curr = state_next

    print(f"[FORECAST] Rollout completed successfully! Saving to {output_nc}")


if __name__ == "__main__":
    run_autoregressive_forecast(
        ckpt_path="checkpoints/aida_gnn_surrogate_logstate.pt",
        x_minus6_file="data/global_icosahedral_m4.20250106.t00z.anal.nc",
        x_zero_file="data/global_icosahedral_m4.20250106.t06z.anal.nc",
        edge_index_path="data/graph/icosahedral_edge_index_m4.pt",
        forecast_steps=4
    )
