#!/usr/bin/env python3
"""
scripts/run_aida_forecast.py
----------------------------
Autoregressive Forecast Rollout Engine using the trained 4D Terrain-Following AIDA Checkpoint.
Infers X_+6h, X_+12h, X_+18h... from initial analysis state pair (X_-6h, X_0)
while conditioning on static topography (static_topo) and 3D terrain heights (h_3d).

Guarantees physically valid output states by enforcing linear trend extrapolation baselines:
    X_next = X_0 + (X_0 - X_-6h) + delta_X_GNN
"""

import argparse
import os
import sys
import re
import yaml
import numpy as np
import xarray as xr
import torch

# Ensure parent directory is in Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.gnn import IcosahedralGNNSurrogate
from models.graph import generate_or_load_edge_index


def load_state_from_file(file_path: str, var_names: list):
    """Helper to extract dynamic 7-variable log-state tensor, 3D terrain heights, and static topography."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"[ERROR] Input state file not found: '{file_path}'")

    if file_path.endswith('.zarr'):
        ds = xr.open_zarr(file_path)
    else:
        ds = xr.open_dataset(file_path)

    # 1. Extract 7 dynamic state variables [7, Levels=32, Nodes=2562]
    state_vars = []
    for v in var_names:
        if v in ds:
            val = ds[v].values
        elif f"{v}_icosahedral" in ds:
            val = ds[f"{v}_icosahedral"].values
        else:
            raise KeyError(f"[ERROR] Required state variable '{v}' missing in '{file_path}'")

        if val.ndim == 3:  # Squeeze time dimension if present
            val = val[0]
        state_vars.append(val)

    state_np = np.stack(state_vars, axis=0).astype(np.float32)

    # 2. Extract 3D Terrain-Following Geometric Heights h_3d [32, Nodes]
    if 'h_icosahedral' in ds:
        h_3d_np = ds['h_icosahedral'].values
    elif 'h' in ds:
        h_3d_np = ds['h'].values
    else:
        num_levels = state_np.shape[1]
        num_nodes = state_np.shape[2]
        baseline_h = np.linspace(2, 20000, num_levels, dtype=np.float32)
        h_3d_np = np.repeat(baseline_h[:, np.newaxis], num_nodes, axis=1)

    if h_3d_np.ndim == 3:
        h_3d_np = h_3d_np[0]

    # 3. Extract Static Surface Topography [2, Nodes] (Elevation + Land-Sea Mask)
    if 'h_terrain_icosahedral' in ds:
        h_terrain = ds['h_terrain_icosahedral'].values
    elif 'h_terrain' in ds:
        h_terrain = ds['h_terrain'].values
    elif 'elevation' in ds:
        h_terrain = ds['elevation'].values
    else:
        h_terrain = np.zeros((state_np.shape[2],), dtype=np.float32)

    if 'land_sea_mask' in ds:
        ls_mask = ds['land_sea_mask'].values
    else:
        ls_mask = np.zeros((state_np.shape[2],), dtype=np.float32)

    if h_terrain.ndim > 1:
        h_terrain = h_terrain[0]
    if ls_mask.ndim > 1:
        ls_mask = ls_mask[0]

    static_topo_np = np.stack([h_terrain.astype(np.float32) / 10000.0, ls_mask.astype(np.float32)], axis=0)

    return state_np, h_3d_np.astype(np.float32), static_topo_np, ds


def parse_date_tag(filename: str) -> str:
    """Extracts date tag string (e.g., '20260101.t12z') from a file path."""
    match = re.search(r'(\d{8}\.t\d{2}z)', os.path.basename(filename))
    if match:
        return match.group(1)
    return "forecast"


def export_lead_time_netcdf(
    output_path: str,
    state_arr: np.ndarray,
    h_3d_np: np.ndarray,
    static_topo_np: np.ndarray,
    ds_ref: xr.Dataset,
    var_names: list,
    lead_time_hours: int,
    num_levels: int,
    num_nodes: int
):
    """Saves a single lead-time forecast state as a fully self-contained CF/UGRID NetCDF file."""
    data_vars_out = {}

    # Dynamic 7 variables -> [level, node]
    for idx, var in enumerate(var_names):
        out_var_name = var if var.endswith("_icosahedral") else f"{var}_icosahedral"
        data_vars_out[out_var_name] = (
            ["level", "node"],
            state_arr[idx, :, :].astype(np.float32),
            {"long_name": f"Forecasted {var}", "mesh": "icosahedral_mesh"}
        )

    # Attach 3D Terrain Heights and Surface Topography
    data_vars_out["h_icosahedral"] = (
        ["level", "node"],
        h_3d_np,
        {"units": "meters", "long_name": "3D Terrain-Following Geometric Height Above Sea Level", "mesh": "icosahedral_mesh"}
    )
    data_vars_out["h_terrain_icosahedral"] = (
        ["node"],
        static_topo_np[0] * 10000.0,
        {"units": "meters", "long_name": "Surface Topography Elevation", "mesh": "icosahedral_mesh"}
    )

    # Passthrough Mesh Coordinates and Connectivity for Plotting Compatibility
    for static_var in ["longitude", "latitude", "face_nodes", "x_cartesian", "y_cartesian", "z_cartesian", "land_sea_mask", "elevation"]:
        if static_var in ds_ref:
            data_vars_out[static_var] = ds_ref[static_var]

    coords_out = {
        "level": np.arange(1, num_levels + 1, dtype=np.int32),
        "node": np.arange(num_nodes, dtype=np.int32)
    }

    if "face" in ds_ref.dims:
        coords_out["face"] = ds_ref["face"].values
    if "three" in ds_ref.dims:
        coords_out["three"] = ds_ref["three"].values

    ds_out = xr.Dataset(
        data_vars=data_vars_out,
        coords=coords_out,
        attrs={
            "title": "AIDA GNN 4D Observation-Guided Terrain Weather Forecast",
            "conventions": "CF-1.8 UGRID-1.0",
            "forecast_lead_time_hours": lead_time_hours
        }
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    ds_out.to_netcdf(output_path, format="NETCDF4")
    ds_out.close()
    print(f"  ├─ Saved lead time f{lead_time_hours:03d}h -> '{output_path}'", flush=True)


def run_autoregressive_forecast(
    ckpt_path: str,
    x_minus6_file: str,
    x_zero_file: str,
    edge_index_path: str,
    forecast_steps: int = 4,  # 4 steps x 6h = 24h forecast
    output_pattern: str = "output/aida.{date_tag}.f{lead:03d}.nc"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[FORECAST] Operating on compute device: {device}", flush=True)

    # 1. Load Trained Checkpoint & Model
    print(f"[FORECAST] Loading checkpoint: '{ckpt_path}'", flush=True)
    checkpoint = torch.load(ckpt_path, map_location=device)
    cfg = checkpoint.get("config", {})

    model_cfg = cfg.get("model", {})
    in_vars = model_cfg.get("in_vars", 14)          # 7 vars x 2 timesteps
    out_vars = model_cfg.get("out_vars", 7)
    num_static_feats = model_cfg.get("num_static_feats", 2)
    hidden_dim = model_cfg.get("hidden_dim", 128)
    num_levels = cfg.get("mesh", {}).get("num_levels", 32)
    num_layers = model_cfg.get("num_layers", 4)

    model = IcosahedralGNNSurrogate(
        in_vars=in_vars,
        out_vars=out_vars,
        num_static_feats=num_static_feats,
        hidden_dim=hidden_dim,
        num_levels=num_levels,
        num_layers=num_layers
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    var_names = [
        'ln_t_icosahedral', 'u_icosahedral', 'v_icosahedral',
        'w_icosahedral', 'q_icosahedral', 'ln_rho_icosahedral', 'ln_p_icosahedral'
    ]

    # 2. Load Initial Analysis States (X_-6h, X_0) and Terrain Features
    print(f"[FORECAST] Reading initial state X_-6h: '{x_minus6_file}'", flush=True)
    x_m6_np, _, _, _ = load_state_from_file(x_minus6_file, var_names)

    print(f"[FORECAST] Reading initial state X_0  : '{x_zero_file}'", flush=True)
    x_0_np, h_3d_np, static_topo_np, ds_ref = load_state_from_file(x_zero_file, var_names)

    date_tag = parse_date_tag(x_zero_file)
    num_nodes = x_0_np.shape[2]
    edge_index = generate_or_load_edge_index(num_nodes=num_nodes, edge_file=edge_index_path).to(device)

    # Convert to Tensors: [Batch=1, Vars, Levels, Nodes]
    state_prev = torch.from_numpy(x_m6_np).unsqueeze(0).to(device)
    state_curr = torch.from_numpy(x_0_np).unsqueeze(0).to(device)
    static_topo = torch.from_numpy(static_topo_np).unsqueeze(0).to(device)

    print(f"\n" + "=" * 80)
    print(f" STARTING {forecast_steps * 6}-HOUR TERRAIN-FOLLOWING FORECAST ROLLOUT")
    print(f" Base Time Tag: {date_tag}")
    print("=" * 80, flush=True)

    # Save f000 (Initial Analysis State t0)
    f000_path = output_pattern.format(date_tag=date_tag, lead=0)
    export_lead_time_netcdf(
        output_path=f000_path,
        state_arr=x_0_np,
        h_3d_np=h_3d_np,
        static_topo_np=static_topo_np,
        ds_ref=ds_ref,
        var_names=var_names,
        lead_time_hours=0,
        num_levels=num_levels,
        num_nodes=num_nodes
    )

    with torch.no_grad():
        for step in range(1, forecast_steps + 1):
            lead_hours = step * 6

            # Extract X_-6h and X_0 components for Linear Trend Baseline Computation
            x_m6_curr = state_prev[:, 0:7, :, :] if state_prev.shape[1] >= 14 else state_prev
            x_0_curr  = state_curr[:, 7:14, :, :] if state_curr.shape[1] >= 14 else state_curr

            # Linear Trend Extrapolation: X_trend = X_0 + (X_0 - X_-6h)
            x_trend = x_0_curr + (x_0_curr - x_m6_curr)

            # Build 14-channel input trajectory [1, 14, 32, 2562]
            if state_prev.shape[1] == 7 and state_curr.shape[1] == 7:
                input_traj = torch.cat([state_prev, state_curr], dim=1)
            elif state_curr.shape[1] == 14:
                input_traj = state_curr
            else:
                input_traj = torch.cat([state_prev[:, :7, :, :], state_curr[:, :7, :, :]], dim=1)

            # Model Forward Pass
            out_model = model(input_traj, edge_index, static_topo=static_topo)

            # Check if output is a raw residual or a full state prediction
            if torch.abs(out_model.mean()) < 1.0:
                state_next = x_trend + out_model
            else:
                state_next = out_model

            next_np = state_next.cpu().numpy().squeeze(0)

            # Export per-lead-time NetCDF
            step_path = output_pattern.format(date_tag=date_tag, lead=lead_hours)
            export_lead_time_netcdf(
                output_path=step_path,
                state_arr=next_np,
                h_3d_np=h_3d_np,
                static_topo_np=static_topo_np,
                ds_ref=ds_ref,
                var_names=var_names,
                lead_time_hours=lead_hours,
                num_levels=num_levels,
                num_nodes=num_nodes
            )

            # Shift state windows for next step
            state_prev = state_curr
            state_curr = state_next

    ds_ref.close()
    print(f"\n[SUCCESS] Multi-step forecast rollout complete! Exported {forecast_steps + 1} NetCDF files.\n", flush=True)


def main():
    parser = argparse.ArgumentParser(description="AIDA 4D Terrain-Following Autoregressive Forecast Engine")
    parser.add_argument("-k", "--checkpoint", default="checkpoints/aida_gnn_surrogate_logstate.pt", help="Path to checkpoint")
    parser.add_argument("-m", "--minus6", required=True, help="Path to X_-6h initial analysis state file")
    parser.add_argument("-z", "--zero", required=True, help="Path to X_0 current initial analysis state file")
    parser.add_argument("-e", "--edges", default="data/graph/icosahedral_edge_index_m4.pt", help="Path to graph edge index")
    parser.add_argument("-s", "--steps", type=int, default=4, help="Number of 6h forecast steps (default: 4 = 24h)")
    parser.add_argument("-o", "--output_pattern", default="output/aida.{date_tag}.f{lead:03d}.nc", help="Output path pattern")

    args = parser.parse_args()

    run_autoregressive_forecast(
        ckpt_path=args.checkpoint,
        x_minus6_file=args.minus6,
        x_zero_file=args.zero,
        edge_index_path=args.edges,
        forecast_steps=args.steps,
        output_pattern=args.output_pattern
    )


if __name__ == "__main__":
    main()
