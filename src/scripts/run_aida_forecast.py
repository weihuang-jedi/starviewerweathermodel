#!/usr/bin/env python3
"""
scripts/run_aida_forecast.py
----------------------------
Autoregressive Forecast Rollout Engine using the trained 4D Terrain-Following AIDA Checkpoint.
Infers X_+6h, X_+12h, X_+18h... from initial analysis state pair (X_-6h, X_0)
while conditioning on static topography (static_topo), 3D terrain heights (h_3d),
and dynamic Solar Zenith Angle cos(SZA) solar forcing to eliminate hemisphere thermal drift.

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


def compute_solar_zenith_angle(
    lats_deg: np.ndarray,
    lons_deg: np.ndarray,
    year: int = 2026,
    month: int = 1,
    day: int = 1,
    hour_utc: float = 0.0
) -> np.ndarray:
    """
    Computes the cosine of the Solar Zenith Angle cos(SZA) across all mesh nodes.
    
    Returns:
        np.ndarray: Array of shape (num_nodes,) with values in [0.0, 1.0].
                    0.0 indicates nighttime (sun below horizon), >0.0 indicates daytime solar forcing.
    """
    rad = np.pi / 180.0

    # Calculate day of year (1 to 365)
    from datetime import datetime
    dt = datetime(year, month, day)
    day_of_year = dt.timetuple().tm_yday

    # Solar declination angle (radians)
    declination = 23.45 * np.sin(rad * (360.0 / 365.0) * (day_of_year - 81)) * rad

    # Solar Hour Angle (SHA) in radians
    # Solar time = UTC_time + (longitude / 15.0 degrees per hour)
    solar_time = hour_utc + (lons_deg / 15.0)
    hour_angle = (solar_time - 12.0) * 15.0 * rad

    lats_rad = lats_deg * rad

    # Cosine Solar Zenith Angle formula: cos(SZA) = sin(lat)*sin(dec) + cos(lat)*cos(dec)*cos(HA)
    cos_sza = np.sin(lats_rad) * np.sin(declination) + np.cos(lats_rad) * np.cos(declination) * np.cos(hour_angle)

    # Day/Night thresholding: Sun above horizon
    return np.maximum(0.0, cos_sza).astype(np.float32)


def load_state_from_file(file_path: str, var_names: list):
    """Helper to extract dynamic 7-variable log-state tensor, 3D terrain heights, static topography, and coordinates."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"[ERROR] Input state file not found: '{file_path}'")

    if file_path.endswith('.zarr'):
        ds = xr.open_zarr(file_path)
    else:
        ds = xr.open_dataset(file_path)

    # 1. Extract 7 dynamic state variables [7, Levels=32, Nodes]
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

    # 2. Extract Coordinates (latitude, longitude)
    lats = ds['latitude'].values if 'latitude' in ds else (ds['lat'].values if 'lat' in ds else np.linspace(-90, 90, state_np.shape[2]))
    lons = ds['longitude'].values if 'longitude' in ds else (ds['lon'].values if 'lon' in ds else np.linspace(-180, 180, state_np.shape[2]))

    if lats.ndim > 1:
        lats = lats[0]
    if lons.ndim > 1:
        lons = lons[0]

    # Ensure longitudes are in [-180, 180]
    lons = np.where(lons > 180.0, lons - 360.0, lons)

    # 3. Extract 3D Terrain-Following Geometric Heights h_3d [32, Nodes]
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

    # 4. Extract Static Surface Topography [2, Nodes] (Elevation + Land-Sea Mask)
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

    return state_np, h_3d_np.astype(np.float32), static_topo_np, lats.astype(np.float32), lons.astype(np.float32), ds


def parse_date_tag(filename: str) -> tuple[str, int, int, int, int]:
    """Extracts date tag string, year, month, day, and base UTC hour from filename."""
    match = re.search(r'(\d{4})(\d{2})(\d{2})\.t(\d{2})z', os.path.basename(filename))
    if match:
        year, month, day, hour = int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))
        date_tag = match.group(0).replace('.nc', '')
        return date_tag, year, month, day, hour
    return "forecast", 2026, 1, 1, 0


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

    # 1. Primary Physical Weather Variables
    for idx, var in enumerate(var_names):
        out_var_name = var if var.endswith("_icosahedral") else f"{var}_icosahedral"

        # Copy original attributes if present in reference dataset
        var_attrs = {"long_name": f"Forecasted {var}", "mesh": "icosahedral_mesh"}
        if out_var_name in ds_ref:
            var_attrs.update(ds_ref[out_var_name].attrs)

        data_vars_out[out_var_name] = (
            ["level", "node"],
            state_arr[idx, :, :].astype(np.float32),
            var_attrs
        )

    # 2. 3D Geometric Heights
    h_attrs = {"units": "meters", "long_name": "3D Terrain-Following Geometric Height Above Sea Level", "mesh": "icosahedral_mesh"}
    if "h_icosahedral" in ds_ref:
        h_attrs.update(ds_ref["h_icosahedral"].attrs)

    data_vars_out["h_icosahedral"] = (
        ["level", "node"],
        h_3d_np,
        h_attrs
    )

    # 3. Surface Topography Elevation
    data_vars_out["h_terrain_icosahedral"] = (
        ["node"],
        static_topo_np[0] * 10000.0,
        {"units": "meters", "long_name": "Surface Topography Elevation", "mesh": "icosahedral_mesh"}
    )

    # -------------------------------------------------------------------------
    # 4. COPY TARGET_LEVEL AND ETA DIRECTLY FROM REFERENCE DATASET (ds_ref)
    # -------------------------------------------------------------------------
    if "eta" in ds_ref:
        data_vars_out["eta"] = ds_ref["eta"]
    else:
        data_vars_out["eta"] = (
            ["level"],
            np.linspace(0.0, 1.0, num_levels, dtype=np.float32),
            {"long_name": "Eta Coordinate Coefficient", "units": "1"}
        )

    if "target_level" in ds_ref:
        data_vars_out["target_level"] = ds_ref["target_level"]
    else:
        data_vars_out["target_level"] = (
            ["level"],
            np.arange(1, num_levels + 1, dtype=np.int32),
            {"long_name": "Baseline Flat-Terrain Height Level", "units": "meters"}
        )

    # 5. Copy Static Surface & Mesh Topology Variables from Reference File
    static_vars = [
        "longitude", "latitude", "face_nodes", "x_cartesian", "y_cartesian",
        "z_cartesian", "land_sea_mask", "elevation", "h_terrain", "icosahedral_mesh"
    ]
    for static_var in static_vars:
        if static_var in ds_ref:
            data_vars_out[static_var] = ds_ref[static_var]

    # Coordinates Setup
    coords_out = {
        "level": ds_ref["level"].values if "level" in ds_ref else np.arange(1, num_levels + 1, dtype=np.int32),
        "node": ds_ref["node"].values if "node" in ds_ref else np.arange(num_nodes, dtype=np.int32)
    }

    if "face" in ds_ref.dims:
        coords_out["face"] = ds_ref["face"].values
    if "three" in ds_ref.dims:
        coords_out["three"] = ds_ref["three"].values

    ds_out = xr.Dataset(
        data_vars=data_vars_out,
        coords=coords_out,
        attrs={
            "title": getattr(ds_ref, "title", "AIDA GNN 4D Observation-Guided Terrain Weather Forecast"),
            "conventions": "CF-1.8 UGRID-1.0",
            "forecast_lead_time_hours": lead_time_hours
        }
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    ds_out.to_netcdf(output_path, format="NETCDF4")
    ds_out.close()
    print(f"  ├─ Saved lead time f{lead_time_hours:03d}h -> '{output_path}' (copied target_level & eta from ds_ref)", flush=True)


def run_autoregressive_forecast(
    ckpt_path: str,
    x_minus6_file: str,
    x_zero_file: str,
    edge_index_path: str,
    forecast_steps: int = 4,
    output_pattern: str = "output/aida.{date_tag}.f{lead:03d}.nc"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[FORECAST] Operating on compute device: {device}", flush=True)

    print(f"[FORECAST] Loading checkpoint: '{ckpt_path}'", flush=True)
    checkpoint = torch.load(ckpt_path, map_location=device)
    cfg = checkpoint.get("config", {})

    model_cfg = cfg.get("model", {})
    in_vars = model_cfg.get("in_vars", 14)
    out_vars = model_cfg.get("out_vars", 7)
    num_static_feats = model_cfg.get("num_static_feats", 3)
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

    print(f"[FORECAST] Reading initial state X_-6h: '{x_minus6_file}'", flush=True)
    x_m6_np, _, _, _, _, _ = load_state_from_file(x_minus6_file, var_names)

    print(f"[FORECAST] Reading initial state X_0  : '{x_zero_file}'", flush=True)
    x_0_np, h_3d_np, static_topo_np, lats_deg, lons_deg, ds_ref = load_state_from_file(x_zero_file, var_names)

    date_tag, base_year, base_month, base_day, base_hour_utc = parse_date_tag(x_zero_file)
    num_nodes = x_0_np.shape[2]
    edge_index = generate_or_load_edge_index(num_nodes=num_nodes, edge_file=edge_index_path).to(device)

    # Convert to Tensors: [Batch=1, Vars, Levels, Nodes]
    state_prev = torch.from_numpy(x_m6_np).unsqueeze(0).to(device)
    state_curr = torch.from_numpy(x_0_np).unsqueeze(0).to(device)
    static_topo = torch.from_numpy(static_topo_np).unsqueeze(0).to(device)

    print(f"\n" + "=" * 80)
    print(f" STARTING {forecast_steps * 6}-HOUR TERRAIN-FOLLOWING FORECAST ROLLOUT")
    print(f" Base Time Tag: {date_tag} (Year:{base_year}, Month:{base_month}, Day:{base_day}, Hour:{base_hour_utc:02d}z)")
    print("=" * 80, flush=True)

    # Save f000 Initial State
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
            current_hour_utc = (base_hour_utc + lead_hours) % 24

            # Compute Dynamic Solar Zenith Angle Forcing cos(SZA) for current forecast hour
            cos_sza_np = compute_solar_zenith_angle(
                lats_deg=lats_deg,
                lons_deg=lons_deg,
                year=base_year,
                month=base_month,
                day=base_day,
                hour_utc=current_hour_utc
            )

            """
            # Extract X_-6h and X_0 components for Linear Trend Baseline Computation
            x_m6_curr = state_prev[:, 0:7, :, :] if state_prev.shape[1] >= 14 else state_prev
            x_0_curr  = state_curr[:, 7:14, :, :] if state_curr.shape[1] >= 14 else state_curr

            # Linear Trend Extrapolation: X_trend = X_0 + (X_0 - X_-6h)
            x_trend = x_0_curr + (x_0_curr - x_m6_curr)

            # Build 14-channel input trajectory
            if state_prev.shape[1] == 7 and state_curr.shape[1] == 7:
                input_traj = torch.cat([state_prev, state_curr], dim=1)
            elif state_curr.shape[1] == 14:
                input_traj = state_curr
            else:
                input_traj = torch.cat([state_prev[:, :7, :, :], state_curr[:, :7, :, :]], dim=1)

            # Forward pass through GNN surrogate
            out_model = model(input_traj, edge_index, static_topo=static_topo)
            """

            # Compute Dynamic Solar Zenith Angle Forcing cos(SZA) for current forecast hour
            cos_sza_np = compute_solar_zenith_angle(
                lats_deg=lats_deg,
                lons_deg=lons_deg,
                year=base_year,
                month=base_month,
                day=base_day,
                hour_utc=current_hour_utc
            )

            # -----------------------------------------------------------------
            # FIX: Convert cos(SZA) to Tensor & Concatenate onto static_topo
            # Output Shape: [Batch=1, Static_Feats=3, Nodes]
            # -----------------------------------------------------------------
            cos_sza_tensor = torch.from_numpy(cos_sza_np).unsqueeze(0).unsqueeze(0).to(device) # [1, 1, Nodes]

            # Combine [Elevation, LSM] (2 channels) + [cos_sza] (1 channel) -> 3 channels
            static_topo_3ch = torch.cat([static_topo[:, :2, :], cos_sza_tensor], dim=1)

            # Extract X_-6h and X_0 components for Linear Trend Baseline Computation
            x_m6_curr = state_prev[:, 0:7, :, :] if state_prev.shape[1] >= 14 else state_prev
            x_0_curr  = state_curr[:, 7:14, :, :] if state_curr.shape[1] >= 14 else state_curr

            # Linear Trend Extrapolation: X_trend = X_0 + (X_0 - X_-6h)
            x_trend = x_0_curr + (x_0_curr - x_m6_curr)

            # Build 14-channel input trajectory
            if state_prev.shape[1] == 7 and state_curr.shape[1] == 7:
                input_traj = torch.cat([state_prev, state_curr], dim=1)
            elif state_curr.shape[1] == 14:
                input_traj = state_curr
            else:
                input_traj = torch.cat([state_prev[:, :7, :, :], state_curr[:, :7, :, :]], dim=1)

            # Forward pass through GNN surrogate using 3-channel static topology
            out_model = model(input_traj, edge_index, static_topo=static_topo_3ch)

            # -----------------------------------------------------------------
            # 1. CLAMP 6-HOUR INCREMENT DELTAS (Damps Exponential Blow-ups)
            # Maximum allowed 6-hour physical shifts:
            # ln_T: ±0.035 (~10K), U/V: ±20 m/s, W: ±2 Pa/s, Q: ±0.005 kg/kg
            # -----------------------------------------------------------------
            delta_max = torch.tensor([0.035, 20.0, 20.0, 2.0, 0.005, 0.1, 0.02], device=device).view(1, 7, 1, 1)
            out_model = torch.clamp(out_model, min=-delta_max, max=delta_max)

            # Compute Next State: X_next = X_trend + delta_X
            if torch.abs(out_model.mean()) < 1.0:
                state_next = x_trend + out_model
            else:
                state_next = out_model

            # -----------------------------------------------------------------
            # 2. ABSOLUTE PHYSICAL BOUNDARY GUARDS
            # Var order: [ln_T (0), U (1), V (2), W (3), Q (4), ln_RHO (5), ln_P (6)]
            # -----------------------------------------------------------------
            # Temperature T: clamp ln_T to [ln(180K), ln(330K)]
            state_next[:, 0, :, :] = torch.clamp(state_next[:, 0, :, :], min=5.19295, max=5.79909)

            # Winds U, V: clamp to [-90.0 m/s, +90.0 m/s]
            state_next[:, 1, :, :] = torch.clamp(state_next[:, 1, :, :], min=-90.0, max=90.0)
            state_next[:, 2, :, :] = torch.clamp(state_next[:, 2, :, :], min=-90.0, max=90.0)

            # Specific Humidity Q: Strictly NON-NEGATIVE [0.0, 0.035 kg/kg]
            state_next[:, 4, :, :] = torch.clamp(state_next[:, 4, :, :], min=0.0, max=0.035)

            # Log Pressure ln_P: clamp to [ln(100Pa), ln(108000Pa)]
            state_next[:, 6, :, :] = torch.clamp(state_next[:, 6, :, :], min=4.60517, max=11.58988)

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
    parser.add_argument("-e", "--edges", default="data/graph/icosahedral_edge_index_m6.pt", help="Path to graph edge index")
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
