#!/usr/bin/env python3
"""
scripts/run_aida_forecast.py
-----------------------------
Autoregressive Evaluation & Inference Pipeline for AIDA GNN Model.
Executes multi-step forecast rollouts and exports complete NetCDF files
cloned directly from the truth/initial NetCDF dataset structure and attributes.
"""

import os
import sys
import argparse
import numpy as np
import netCDF4 as nc
import torch

# Ensure src directory is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.gnn import IcosahedralGNNSurrogate


def remove_pentagon_artifacts(field_3d, edge_index, num_nodes=40962, passes=2):
    """
    Applies multi-pass 2-hop ring smoothing strictly to the 12 pentagonal nodes
    and their immediate neighbor ring on a physical 3D field [Levels, Nodes].
    """
    src_raw, dst_raw = edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy()
    base_mask = (src_raw < num_nodes) & (dst_raw < num_nodes)
    src, dst = src_raw[base_mask], dst_raw[base_mask]

    deg = np.bincount(dst, minlength=num_nodes)
    pent_idx = np.where(deg == 5)[0]  # Exactly 12 pentagon nodes

    field_clean = field_3d.copy() if isinstance(field_3d, np.ndarray) else field_3d.clone()

    for _ in range(passes):
        for p_node in pent_idx:
            # 1-hop neighbors
            nbrs1 = src[dst == p_node]
            if len(nbrs1) == 0:
                nbrs1 = dst[src == p_node]
            nbrs1 = np.unique(nbrs1)

            # 2-hop neighbors (neighbors of 1-hop nodes excluding the pentagon itself)
            nbrs2_list = []
            for n1 in nbrs1:
                n2 = src[dst == n1]
                nbrs2_list.extend(n2)
            nbrs2 = np.unique(nbrs2_list)
            nbrs2 = nbrs2[nbrs2 != p_node]

            # Smooth 1-hop ring nodes using 2-hop context
            for n1 in nbrs1:
                sub_nbrs = src[dst == n1]
                sub_nbrs = sub_nbrs[sub_nbrs != p_node]
                if len(sub_nbrs) > 0:
                    if isinstance(field_clean, np.ndarray):
                        field_clean[:, n1] = np.mean(field_clean[:, sub_nbrs], axis=-1)
                    else:
                        field_clean[:, n1] = torch.mean(field_clean[:, sub_nbrs], dim=-1)

            # Smooth center pentagon node
            if isinstance(field_clean, np.ndarray):
                field_clean[:, p_node] = np.mean(field_clean[:, nbrs1], axis=-1)
            else:
                field_clean[:, p_node] = torch.mean(field_clean[:, nbrs1], dim=-1)

    return field_clean

def remove_pentagon_artifacts_1hop(field_3d, edge_index, num_nodes=40962):
    """
    Applies 1-hop ring smoothing strictly to the 12 pentagonal nodes on a physical 3D field [Levels, Nodes].
    """
    src_raw, dst_raw = edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy()
    base_mask = (src_raw < num_nodes) & (dst_raw < num_nodes)
    src, dst = src_raw[base_mask], dst_raw[base_mask]

    deg = np.bincount(dst, minlength=num_nodes)
    pent_idx = np.where(deg == 5)[0]  # Exactly 12 nodes

    field_clean = field_3d.copy() if isinstance(field_3d, np.ndarray) else field_3d.clone()

    for p_node in pent_idx:
        neighbors = src[dst == p_node]
        if len(neighbors) == 0:
            neighbors = dst[src == p_node]
        neighbors = np.unique(neighbors)[:5]

        if isinstance(field_clean, np.ndarray):
            field_clean[:, p_node] = np.mean(field_clean[:, neighbors], axis=-1)
        else:
            field_clean[:, p_node] = torch.mean(field_clean[:, neighbors], dim=-1)

    return field_clean


def load_logstate_file(filepath):
    """
    Loads initial NetCDF file into natural log-states [1, 7, 32, 40962]
    and extracts static topography channels.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"[ERROR] Required file not found: '{filepath}'")

    with nc.Dataset(filepath, "r") as ds:
        var_spec = [
            ["ln_t_icosahedral", "log_T", "ln_T", "T"],          # Ch 0: ln(T) ~ [5.10, 5.85]
            ["u_icosahedral", "u", "U"],                         # Ch 1: u
            ["v_icosahedral", "v", "V"],                         # Ch 2: v
            ["w_icosahedral", "w", "W"],                         # Ch 3: w
            ["q_icosahedral", "q", "Q"],                         # Ch 4: q
            ["ln_rho_icosahedral", "log_rho", "ln_rho", "rho"], # Ch 5: ln(rho)
            ["ln_p_icosahedral", "log_P", "ln_P", "P"]          # Ch 6: ln(P_Pa) ~ [4.60, 11.60]
        ]

        data_list = []
        for aliases in var_spec:
            found_key = next((k for k in aliases if k in ds.variables), None)
            if found_key is None:
                raise KeyError(f"[ERROR] Missing key {aliases} in '{filepath}'")

            var_data = ds.variables[found_key][:]
            if var_data.ndim == 2:
                var_data = var_data[np.newaxis, ...]

            # Convert raw physical variables to natural log IF necessary
            if found_key == "T":
                var_data = np.log(np.maximum(var_data, 100.0))
            elif found_key == "P":
                p_pa = var_data * 100.0 if np.mean(var_data) < 2000.0 else var_data
                var_data = np.log(np.maximum(p_pa, 1.0))
            elif found_key == "rho":
                var_data = np.log(np.maximum(var_data, 1e-6))

            data_list.append(var_data)

        state = np.stack(data_list, axis=1)  # [1, 7, Levels, Nodes]

        # Extract coordinates and terrain for static topo
        lats = ds.variables["latitude"][:] if "latitude" in ds.variables else None
        lons = ds.variables["longitude"][:] if "longitude" in ds.variables else None

        num_nodes = state.shape[-1]
        h_topo = ds.variables["h_terrain"][:] if "h_terrain" in ds.variables else (ds.variables["h_terrain_icosahedral"][:] if "h_terrain_icosahedral" in ds.variables else np.zeros(num_nodes))
        lsm = ds.variables["land_sea_mask"][:] if "land_sea_mask" in ds.variables else np.zeros(num_nodes)

        lat_norm = (lats / 90.0) if lats is not None else np.zeros(num_nodes)
        lon_norm = (lons / 180.0) if lons is not None else np.zeros(num_nodes)

        static_topo = np.stack([h_topo / 1000.0, lsm, lat_norm, lon_norm], axis=0)

        return torch.from_numpy(state).float(), torch.from_numpy(static_topo).float()


def save_forecast_netcdf(output_filepath, pred_tensor, edge_index, template_filepath):
    """
    Clones global attributes, dimensions, coordinates, and topological variables
    directly from template_filepath and writes model forecast fields in natural log space.
    """
    os.makedirs(os.path.dirname(output_filepath) or ".", exist_ok=True)

    pred_data = pred_tensor.squeeze(0).cpu().numpy()  # [7, 32, 40962]

    # 1. Pentagon 1-hop ring smoothing directly on natural log predictions
    pred_data[0] = remove_pentagon_artifacts(pred_data[0], edge_index)  # ln(T)
    pred_data[6] = remove_pentagon_artifacts(pred_data[6], edge_index)  # ln(P_Pa)

    with nc.Dataset(template_filepath, "r") as src, nc.Dataset(output_filepath, "w", format="NETCDF4") as dst:
        # Clone Global Attributes
        dst.setncatts({k: src.getncattr(k) for k in src.ncattrs()})

        # Clone Dimensions
        for name, dimension in src.dimensions.items():
            dst.createDimension(name, (len(dimension) if not dimension.isunlimited() else None))

        # Copy non-dynamic variables (coordinates, mesh topology, terrain, elevation)
        dynamic_fcst_keys = [
            "ln_t_icosahedral", "ln_p_icosahedral", "ln_rho_icosahedral",
            "u_icosahedral", "v_icosahedral", "w_icosahedral", "q_icosahedral"
        ]

        for var_name, var in src.variables.items():
            if var_name not in dynamic_fcst_keys:
                out_var = dst.createVariable(var_name, var.datatype, var.dimensions)
                out_var.setncatts({k: var.getncattr(k) for k in var.ncattrs()})
                out_var[:] = var[:]

        # Populate forecast dynamic fields with exact truth variable names and attributes
        field_payload = {
            "ln_t_icosahedral": pred_data[0],   # ln(T) in Kelvin
            "u_icosahedral": pred_data[1],      # u wind
            "v_icosahedral": pred_data[2],      # v wind
            "w_icosahedral": pred_data[3],      # w velocity
            "q_icosahedral": pred_data[4],      # specific humidity
            "ln_rho_icosahedral": pred_data[5], # ln(rho) in kg/m3
            "ln_p_icosahedral": pred_data[6],   # ln(P) in Pa
        }

        for var_name, data_arr in field_payload.items():
            if var_name in src.variables:
                template_var = src.variables[var_name]
                out_var = dst.createVariable(var_name, template_var.datatype, template_var.dimensions)
                out_var.setncatts({k: template_var.getncattr(k) for k in template_var.ncattrs()})
            else:
                out_var = dst.createVariable(var_name, "f4", ("level", "node"))

            out_var[:] = data_arr

    print(f"[FORECAST] Exported full forecast step to: '{output_filepath}'")


def run_forecast():
    parser = argparse.ArgumentParser(description="AIDA GNN Autoregressive Forecast Runner")
    parser.add_argument("-e", "--edge_index", type=str, required=True, help="Path to graph edge_index .pt file")
    parser.add_argument("-k", "--checkpoint", type=str, required=True, help="Path to model checkpoint .pt file")
    parser.add_argument("-s", "--steps", type=int, default=20, help="Number of 6-hour forecast rollout steps")
    parser.add_argument("-m", "--m06h_file", type=str, required=True, help="Path to -6h initial logstate NetCDF file")
    parser.add_argument("-z", "--zero_file", type=str, required=True, help="Path to 0h initial logstate NetCDF file")
    parser.add_argument("-o", "--output_template", type=str, required=True, help="Output template string, e.g. fcst.f{lead:03d}.nc")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[FORECAST] Operating on compute device: {device}")

    # 1. Load Topology Edge Index
    edge_data = torch.load(args.edge_index, map_location=device)
    if isinstance(edge_data, dict):
        edge_index = edge_data["edge_index"].to(device)
        edge_index_vert = edge_data.get("edge_index_vert", None)
        if edge_index_vert is not None:
            edge_index_vert = edge_index_vert.to(device)
    else:
        edge_index = edge_data.to(device)
        edge_index_vert = None

    # 2. Build and Load GNN Model
    print(f"[FORECAST] Loading Model Checkpoint: '{args.checkpoint}'...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    num_layers = 8 if any("gnn_layers.4" in k for k in state_dict.keys()) else 4

    model = IcosahedralGNNSurrogate(
        in_vars=14,
        out_vars=7,
        num_static_feats=4,
        hidden_dim=256,
        num_levels=32,
        num_layers=num_layers
    ).to(device)

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if unexpected_keys:
        print(f"[FORECAST] Ignored non-matching keys: {unexpected_keys}")
    model.eval()

    # 3. Load Initial Background States (-6h and 0h)
    print(f"[FORECAST] Ingesting -6h state: '{args.m06h_file}'")
    print(f"[FORECAST] Ingesting  0h state: '{args.zero_file}'")

    x_m6h, _ = load_logstate_file(args.m06h_file)
    x_0h, static_topo = load_logstate_file(args.zero_file)

    x_m6h = x_m6h.to(device)
    x_0h = x_0h.to(device)
    static_topo = static_topo.to(device)

    # Combine into 14-channel initial condition: [1, 14, 32, 40962]
    curr_input = torch.cat([x_m6h, x_0h], dim=1)

    # 4. Autoregressive Rollout Loop
    print(f"[FORECAST] Starting {args.steps}-step ({args.steps * 6} hours) Autoregressive Rollout...")

    with torch.no_grad():
        for step in range(1, args.steps + 1):
            lead_hours = step * 6
            output_filepath = args.output_template.format(lead=lead_hours)

            # Predict next step log-state [1, 7, 32, 40962]
            pred_next = model(
                curr_input,
                edge_index,
                edge_index_vert=edge_index_vert,
                static_topo=static_topo
            )

            # Save NetCDF prediction cloned directly from zero_file
            save_forecast_netcdf(
                output_filepath,
                pred_next,
                edge_index,
                template_filepath=args.zero_file
            )

            # Shift state for next autoregressive step
            curr_input = torch.cat([curr_input[:, 7:14, :, :], pred_next], dim=1)

    print(f"[SUCCESS] Completed all {args.steps} rollout steps successfully!")


if __name__ == "__main__":
    run_forecast()
