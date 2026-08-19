#!/usr/bin/env python3
"""
utils/plot_forecast_leadtime_series.py
---------------------------------------
Generates a 3x5 multi-panel lead-time progression plot for a single variable
at a given vertical level across forecast hours (+00h, +06h, +12h, +18h, +24h).

Panels:
  Row 1: Forecast  (f000, f006, f012, f018, f024)
  Row 2: Truth     (t06z, t12z, t18z, t00z_next, t06z_next)
  Row 3: Error     (Forecast - Truth)
"""

import argparse
import os
import glob
import re
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.interpolate import griddata

R_D = 287.058


def extract_variable_field(nc_file: str, var_name: str, level_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extracts a 2D physical field at a specified vertical level."""
    if not os.path.exists(nc_file):
        raise FileNotFoundError(f"[ERROR] Required NetCDF file not found: '{nc_file}'")

    ds = xr.open_dataset(nc_file)

    def get_var(candidates):
        for c in candidates:
            if c in ds:
                val = ds[c].values
                if val.ndim == 3:  # Squeeze time or lead_time dimension
                    val = val[0]
                return val
        return None

    lons = get_var(['longitude', 'lon'])
    lats = get_var(['latitude', 'lat'])

    if lons is None or lats is None:
        raise KeyError(f"[ERROR] Missing latitude/longitude in '{nc_file}'")

    var_upper = var_name.upper()

    if var_upper == 'T':
        val = get_var(['ln_t_icosahedral', 'ln_t', 't_icosahedral', 't'])
        field = np.exp(val[level_idx]) if np.nanmean(val) < 10.0 else val[level_idx]
    elif var_upper == 'P':
        val = get_var(['ln_p_icosahedral', 'ln_p', 'p_icosahedral', 'p'])
        field = np.exp(val[level_idx]) / 100.0 if np.nanmean(val) < 20.0 else val[level_idx]
        if np.nanmean(field) > 2000.0:
            field = field / 100.0
    elif var_upper == 'Q':
        val = get_var(['q_icosahedral', 'q'])
        field = val[level_idx] * 1000.0 if np.nanmean(val) < 0.1 else val[level_idx]
    elif var_upper == 'U':
        val = get_var(['u_icosahedral', 'u'])
        field = val[level_idx]
    elif var_upper == 'V':
        val = get_var(['v_icosahedral', 'v'])
        field = val[level_idx]
    elif var_upper == 'W':
        val = get_var(['w_icosahedral', 'w'])
        field = val[level_idx]
    elif var_upper == 'RHO':
        val = get_var(['ln_rho_icosahedral', 'ln_rho', 'rho_icosahedral', 'rho'])
        if val is not None:
            field = np.exp(val[level_idx]) if np.nanmean(val) < 2.0 else val[level_idx]
        else:
            # Fallback rho calculation
            t_val = get_var(['ln_t_icosahedral', 'ln_t', 't_icosahedral', 't'])[level_idx]
            p_val = get_var(['ln_p_icosahedral', 'ln_p', 'p_icosahedral', 'p'])[level_idx]
            t_k = np.exp(t_val) if np.nanmean(t_val) < 10.0 else t_val
            p_hpa = np.exp(p_val) / 100.0 if np.nanmean(p_val) < 20.0 else p_val
            field = (p_hpa * 100.0) / (R_D * t_k)
    else:
        raise ValueError(f"[ERROR] Unsupported variable: '{var_name}'. Supported: T, P, Q, U, V, W, RHO")

    ds.close()
    return field, lons, lats


def interpolate_to_regular_grid(lons: np.ndarray, lats: np.ndarray, data: np.ndarray, grid_lon: np.ndarray, grid_lat: np.ndarray):
    """Interpolates unstructured icosahedral nodes onto a 2D regular grid."""
    lons_clean = np.where(lons > 180.0, lons - 360.0, lons)
    points = np.column_stack([lons_clean, lats])
    grid_z = griddata(points, data, (grid_lon, grid_lat), method='linear')

    nan_mask = np.isnan(grid_z)
    if np.any(nan_mask):
        grid_z_near = griddata(points, data, (grid_lon, grid_lat), method='nearest')
        grid_z[nan_mask] = grid_z_near[nan_mask]

    return grid_z


def plot_leadtime_series(
    fcst_files: list[str],
    truth_files: list[str],
    var_name: str = 'T',
    level_idx: int = 0,
    output_png: str = "forecast_leadtime_series.png"
):
    if len(fcst_files) != 5 or len(truth_files) != 5:
        raise ValueError("[ERROR] Must supply exactly 5 forecast files and 5 truth files.")

    lead_labels = ['+00h', '+06h', '+12h', '+18h', '+24h']
    var_titles = {
        'T': ('Temperature', 'K'),
        'P': ('Pressure', 'hPa'),
        'Q': ('Specific Humidity', 'g/kg'),
        'U': ('Zonal Wind U', 'm/s'),
        'V': ('Meridional Wind V', 'm/s'),
        'W': ('Vertical Velocity W', 'Pa/s'),
        'RHO': ('Density ρ', 'kg/m³')
    }

    title_str, unit_str = var_titles.get(var_name.upper(), (var_name, ''))

    # Regular 2D Interpolation Grid (1.0 degree)
    reg_lon = np.linspace(-180, 180, 360)
    reg_lat = np.linspace(-90, 90, 180)
    grid_lon, grid_lat = np.meshgrid(reg_lon, reg_lat)

    fcst_grids, truth_grids, err_grids = [], [], []

    print(f"\n[SERIES PLOTTER] Processing Lead-Time Series for '{var_name}' at Level Index {level_idx + 1}...")

    for i in range(5):
        f_field, lons, lats = extract_variable_field(fcst_files[i], var_name, level_idx)
        t_field, _, _ = extract_variable_field(truth_files[i], var_name, level_idx)

        f_g = interpolate_to_regular_grid(lons, lats, f_field, grid_lon, grid_lat)
        t_g = interpolate_to_regular_grid(lons, lats, t_field, grid_lon, grid_lat)
        e_g = f_g - t_g

        fcst_grids.append(f_g)
        truth_grids.append(t_g)
        err_grids.append(e_g)

    # Calculate Colorbar Limits
    all_fcst_truth = np.concatenate([fcst_grids, truth_grids])
    vmin_state, vmax_state = np.nanmin(all_fcst_truth), np.nanmax(all_fcst_truth)

    vlim_err = max(abs(np.nanmin(err_grids)), abs(np.nanmax(err_grids)))
    if vlim_err < 1e-4:
        vlim_err = 1e-3

    fig = plt.figure(figsize=(24, 11))
    proj = ccrs.PlateCarree()

    rows, cols = 3, 5
    row_labels = ["Forecast", "Truth", "Error (Fcst - Truth)"]

    for col in range(cols):
        grid_data_list = [fcst_grids[col], truth_grids[col], err_grids[col]]

        for row in range(rows):
            ax = fig.add_subplot(rows, cols, row * cols + col + 1, projection=proj)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color='black', alpha=0.7)
            ax.add_feature(cfeature.BORDERS, linewidth=0.3, color='gray', alpha=0.5)

            data = grid_data_list[row]

            if row == 2:  # Error Row (Diverging Colormap)
                sc = ax.pcolormesh(grid_lon, grid_lat, data, cmap='coolwarm', vmin=-vlim_err, vmax=vlim_err, shading='auto', transform=proj)
            else:       # Forecast & Truth Rows (Sequential Colormap)
                sc = ax.pcolormesh(grid_lon, grid_lat, data, cmap='viridis', vmin=vmin_state, vmax=vmax_state, shading='auto', transform=proj)

            # Titles & Subtitles
            if row == 0:
                ax.set_title(f"Lead Time {lead_labels[col]}\n{row_labels[row]}", fontsize=11, fontweight='bold')
            else:
                ax.set_title(f"{row_labels[row]}", fontsize=10, fontweight='bold')

            cbar = plt.colorbar(sc, ax=ax, orientation='horizontal', pad=0.05, shrink=0.85)
            cbar.ax.tick_params(labelsize=8)
            cbar.set_label(f"[{unit_str}]" if row != 2 else f"Error [{unit_str}]", fontsize=8)

    plt.suptitle(
        f"AIDA GNN 24-Hour Forecast Progression vs. Truth | Field: {title_str} ({unit_str}) | Level: {level_idx + 1} (1=Surface, 32=Top)",
        fontsize=16, fontweight='bold', y=0.99
    )

    os.makedirs(os.path.dirname(output_png) or ".", exist_ok=True)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_png, dpi=250, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"[SUCCESS] Multi-leadtime series plot saved to: '{output_png}'\n")


def main():
    parser = argparse.ArgumentParser(description="3x5 Lead-Time Series Plotter for AIDA GNN Forecasts")
    parser.add_argument("-v", "--variable", default="T", help="Variable to plot (T, P, Q, U, V, W, RHO)")
    parser.add_argument("-l", "--level", type=int, default=0, help="Vertical level index (0=Surface, 31=Top)")
    parser.add_argument("-o", "--output", default="aida_leadtime_series.png", help="Destination PNG plot path")
    parser.add_argument("--fcst_dir", default="output", help="Directory containing forecast NetCDF files")
    parser.add_argument("--truth_dir", default="../data/icosahedral-truth", help="Directory containing truth NetCDF files")

    args = parser.parse_args()

    # Match forecast files
    fcst_files = [
        os.path.join(args.fcst_dir, "aida.20260101.t12z.1p00.f000.nc"),
        os.path.join(args.fcst_dir, "aida.20260101.t12z.1p00.f006.nc"),
        os.path.join(args.fcst_dir, "aida.20260101.t12z.1p00.f012.nc"),
        os.path.join(args.fcst_dir, "aida.20260101.t12z.1p00.f018.nc"),
        os.path.join(args.fcst_dir, "aida.20260101.t12z.1p00.f024.nc"),
    ]

    # Match corresponding ground truth files
    truth_files = [
        os.path.join(args.truth_dir, "gfs.20260101.t12z.1p00.f000.nc"),  # +00h
        os.path.join(args.truth_dir, "gfs.20260101.t18z.1p00.f000.nc"),  # +06h
        os.path.join(args.truth_dir, "gfs.20260102.t00z.1p00.f000.nc"),  # +12h
        os.path.join(args.truth_dir, "gfs.20260102.t06z.1p00.f000.nc"),  # +18h
        os.path.join(args.truth_dir, "gfs.20260102.t12z.1p00.f000.nc"),  # +24h
    ]

    plot_leadtime_series(
        fcst_files=fcst_files,
        truth_files=truth_files,
        var_name=args.variable,
        level_idx=args.level,
        output_png=args.output
    )


if __name__ == "__main__":
    main()
