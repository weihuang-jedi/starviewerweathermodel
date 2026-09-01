#!/usr/bin/env python3
"""
utils/plot_forecast_leadtime_series.py
---------------------------------------
Executes diagnostic verification across all atmospheric variables (T, P, U, V, W, Q, RHO) in one run:
1. Generates 2D Vertical Level vs. Lead Time Heatmaps for RMSE, BIAS, and ACC.
2. Generates Line Growth Curves (RMSE, BIAS, ACC) vs. Lead Time specifically for Levels 5, 15, and 25.

Usage:
  python utils/plot_forecast_leadtime_series.py --fcst_dir output/20260101/t12z --truth_dir ../data/icosahedral-truth --out_dir plots_leadtime
"""

import argparse
import os
import glob
import re
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

R_D = 287.058


def extract_variable_field_3d(nc_file: str, var_name: str) -> np.ndarray:
    """Extracts 3D physical field [Levels, Nodes] converted into physical units."""
    if not os.path.exists(nc_file):
        raise FileNotFoundError(f"[ERROR] Required NetCDF file not found: '{nc_file}'")

    ds = xr.open_dataset(nc_file)

    def get_var(candidates):
        for c in candidates:
            if c in ds:
                val = ds[c].values
                if val.ndim == 3:
                    val = val[0]
                return val
        return None

    var_upper = var_name.upper()

    if var_upper == 'T':
        val = get_var(['ln_t_icosahedral', 'ln_t', 't_icosahedral', 't'])
        field = np.exp(val) if np.nanmean(val) < 10.0 else val
    elif var_upper == 'P':
        val = get_var(['ln_p_icosahedral', 'ln_p', 'p_icosahedral', 'p'])
        field = np.exp(val) / 100.0 if np.nanmean(val) < 20.0 else val
        if np.nanmean(field) > 2000.0:
            field = field / 100.0
    elif var_upper == 'Q':
        val = get_var(['q_icosahedral', 'q'])
        field = val * 1000.0 if np.nanmean(val) < 0.05 else val  # kg/kg to g/kg
    elif var_upper == 'U':
        field = get_var(['u_icosahedral', 'u'])
    elif var_upper == 'V':
        field = get_var(['v_icosahedral', 'v'])
    elif var_upper == 'W':
        field = get_var(['w_icosahedral', 'w'])
    elif var_upper == 'RHO':
        val = get_var(['ln_rho_icosahedral', 'ln_rho', 'rho_icosahedral', 'rho'])
        if val is not None:
            field = np.exp(val) if np.nanmean(val) < 2.0 else val
        else:
            t_val = get_var(['ln_t_icosahedral', 'ln_t', 't_icosahedral', 't'])
            p_val = get_var(['ln_p_icosahedral', 'ln_p', 'p_icosahedral', 'p'])
            t_k = np.exp(t_val) if np.nanmean(t_val) < 10.0 else t_val
            p_hpa = np.exp(p_val) / 100.0 if np.nanmean(p_val) < 20.0 else p_val
            field = (p_hpa * 100.0) / (R_D * t_k)
    else:
        raise ValueError(f"[ERROR] Unsupported variable: '{var_name}'.")

    ds.close()
    return field


def compute_metrics_3d(fcst_3d: np.ndarray, truth_3d: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes RMSE, BIAS, and ACC per vertical level for 3D state tensors [Levels, Nodes]."""
    num_levels = fcst_3d.shape[0]
    rmse = np.zeros(num_levels)
    bias = np.zeros(num_levels)
    acc = np.zeros(num_levels)

    for l in range(num_levels):
        f = fcst_3d[l].astype(np.float64)
        t = truth_3d[l].astype(np.float64)

        mask = ~np.isnan(f) & ~np.isnan(t)
        if not np.any(mask):
            continue

        f_c, t_c = f[mask], t[mask]
        bias[l] = np.mean(f_c - t_c)
        rmse[l] = np.sqrt(np.mean((f_c - t_c) ** 2))

        f_anom = f_c - np.mean(f_c)
        t_anom = t_c - np.mean(t_c)
        denom = np.sqrt(np.sum(f_anom ** 2) * np.sum(t_anom ** 2)) + 1e-8

        if np.std(f_c) < 1e-6 or np.std(t_c) < 1e-6:
            acc[l] = 0.0
        else:
            acc[l] = np.sum(f_anom * t_anom) / denom

    return rmse, bias, acc


def generate_heatmap_metrics(
    var_name: str,
    leads: list[int],
    rmse_matrix: np.ndarray,
    bias_matrix: np.ndarray,
    acc_matrix: np.ndarray,
    out_dir: str,
    show: bool = False
):
    """Generates 2D Vertical Level vs Lead Time Heatmaps for RMSE, BIAS, and ACC."""
    num_levels = rmse_matrix.shape[0]
    levels = np.arange(1, num_levels + 1)
    lead_mesh, level_mesh = np.meshgrid(leads, levels)

    fig, axes = plt.subplots(1, 3, figsize=(22, 7), sharey=True)

    # 1. RMSE
    im0 = axes[0].pcolormesh(lead_mesh, level_mesh, rmse_matrix, cmap='YlOrRd', shading='auto')
    axes[0].set_title(f"RMSE ({var_name})", fontsize=13, fontweight='bold')
    axes[0].set_ylabel("Vertical Level Index (1=Surface, 32=Top)", fontsize=11)
    axes[0].set_xlabel("Forecast Lead Time (Hours)", fontsize=11)
    fig.colorbar(im0, ax=axes[0], pad=0.02)

    # 2. BIAS
    max_bias = max(abs(np.nanmin(bias_matrix)), abs(np.nanmax(bias_matrix))) or 1.0
    im1 = axes[1].pcolormesh(lead_mesh, level_mesh, bias_matrix, cmap='coolwarm', vmin=-max_bias, vmax=max_bias, shading='auto')
    axes[1].set_title(f"BIAS ({var_name})", fontsize=13, fontweight='bold')
    axes[1].set_xlabel("Forecast Lead Time (Hours)", fontsize=11)
    fig.colorbar(im1, ax=axes[1], pad=0.02)

    # 3. ACC
    im2 = axes[2].pcolormesh(lead_mesh, level_mesh, acc_matrix, cmap='viridis', vmin=0.0, vmax=1.0, shading='auto')
    axes[2].set_title(f"ACC Correlation ({var_name})", fontsize=13, fontweight='bold')
    axes[2].set_xlabel("Forecast Lead Time (Hours)", fontsize=11)
    fig.colorbar(im2, ax=axes[2], pad=0.02)

    for ax in axes:
        ax.set_xticks(leads)
        ax.grid(True, linestyle=':', alpha=0.5)

    plt.suptitle(
        f"AIDA GNN Forecast Verification Heatmap vs. Lead Time | Variable: {var_name}",
        fontsize=16, fontweight='bold', y=0.98
    )

    heatmap_png = os.path.join(out_dir, f"heatmap_{var_name}.png")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(heatmap_png, dpi=250, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()
    print(f"  ├─ Saved 2D Heatmap: '{heatmap_png}'")


def generate_level_curves(
    var_name: str,
    unit_str: str,
    leads: list[int],
    rmse_matrix: np.ndarray,
    bias_matrix: np.ndarray,
    acc_matrix: np.ndarray,
    target_levels: list[int],
    out_dir: str,
    show: bool = False
):
    """Generates Lead-Time Growth Curves for specific target levels (e.g. L5, L15, L25)."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))

    colors = ['#d95f02', '#7570b3', '#1b9e77', '#e7298a']
    markers = ['o', 's', '^', 'D']

    for idx, lvl in enumerate(target_levels):
        lvl_idx = lvl - 1  # 0-based indexing
        c = colors[idx % len(colors)]
        m = markers[idx % len(markers)]
        label_str = f"Level {lvl:02d}"

        # 1. RMSE Growth
        axes[0].plot(leads, rmse_matrix[lvl_idx, :], marker=m, color=c, linewidth=2, label=label_str)

        # 2. BIAS Drift
        axes[1].plot(leads, bias_matrix[lvl_idx, :], marker=m, color=c, linewidth=2, label=label_str)

        # 3. ACC Decay
        axes[2].plot(leads, acc_matrix[lvl_idx, :], marker=m, color=c, linewidth=2, label=label_str)

    axes[0].set_title(f"RMSE Growth Curve ({unit_str})", fontsize=12, fontweight='bold')
    axes[0].set_ylabel(f"RMSE [{unit_str}]", fontsize=11)
    axes[0].set_xlabel("Forecast Lead Time (Hours)", fontsize=11)
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend(loc='upper left', fontsize=10)

    axes[1].set_title(f"BIAS Drift Curve ({unit_str})", fontsize=12, fontweight='bold')
    axes[1].set_ylabel(f"BIAS [{unit_str}]", fontsize=11)
    axes[1].set_xlabel("Forecast Lead Time (Hours)", fontsize=11)
    axes[1].axhline(0, color='black', linestyle=':', linewidth=1)
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend(loc='best', fontsize=10)

    axes[2].set_title("ACC Correlation Decay Curve", fontsize=12, fontweight='bold')
    axes[2].set_ylabel("ACC Score", fontsize=11)
    axes[2].set_xlabel("Forecast Lead Time (Hours)", fontsize=11)
    axes[2].set_ylim([-0.05, 1.05])
    axes[2].axhline(0.6, color='red', linestyle='--', linewidth=1, label="ACC = 0.6 Threshold")
    axes[2].grid(True, linestyle='--', alpha=0.6)
    axes[2].legend(loc='lower left', fontsize=10)

    for ax in axes:
        ax.set_xticks(leads)

    plt.suptitle(
        f"AIDA GNN Lead-Time Performance Curves for Levels {target_levels} | Variable: {var_name}",
        fontsize=15, fontweight='bold', y=0.98
    )

    curves_png = os.path.join(out_dir, f"curves_{var_name}_L5_15_25.png")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(curves_png, dpi=250, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()
    print(f"  ├─ Saved Level Curves: '{curves_png}'")


def resolve_file_pairs(fcst_dir: str, truth_dir: str) -> tuple[list[str], list[str], list[int]]:
    """
    Pairs forecast files with ground truth files.
    Supports directory structures like:
    - output/20260101/t12z/fcst.0p25.f024.nc
    - output/aida.20260101.t12z.0p25.f024.nc
    """
    pattern = os.path.join(fcst_dir, "**", "*.nc")
    found_fcst = sorted(glob.glob(pattern, recursive=True))

    if not found_fcst:
        pattern_flat = os.path.join(fcst_dir, "*.nc")
        found_fcst = sorted(glob.glob(pattern_flat))

    if not found_fcst:
        raise FileNotFoundError(f"[ERROR] No NetCDF forecast files found in '{fcst_dir}'")

    fcst_files, truth_files, leads = [], [], []

    for f_path in found_fcst:
        base_name = os.path.basename(f_path)
        
        # Regex Option A: Legacy filename containing date and cycle (e.g. aida.20260101.t12z.0p25.f024.nc)
        match_legacy = re.search(r'(\d{8})\.t(\d{2})z.*?f(\d{3})\.nc', base_name)
        
        # Regex Option B: Subdirectory structure (e.g. .../20260101/t12z/fcst.0p25.f024.nc)
        match_sub = re.search(r'fcst.*?f(\d{3})\.nc', base_name)
        
        date_str = None
        cycle_str = None
        lead_hr = None

        if match_legacy:
            date_str = match_legacy.group(1)
            cycle_str = match_legacy.group(2)
            lead_hr = int(match_legacy.group(3))
        elif match_sub:
            lead_hr = int(match_sub.group(1))
            
            # Extract date and cycle from parent path components
            dir_parts = os.path.normpath(f_path).split(os.sep)
            for part in dir_parts:
                if re.match(r'^\d{8}$', part):
                    date_str = part
                elif re.match(r'^t\d{2}z$', part):
                    cycle_str = part.replace('t', '').replace('z', '')

        if date_str is None or cycle_str is None or lead_hr is None:
            continue

        # Compute ground truth file target timestamp
        dt_base = datetime.strptime(f"{date_str}{cycle_str}", "%Y%m%d%H")
        dt_truth = dt_base + timedelta(hours=lead_hr)

        truth_tag = dt_truth.strftime("%Y%m%d.t%Hz")
        truth_file = os.path.join(truth_dir, f"icosahedral_logstate_m6.{truth_tag}.0p25.f000.nc")

        if os.path.exists(truth_file):
            fcst_files.append(f_path)
            truth_files.append(truth_file)
            leads.append(lead_hr)
        else:
            print(f"[WARNING] Truth file not found for forecast '{base_name}': expected '{truth_file}'")

    # Sort files by lead time
    if fcst_files:
        sorted_pairs = sorted(zip(leads, fcst_files, truth_files), key=lambda x: x[0])
        leads, fcst_files, truth_files = zip(*sorted_pairs)
        leads, fcst_files, truth_files = list(leads), list(fcst_files), list(truth_files)

    return fcst_files, truth_files, leads


def main():
    parser = argparse.ArgumentParser(description="Batch Lead-Time Diagnostic Generator for All State Variables")
    parser.add_argument("--fcst_dir", default="output", help="Directory containing forecast NetCDF files")
    parser.add_argument("--truth_dir", default="../data/icosahedral-truth", help="Directory containing truth NetCDF files")
    parser.add_argument("--out_dir", default="plots_leadtime", help="Destination directory for output plots")
    parser.add_argument("-s", "--show", action="store_true", help="Display plot interactively")

    args = parser.parse_args()

    fcst_files, truth_files, leads = resolve_file_pairs(args.fcst_dir, args.truth_dir)

    if not fcst_files:
        raise RuntimeError(f"[ERROR] No valid forecast-truth file pairs resolved from fcst_dir='{args.fcst_dir}' and truth_dir='{args.truth_dir}'")

    os.makedirs(args.out_dir, exist_ok=True)

    variables_dict = {
        'T': 'K',
        'P': 'hPa',
        'U': 'm/s',
        'V': 'm/s',
        'W': 'Pa/s',
        'Q': 'g/kg',
        'RHO': 'kg/m³'
    }

    target_levels = [5, 15, 25]  # Target levels requested

    print(f"\n" + "=" * 80)
    print(f" BATCH DIAGNOSTIC EVALUATION ACROSS ALL VARIABLES")
    print(f" Forecast Files Directory : '{args.fcst_dir}'")
    print(f" Ground Truth Directory   : '{args.truth_dir}'")
    print(f" Resolved Lead Times      : {leads} (Hours)")
    print(f" Target Level Curves      : {target_levels} (1=Surface, 32=Top)")
    print(f" Output Plots Directory   : '{args.out_dir}'")
    print("=" * 80 + "\n")

    for var_name, unit_str in variables_dict.items():
        print(f"[PROCESSING] Variable: {var_name} [{unit_str}]...")

        sample_f = extract_variable_field_3d(fcst_files[0], var_name)
        num_levels = sample_f.shape[0]
        num_leads = len(leads)

        rmse_matrix = np.zeros((num_levels, num_leads))
        bias_matrix = np.zeros((num_levels, num_leads))
        acc_matrix = np.zeros((num_levels, num_leads))

        for idx, (f_file, t_file) in enumerate(zip(fcst_files, truth_files)):
            f_3d = extract_variable_field_3d(f_file, var_name)
            t_3d = extract_variable_field_3d(t_file, var_name)

            r, b, a = compute_metrics_3d(f_3d, t_3d)
            rmse_matrix[:, idx] = r
            bias_matrix[:, idx] = b
            acc_matrix[:, idx] = a

        # 1. Plot 2D Level vs. Lead Time Heatmaps
        generate_heatmap_metrics(var_name, leads, rmse_matrix, bias_matrix, acc_matrix, args.out_dir, args.show)

        # 2. Plot RMSE, BIAS, and ACC curves for Levels 5, 15, 25
        generate_level_curves(var_name, unit_str, leads, rmse_matrix, bias_matrix, acc_matrix, target_levels, args.out_dir, args.show)

        print()

    print(f"[SUCCESS] All diagnostic plots generated and saved to '{args.out_dir}/'!\n")


if __name__ == "__main__":
    main()
