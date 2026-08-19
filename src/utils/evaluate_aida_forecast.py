#!/usr/bin/env python3
"""
scripts/evaluate_aida_forecast.py
---------------------------------
Evaluates AIDA GNN forecast performance against ground truth datasets.
Computes Root Mean Square Error (RMSE), Mean Error (BIAS), and Anomaly Correlation
Coefficient (ACC) across 3D vertical levels for physical fields (T, p, q, u, v, w, rho).
Supports CSV export and vertical profile summary plotting.
"""

import argparse
import os
import glob
import re
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

R_D = 287.058  # Dry air gas constant J/(kg*K)


def load_physical_fields(file_path: str) -> dict:
    """Extracts dynamic log-state fields and converts them to physical units."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"[ERROR] File not found: '{file_path}'")

    ds = xr.open_dataset(file_path)

    # 1. Temperature T (K)
    ln_t_key = 'ln_t_icosahedral' if 'ln_t_icosahedral' in ds else ('ln_t' if 'ln_t' in ds else None)
    if ln_t_key:
        t_k = np.exp(ds[ln_t_key].values)
    elif 't' in ds or 't_icosahedral' in ds:
        t_k = ds['t_icosahedral' if 't_icosahedral' in ds else 't'].values
    else:
        raise KeyError(f"Missing Temperature in {file_path}")

    # 2. Pressure p (hPa)
    ln_p_key = 'ln_p_icosahedral' if 'ln_p_icosahedral' in ds else ('ln_p' if 'ln_p' in ds else None)
    if ln_p_key:
        p_hpa = np.exp(ds[ln_p_key].values) / 100.0
    elif 'p' in ds or 'p_icosahedral' in ds:
        p_hpa = ds['p_icosahedral' if 'p_icosahedral' in ds else 'p'].values
        if np.nanmean(p_hpa) > 2000.0:
            p_hpa = p_hpa / 100.0
    else:
        raise KeyError(f"Missing Pressure in {file_path}")

    # 3. Specific Humidity q (g/kg)
    q_key = 'q_icosahedral' if 'q_icosahedral' in ds else ('q' if 'q' in ds else None)
    q_gkg = ds[q_key].values * 1000.0 if q_key else np.zeros_like(t_k)

    # 4. Zonal Wind u (m/s)
    u_key = 'u_icosahedral' if 'u_icosahedral' in ds else ('u' if 'u' in ds else None)
    u_ms = ds[u_key].values if u_key else np.zeros_like(t_k)

    # 5. Meridional Wind v (m/s)
    v_key = 'v_icosahedral' if 'v_icosahedral' in ds else ('v' if 'v' in ds else None)
    v_ms = ds[v_key].values if v_key else np.zeros_like(t_k)

    # 6. Vertical Velocity w (Pa/s)
    w_key = 'w_icosahedral' if 'w_icosahedral' in ds else ('w' if 'w' in ds else None)
    w_pas = ds[w_key].values if w_key else np.zeros_like(t_k)

    # 7. Density rho (kg/m3)
    ln_rho_key = 'ln_rho_icosahedral' if 'ln_rho_icosahedral' in ds else ('ln_rho' if 'ln_rho' in ds else None)
    if ln_rho_key:
        rho_kgm3 = np.exp(ds[ln_rho_key].values)
    else:
        rho_kgm3 = (p_hpa * 100.0) / (R_D * t_k)

    # Extract level height coordinates
    h_3d = ds['h_icosahedral'].values if 'h_icosahedral' in ds else ds['h'].values if 'h' in ds else None

    # Handle time/lead_time dimension squeezing
    fields = {'T': t_k, 'P': p_hpa, 'Q': q_gkg, 'U': u_ms, 'V': v_ms, 'W': w_pas, 'RHO': rho_kgm3}
    for k, v in fields.items():
        if v.ndim == 3:
            fields[k] = v[0]

    if h_3d is not None and h_3d.ndim == 3:
        h_3d = h_3d[0]

    ds.close()
    return fields, h_3d


def compute_metrics(forecast: np.ndarray, truth: np.ndarray):
    """
    Computes level-by-level RMSE, Bias (Mean Error), and ACC.
    Shape: [Levels=32, Nodes=2562]
    """
    num_levels = forecast.shape[0]

    rmse = np.zeros(num_levels, dtype=np.float32)
    bias = np.zeros(num_levels, dtype=np.float32)
    acc  = np.zeros(num_levels, dtype=np.float32)

    for k in range(num_levels):
        f = forecast[k]
        t = truth[k]

        diff = f - t
        bias[k] = np.mean(diff)
        rmse[k] = np.sqrt(np.mean(diff ** 2))

        # Anomaly Correlation Coefficient (ACC)
        f_ano = f - np.mean(f)
        t_ano = t - np.mean(t)
        denom = np.sqrt(np.sum(f_ano ** 2) * np.sum(t_ano ** 2))
        
        if denom > 1e-8:
            acc[k] = np.sum(f_ano * t_ano) / denom
        else:
            acc[k] = 1.0

    return rmse, bias, acc


def plot_vertical_metrics(df_metrics: pd.DataFrame, lead_time_str: str, output_png: str):
    """Generates vertical profile plot of RMSE, Bias, and ACC for T, U, V, P."""
    num_levels = df_metrics['level'].nunique()
    levels = np.arange(1, num_levels + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=True)

    target_vars = [('T', 'coolwarm'), ('U', 'Blues'), ('V', 'Greens'), ('P', 'Purples')]

    for var_name, cmap_name in target_vars:
        sub_df = df_metrics[df_metrics['variable'] == var_name]
        
        axes[0].plot(sub_df['rmse'], levels, label=f"{var_name}", linewidth=2)
        axes[1].plot(sub_df['bias'], levels, label=f"{var_name}", linewidth=2)
        axes[2].plot(sub_df['acc'],  levels, label=f"{var_name}", linewidth=2)

    axes[0].set_title("RMSE (Root Mean Square Error)", fontweight='bold')
    axes[0].set_ylabel("Model Vertical Level Index (1=Surface, 32=Top)", fontweight='bold')
    axes[0].grid(True, linestyle='--', alpha=0.6)

    axes[1].set_title("BIAS (Mean Error: Forecast - Truth)", fontweight='bold')
    axes[1].axvline(0.0, color='black', linestyle=':', alpha=0.8)
    axes[1].grid(True, linestyle='--', alpha=0.6)

    axes[2].set_title("ACC (Anomaly Correlation)", fontweight='bold')
    axes[2].set_xlim([0.0, 1.05])
    axes[2].axvline(1.0, color='black', linestyle=':', alpha=0.8)
    axes[2].grid(True, linestyle='--', alpha=0.6)

    for ax in axes:
        ax.legend(loc='best')

    fig.suptitle(f"AIDA GNN Forecast Performance Metrics vs. GFS Truth | Lead Time: +{lead_time_str}",
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_png, dpi=250, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"[SUCCESS] Metric profile plot generated: '{output_png}'")


def evaluate_pair(forecast_file: str, truth_file: str, output_csv: str = None, output_png: str = None):
    print(f"\n[EVALUATOR] Evaluating Forecast: '{forecast_file}'")
    print(f"[EVALUATOR] Against Truth File : '{truth_file}'")

    f_fields, h_3d = load_physical_fields(forecast_file)
    t_fields, _    = load_physical_fields(truth_file)

    num_levels = f_fields['T'].shape[0]
    records = []

    var_units = {
        'T': 'K', 'P': 'hPa', 'Q': 'g/kg', 'U': 'm/s', 'V': 'm/s', 'W': 'Pa/s', 'RHO': 'kg/m3'
    }

    # Match forecast lead time string (e.g. f012)
    lead_match = re.search(r'\.f(\d{3})\.nc', forecast_file)
    lead_str = f"{int(lead_match.group(1)):02d}h" if lead_match else "00h"

    print("=" * 85)
    print(f" METRIC SUMMARY TABLE | Lead Time: +{lead_str}")
    print("=" * 85)
    print(f" {'Var':5s} | {'Level':7s} | {'Mean Truth':12s} | {'RMSE':12s} | {'BIAS':12s} | {'ACC':8s}")
    print("-" * 85)

    for var_name, f_arr in f_fields.items():
        t_arr = t_fields[var_name]
        rmse, bias, acc = compute_metrics(f_arr, t_arr)

        for k in range(num_levels):
            mean_truth = float(np.mean(t_arr[k]))
            records.append({
                'lead_time': lead_str,
                'variable': var_name,
                'level': k + 1,
                'height_m': float(np.mean(h_3d[k])) if h_3d is not None else 0.0,
                'mean_truth': mean_truth,
                'rmse': float(rmse[k]),
                'bias': float(bias[k]),
                'acc': float(acc[k]),
                'unit': var_units[var_name]
            })

            if k in [0, num_levels // 2, num_levels - 1]:
                print(f" {var_name:5s} | L{k + 1:02d}    | {mean_truth:12.4f} | {rmse[k]:12.4f} | {bias[k]:12.4f} | {acc[k]:8.4f}")

    df_metrics = pd.DataFrame(records)

    if output_csv:
        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        df_metrics.to_csv(output_csv, index=False)
        print(f"[SUCCESS] CSV evaluation metrics saved to: '{output_csv}'")

    if output_png:
        os.makedirs(os.path.dirname(output_png) or ".", exist_ok=True)
        plot_vertical_metrics(df_metrics, lead_str, output_png)

    return df_metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate AIDA Forecast against Ground Truth")
    parser.add_argument("-f", "--forecast", required=True, help="Path to AIDA forecast NetCDF file (e.g. output/aida.20260101.t12z.1p00.f012.nc)")
    parser.add_argument("-t", "--truth", required=True, help="Path to Truth NetCDF file (e.g. ../data/icosahedral-truth/gfs.20260102.t00z.1p00.f000.nc)")
    parser.add_argument("-c", "--csv", help="Destination path for output CSV metrics")
    parser.add_argument("-p", "--png", help="Destination path for output PNG metric plot")

    args = parser.parse_args()

    evaluate_pair(
        forecast_file=args.forecast,
        truth_file=args.truth,
        output_csv=args.csv,
        output_png=args.png
    )


if __name__ == "__main__":
    main()
