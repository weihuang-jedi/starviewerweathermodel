#!/usr/bin/env python3
"""
utils/evaluate_aida_forecast.py
-------------------------------
Evaluates AIDA forecast NetCDF files against ground truth files.
Computes RMSE, BIAS, Mean Truth, and Anomaly Correlation Coefficient (ACC)
across levels and physical state variables.
"""

import argparse
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt


def to_physical_units(var_name: str, val: np.ndarray) -> np.ndarray:
    """
    Safely converts both normalized GNN model log-states and natural log truth fields back to physical units:
    - Temperature: Kelvin (K)
    - Pressure: hectopascals (hPa)
    - Density: kg/m3
    """
    val = np.nan_to_num(val, nan=0.0)
    clean_var = var_name.replace('_icosahedral', '').lower()

    # 1. Temperature (T or log_T / ln_t)
    if 't' in clean_var:
        mean_v = np.nanmean(val)
        if -5.0 <= mean_v <= 5.0:  # Model normalized log-state space: (ln_T - 5.50) / 0.15
            ln_t = val * 0.15 + 5.50
            return np.clip(np.exp(ln_t), 150.0, 350.0)
        elif 5.0 < mean_v < 7.0:   # Natural log space: ln(T)
            return np.clip(np.exp(val), 150.0, 350.0)
        else:                      # Raw physical space: T (K)
            return np.clip(val, 150.0, 350.0)

    # 2. Pressure (P or log_P / ln_p)
    elif 'p' in clean_var:
        mean_v = np.nanmean(val)
        if -5.0 <= mean_v <= 5.0:  # Model normalized log-state space: (ln_P - 10.50) / 1.20
            ln_p = val * 1.20 + 10.50
            p_pa = np.exp(ln_p)
            return np.clip(p_pa / 100.0, 0.01, 1100.0)  # Convert Pa to hPa
        elif 4.0 < mean_v < 13.0:  # Natural log space: ln(P_Pa)
            return np.clip(np.exp(val) / 100.0, 0.01, 1100.0)
        elif mean_v > 10000.0:     # Raw Pa
            return np.clip(val / 100.0, 0.01, 1100.0)
        else:                      # Raw hPa
            return np.clip(val, 0.01, 1100.0)

    # 3. Density (rho or log_rho / ln_rho)
    elif 'rho' in clean_var:
        mean_v = np.nanmean(val)
        if -5.0 <= mean_v <= 5.0:  # Model normalized log-state space: (ln_rho + 0.20) / 0.80
            ln_rho = val * 0.80 - 0.20
            return np.clip(np.exp(ln_rho), 1e-5, 3.0)
        elif mean_v < 3.0:         # Natural log space
            return np.clip(np.exp(val), 1e-5, 3.0)
        else:                      # Raw physical space
            return np.clip(val, 1e-5, 3.0)

    return val

def evaluate_forecast(fcst_file: str, truth_file: str, csv_out: str, plot_out: str):
    if not os.path.exists(fcst_file):
        raise FileNotFoundError(f"[ERROR] Forecast file not found: '{fcst_file}'")
    if not os.path.exists(truth_file):
        raise FileNotFoundError(f"[ERROR] Truth file not found: '{truth_file}'")

    print(f"[EVALUATOR] Evaluating Forecast: '{fcst_file}'")
    print(f"[EVALUATOR] Against Truth File : '{truth_file}'")

    ds_fcst = xr.open_dataset(fcst_file)
    ds_truth = xr.open_dataset(truth_file)

    var_mapping = {
        'T': ['ln_t_icosahedral', 't_icosahedral', 't', 'ln_t'],
        'P': ['ln_p_icosahedral', 'p_icosahedral', 'p', 'ln_p'],
        'Q': ['q_icosahedral', 'q'],
        'U': ['u_icosahedral', 'u'],
        'V': ['v_icosahedral', 'v'],
        'W': ['w_icosahedral', 'w'],
        'RHO': ['ln_rho_icosahedral', 'rho_icosahedral', 'rho', 'ln_rho']
    }

    metrics_list = []
    num_levels = ds_fcst.sizes.get('level', 32)

    for var_std, keys in var_mapping.items():
        key_fcst = next((k for k in keys if k in ds_fcst), None)
        key_truth = next((k for k in keys if k in ds_truth), None)

        if key_fcst is None or key_truth is None:
            continue

        arr_fcst = ds_fcst[key_fcst].values
        arr_truth = ds_truth[key_truth].values

        if arr_fcst.ndim == 3:
            arr_fcst = arr_fcst[0]
        if arr_truth.ndim == 3:
            arr_truth = arr_truth[0]

        # Convert both forecast and truth safely into physical space
        arr_fcst = to_physical_units(key_fcst, arr_fcst)
        arr_truth = to_physical_units(key_truth, arr_truth)

        for l_idx in range(num_levels):
            f_layer = arr_fcst[l_idx].astype(np.float64)
            t_layer = arr_truth[l_idx].astype(np.float64)

            mask = ~np.isnan(f_layer) & ~np.isnan(t_layer)
            if not np.any(mask):
                continue

            f_clean = f_layer[mask]
            t_clean = t_layer[mask]

            mean_truth = np.mean(t_clean)
            bias = np.mean(f_clean - t_clean)
            rmse = np.sqrt(np.mean((f_clean - t_clean) ** 2))

            # Anomaly Correlation Coefficient (ACC)
            f_anom = f_clean - np.mean(f_clean)
            t_anom = t_clean - mean_truth
            denom = np.sqrt(np.sum(f_anom ** 2) * np.sum(t_anom ** 2)) + 1e-8

            if np.std(f_clean) < 1e-6 or np.std(t_clean) < 1e-6:
                acc = 0.0
            else:
                acc = np.sum(f_anom * t_anom) / denom

            metrics_list.append({
                'Variable': var_std,
                'Level': l_idx + 1,
                'Mean_Truth': mean_truth,
                'RMSE': rmse,
                'BIAS': bias,
                'ACC': acc
            })

    df = pd.DataFrame(metrics_list)

    if csv_out:
        os.makedirs(os.path.dirname(csv_out) or ".", exist_ok=True)
        df.to_csv(csv_out, index=False)
        print(f"[SUCCESS] CSV evaluation metrics saved to: '{csv_out}'")

    # Print Summary Table for Levels 1, 17, 32
    print("=" * 85)
    print(" METRIC SUMMARY TABLE | Lead Time Evaluation")
    print("=" * 85)
    print(f" {'Var':<5} | {'Level':<8} | {'Mean Truth':>12} | {'RMSE':>12} | {'BIAS':>12} | {'ACC':>8}")
    print("-" * 85)

    for v in ['T', 'P', 'Q', 'U', 'V', 'W', 'RHO']:
        sub = df[df['Variable'] == v]
        if sub.empty:
            continue
        for lvl in [1, 17, num_levels]:
            row = sub[sub['Level'] == lvl]
            if not row.empty:
                r = row.iloc[0]
                print(f" {r['Variable']:<5} | L{r['Level']:02d}     | {r['Mean_Truth']:12.4f} | {r['RMSE']:12.4f} | {r['BIAS']:12.4f} | {r['ACC']:8.4f}")

    print("=" * 85)

    ds_fcst.close()
    ds_truth.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate AIDA Forecast NetCDF against Ground Truth")
    parser.add_argument("-f", "--forecast", required=True, help="Forecast NetCDF file")
    parser.add_argument("-t", "--truth", required=True, help="Ground Truth NetCDF file")
    parser.add_argument("-c", "--csv", default="metrics.csv", help="Output CSV file")
    parser.add_argument("-p", "--plot", default="metrics_profile.png", help="Output profile plot file")

    args = parser.parse_args()
    evaluate_forecast(args.forecast, args.truth, args.csv, args.plot)


if __name__ == "__main__":
    main()
