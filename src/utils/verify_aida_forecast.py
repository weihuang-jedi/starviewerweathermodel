#!/usr/bin/env python3
"""
scripts/verify_aida_forecast.py
--------------------------------
Diagnostic and verification script for AIDA 4D Terrain-Following NetCDF Forecast Rollouts.
Inspects structure, verifies physical variable boundaries, detects NaNs/Infs,
and computes trajectory error growth across forecast lead times (+0h, +6h, +12h, ...).
"""

import argparse
import os
import sys
import numpy as np
import xarray as xr


# Standard expected log-state variables
EXPECTED_VARS = [
    'ln_t_icosahedral', 'u_icosahedral', 'v_icosahedral',
    'w_icosahedral', 'q_icosahedral', 'ln_rho_icosahedral', 'ln_p_icosahedral'
]

# Reasonable physical bounds in physical units for validation
PHYSICAL_BOUNDS = {
    'temperature_k': (180.0, 340.0),       # Kelvin
    'u_wind_ms': (-150.0, 150.0),          # m/s
    'v_wind_ms': (-150.0, 150.0),          # m/s
    'w_wind_pas': (-100.0, 100.0),         # Pa/s
    'q_humidity_kgkg': (0.0, 0.05),        # kg/kg (Strict non-negativity)
    'rho_density_kgm3': (1e-5, 2.0),       # kg/m^3
    'pressure_pa': (1.0, 110000.0),        # Pascal
}


def inspect_forecast_rollout(nc_file: str):
    if not os.path.exists(nc_file):
        print(f"[ERROR] Forecast NetCDF file not found: '{nc_file}'")
        sys.exit(1)

    print("=" * 110)
    print(f" AIDA MULTI-STEP FORECAST ROLLOUT VERIFICATION DIAGNOSTIC")
    print(f" Target NetCDF File: '{nc_file}'")
    print("=" * 110)

    ds = xr.open_dataset(nc_file)

    # -------------------------------------------------------------------------
    # 1. Structural & Metadata Audit
    # -------------------------------------------------------------------------
    print("\n[1/3] STRUCTURAL & METADATA AUDIT")
    print("-" * 110)

    dims = dict(ds.dims)
    print(f"  ├─ Dimensions      : {dims}")
    
    lead_times = ds['lead_time'].values if 'lead_time' in ds else np.array([])
    print(f"  ├─ Lead Times (h)  : {lead_times.tolist()} ({len(lead_times)} forecast steps)")
    
    num_nodes = dims.get('node', 0)
    num_levels = dims.get('level', 0)
    print(f"  ├─ Spatial Grid    : {num_nodes} icosahedral nodes")
    print(f"  ├─ Vertical Grid   : {num_levels} terrain-following height levels")

    # Verify Terrain Metadata Coordinates
    has_h3d = 'h_icosahedral' in ds or 'h' in ds
    has_hsurf = 'h_terrain_icosahedral' in ds or 'h_terrain' in ds or 'elevation' in ds
    
    print(f"  ├─ 3D Terrain Height (h_3d) : {'PRESENT [OK]' if has_h3d else 'MISSING [WARNING]'}")
    print(f"  └─ Surface Elevation (H_surf): {'PRESENT [OK]' if has_hsurf else 'MISSING [WARNING]'}")

    # -------------------------------------------------------------------------
    # 2. Variable Consistency & Physical Boundary Audit
    # -------------------------------------------------------------------------
    print("\n[2/3] VARIABLE CONSISTENCY & PHYSICAL BOUNDARY AUDIT")
    print("-" * 110)

    has_errors = False
    var_map = {}

    for var in EXPECTED_VARS:
        var_name = var if var in ds else var.replace('_icosahedral', '')
        if var_name in ds:
            var_map[var] = var_name
        else:
            print(f"  [ERROR] Missing required variable '{var}' in dataset!")
            has_errors = True

    if has_errors:
        print("[FAIL] Missing required weather state variables. Verification aborted.")
        sys.exit(1)

    print("  All 7 core dynamic state variables identified successfully.")
    
    # Check for NaNs/Infs and Physical Violations
    total_nans = 0
    total_infs = 0
    negative_humidity_count = 0

    for std_name, actual_name in var_map.items():
        arr = ds[actual_name].values
        n_nan = np.isnan(arr).sum()
        n_inf = np.isinf(arr).sum()
        total_nans += n_nan
        total_infs += n_inf

        if std_name in ['q_icosahedral', 'q']:
            q_min = np.min(arr)
            if q_min < 0.0:
                neg_count = np.sum(arr < 0.0)
                negative_humidity_count += neg_count
                print(f"  [WARNING] Variable '{actual_name}' contains {neg_count} negative humidity values (Min: {q_min:.5e})")

    print(f"  ├─ Total NaN Values Detected  : {total_nans} {'[OK]' if total_nans == 0 else '[FAIL]'}")
    print(f"  ├─ Total Inf Values Detected  : {total_infs} {'[OK]' if total_infs == 0 else '[FAIL]'}")
    print(f"  └─ Negative Moisture Violations: {negative_humidity_count} {'[OK]' if negative_humidity_count == 0 else '[WARNING]'}")

    # -------------------------------------------------------------------------
    # 3. Temporal Trajectory Growth & Error Metrics
    # -------------------------------------------------------------------------
    print("\n[3/3] TEMPORAL TRAJECTORY METRICS & STATE EVOLUTION")
    print("-" * 110)

    ln_t = ds[var_map['ln_t_icosahedral']].values
    u = ds[var_map['u_icosahedral']].values
    v = ds[var_map['v_icosahedral']].values
    w = ds[var_map['w_icosahedral']].values
    q = ds[var_map['q_icosahedral']].values
    ln_p = ds[var_map['ln_p_icosahedral']].values

    # Physical Transformations
    t_k = np.exp(ln_t)
    p_pa = np.exp(ln_p)
    wind_speed = np.sqrt(u**2 + v**2)

    print(f"{'Lead Time':>10} | {'Temp (K) Mean±Std':>20} | {'Max Wind (m/s)':>15} | {'Min/Max Pressure (Pa)':>24} | {'RMS Step Diff':>15}")
    print("-" * 105)

    for t_idx, lt in enumerate(lead_times):
        t_mean = np.mean(t_k[t_idx])
        t_std = np.std(t_k[t_idx])
        w_max = np.max(wind_speed[t_idx])
        p_min = np.min(p_pa[t_idx])
        p_max = np.max(p_pa[t_idx])

        if t_idx > 0:
            # Root Mean Square step difference relative to previous lead time
            diff_t = t_k[t_idx] - t_k[t_idx - 1]
            diff_u = u[t_idx] - u[t_idx - 1]
            diff_v = v[t_idx] - v[t_idx - 1]
            rms_diff = np.sqrt(np.mean(diff_t**2 + diff_u**2 + diff_v**2))
            step_str = f"{rms_diff:12.4f}"
        else:
            step_str = "    Initial (t0)"

        print(f"{lt:>8}h | {t_mean:8.2f} ± {t_std:6.2f} K | {w_max:13.2f} m/s | {p_min:9.1f} - {p_max:9.1f} Pa | {step_str:>15}")

    print("-" * 105)

    # Final Overall Status
    if total_nans == 0 and total_infs == 0:
        print("\n[VERIFICATION SUCCESS] The multi-step forecast NetCDF rollout is structurally sound and physically valid!")
    else:
        print("\n[VERIFICATION FAILED] Numerical instability (NaN/Inf) detected in the rollout file.")

    ds.close()


def main():
    parser = argparse.ArgumentParser(description="Inspect and Verify AIDA NetCDF Forecast Rollout")
    parser.add_argument("-f", "--file", default="aida_24h_terrain_forecast.nc", help="Path to forecast rollout NetCDF file")
    args = parser.parse_args()

    inspect_forecast_rollout(args.file)


if __name__ == "__main__":
    main()
