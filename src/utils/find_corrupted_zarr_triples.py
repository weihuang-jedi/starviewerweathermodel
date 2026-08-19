#!/usr/bin/env python3
# Save as scripts/find_corrupted_zarr_triples.py

import xarray as xr
import numpy as np

zarr_path = '../data/icosahedral_logstate.zarr'
print(f"Scanning Zarr store at: {zarr_path}")
ds = xr.open_zarr(zarr_path)

vars_to_check = [
    'ln_t_icosahedral', 'u_icosahedral', 'v_icosahedral',
    'w_icosahedral', 'q_icosahedral', 'ln_rho_icosahedral', 'ln_p_icosahedral'
]

num_times = len(ds['time'])
corrupted_indices = []

for t_idx in range(1, num_times - 1):
    # Check triple (t-1, t, t+1)
    slice_data = np.stack([ds[v].isel(time=slice(t_idx - 1, t_idx + 2)).values for v in vars_to_check])
    
    n_nan = np.isnan(slice_data).sum()
    n_inf = np.isinf(slice_data).sum()
    
    if n_nan > 0 or n_inf > 0:
        corrupted_indices.append(t_idx)
        print(f"  ❌ Trajectory Triple Index {t_idx:4d} (Time: {str(ds['time'].values[t_idx])[:19]}): {n_nan} NaNs, {n_inf} Infs")

print("-" * 70)
print(f"Scan Complete: {len(corrupted_indices)} / {num_times - 2} trajectory triples are corrupted.")
ds.close()
