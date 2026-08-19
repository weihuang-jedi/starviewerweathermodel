#!/usr/bin/env python3
"""
utils/debug_interpolation_pipeline.py

python utils/debug_interpolation_pipeline.py ../data/terrain-regular-grid/gfs.20240710.t06z.1p00.f000.nc ../graph/graph-grid/global_icosahedral_mesh_m4.nc
"""
import sys
import numpy as np
import xarray as xr

if len(sys.argv) < 3:
    print("Usage: python debug_interpolation_pipeline.py <input_regular.nc> <mesh_m4.nc>")
    sys.exit(1)

nc_file, mesh_file = sys.argv[1], sys.argv[2]
ds = xr.open_dataset(nc_file)
ds_mesh = xr.open_dataset(mesh_file)

print("=" * 80)
print(f"DIAGNOSTIC INSPECTION: '{nc_file}'")
print("=" * 80)

# Check coordinates
lat_key = 'lat' if 'lat' in ds.coords else 'latitude'
lon_key = 'lon' if 'lon' in ds.coords else 'longitude'

lats = ds[lat_key].values
lons = ds[lon_key].values

print(f"Latitude  Key: '{lat_key}' | Range: [{lats.min():.2f}, {lats.max():.2f}] | Order: {'Ascending' if lats[0] < lats[-1] else 'Descending'}")
print(f"Longitude Key: '{lon_key}' | Range: [{lons.min():.2f}, {lons.max():.2f}]")

# Check mesh bounds vs input bounds
mesh_lons = np.mod(ds_mesh['longitude'].values, 360.0)
mesh_lats = ds_mesh['latitude'].values

print(f"Mesh Longitude Range: [{mesh_lons.min():.2f}, {mesh_lons.max():.2f}]")
print(f"Mesh Latitude  Range: [{mesh_lats.min():.2f}, {mesh_lats.max():.2f}]")

# Print data variables & NaNs
print("\nData Variables & Input NaN Counts:")
for v in ds.data_vars:
    n_nan = np.isnan(ds[v].values).sum()
    print(f"  ├─ {v:20s}: {n_nan} NaNs (Shape: {ds[v].shape})")

ds.close()
ds_mesh.close()
