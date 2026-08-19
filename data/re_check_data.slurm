#!/bin/bash
#SBATCH --job-name=re_check_data
#SBATCH --partition=u1-compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --account=epic
#SBATCH --output=log.re_check_%j.out

set -x

# Activate your custom conda environment
EAGLEhome=/scratch5/purged/Wei.Huang/src/EAGLE
source ${EAGLEhome}/conda/etc/profile.d/conda.sh
eval "$(mamba shell hook --shell bash)"
mamba activate anemoi

cd /scratch4/NAGAPE/epic/Wei.Huang/src/starviewerweathermodel/data

rm -rf icosahedral-grid

# 1. Re-run batch interpolation over all regular terrain-following NetCDF files
python interpolate_to_logstate_icosahedral.py \
  --input_dir terrain-regular-grid \
  --output_dir icosahedral-grid \
  -m ../graph/graph-grid/global_icosahedral_mesh_m4.nc \
  --pattern "*.nc"

# 2. Remove old corrupt Zarr directory
rm -rf icosahedral_logstate.zarr

# 3. Pack clean files into Zarr store
python pack_icosahedral_zarr.py \
  -i "icosahedral-grid/icosahedral_logstate_m4.202*.nc" \
  -o icosahedral_logstate.zarr \
  -c 32 \
  -l 3

# 4. Verify NaN count across Zarr variables (must all print 0)
python -c "
import xarray as xr, numpy as np
ds = xr.open_zarr('../data/icosahedral_logstate.zarr')
for v in ds.data_vars:
    n_nan = np.isnan(ds[v].values).sum()
    print(f'{v:25s}: {n_nan} NaNs')
"
