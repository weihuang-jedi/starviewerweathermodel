#!/bin/bash

python -c "
import xarray as xr, numpy as np
ds = xr.open_zarr('../data/icosahedral_logstate.zarr')
for v in ds.data_vars:
    n_nan = np.isnan(ds[v].values).sum()
    print(f'{v:25s}: {n_nan} NaNs')
"
