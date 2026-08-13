#!/bin/bash

set -x

datadir=/scratch4/NAGAPE/epic/Wei.Huang/src/starviewergraphcast/data

python interpolate_to_terrain_heights.py \
  --input ${datadir}/gfs.20240101.t00z.1p00.f000.nc \
  --etopo ${datadir}/etopo/ETOPO_2022_v1_60s_N90W180_geoid.nc \
  --output terrain-regular-grid/th_gfs.20240101.t00z.1p00.f000.nc

