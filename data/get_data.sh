#!/bin/bash

set -x

datadir=/scratch4/NAGAPE/epic/Wei.Huang/src/starviewergraphcast/data

#python interpolate_to_terrain_heights.py \
#  --input tmp_20240101_gfs.t00z.pgrb2b.1p00.f000 \
#  --etopo ${datadir}/etopo/ETOPO_2022_v1_60s_N90W180_geoid.nc \
#  --output terrain-regular-grid/gfs.20240101.t00z.1p00.f000.nc

python interpolate_to_logstate_icosahedral.py \
   -i terrain-regular-grid/gfs.20240101.t00z.1p00.f000.nc \
   -m ../graph/graph-grid/global_icosahedral_mesh_m4.nc \
   -o icosahedral-grid/icosahedral_logstate.20240101.t00z.nc

