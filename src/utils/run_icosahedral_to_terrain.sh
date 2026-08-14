#!/bin/bash

set -x

python utils/interpolate_icosahedral_to_terrain.py \
  -i ../data/icosahedral-grid/gfs.20240316.t06z.1p00.f000.nc \
  -r ../data/terrain-regular-grid/gfs.20240710.t06z.1p00.f000.nc \
  -o output/gfs.20240316.t06z.1p00.f000.nc

