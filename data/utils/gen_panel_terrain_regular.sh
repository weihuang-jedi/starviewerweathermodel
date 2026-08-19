#!/bin/bash

python utils/plot_terrain_regular_panel.py \
  -i terrain-regular-grid/gfs.20240710.t06z.1p00.f000.nc \
  -l 10 \
  -o regular_level_01.png \
  -s
