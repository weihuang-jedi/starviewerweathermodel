#!/bin/bash

python utils/plot_icosahedral_panel.py \
  -i icosahedral-grid/gfs.20240119.t06z.1p00.f000.nc \
  -l 10 \
  -o diagnostic_level_01.png \
  -s
