#!/bin/bash

set -x

python utils/plot_icosahedral_panel.py \
  -i output/aida.20260101.t12z.1p00.f024.nc \
  -l 10 -s \
  -o forecast_f024_level_11.png

