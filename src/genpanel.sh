#!/bin/bash

set -x

python utils/plot_forecast_panel_comparison.py \
  -f output/aida.20260101.t12z.1p00.f024.nc \
  -t ../data/icosahedral-truth/gfs.20260102.t12z.1p00.f000.nc \
  -z ../data/icosahedral-truth/gfs.20260101.t12z.1p00.f000.nc \
  -l 10 \
  -o forecast_surface_comparison_panel.png

