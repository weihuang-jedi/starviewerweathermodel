#!/bin/bash

set -x

# python -W ignore::UserWarning utils/plot_forecast_panel_comparison.py \

python utils/plot_forecast_panel_comparison.py \
  -f output/aida.20260101.t12z.0p25.f024.nc \
  -t ../data/icosahedral-truth/icosahedral_logstate_m6.20260102.t12z.0p25.f000.nc \
  -z ../data/icosahedral-truth/icosahedral_logstate_m6.20260101.t12z.0p25.f000.nc \
  -l 10 \
  -o forecast_surface_comparison_panel.png

