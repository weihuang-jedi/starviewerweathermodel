#!/bin/bash

set -x

python utils/evaluate_aida_forecast.py \
  -f output/aida.20260101.t12z.1p00.f024.nc \
  -t ../data/icosahedral-truth/gfs.20260102.t12z.1p00.f000.nc \
  -c metrics_f012.csv \
  -p metrics_f012_profile.png

