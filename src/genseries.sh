#!/bin/bash

set -x

python utils/plot_forecast_leadtime_series.py \
  --fcst_dir output \
  --truth_dir ../data/icosahedral-truth \
  --out_dir plots_leadtime

