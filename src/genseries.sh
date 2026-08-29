#!/bin/bash

set -x

yyyymmdd=20260101
fcsthour=12

python utils/plot_forecast_leadtime_series.py \
  --fcst_dir output/${yyyymmdd}/t${fcsthour}z \
  --truth_dir ../data/icosahedral-truth \
  --out_dir plots_leadtime/${yyyymmdd}/t${fcsthour}z

