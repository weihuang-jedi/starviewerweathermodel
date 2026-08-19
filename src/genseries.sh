#!/bin/bash

set -x

lvl=10
for var in T P U V
do
  python utils/plot_forecast_leadtime_series.py \
    -v ${var} \
    -l ${lvl} \
    -o leadtime_series_${var}_L${lvl}.png
done

