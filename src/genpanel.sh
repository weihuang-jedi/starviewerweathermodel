#!/bin/bash

set -x

init_date=20260101
init_hour=12
fcst_length=24
# fcst_length=48
res="0p25"

# Add UTC to prevent timezone offset interference
base="${init_date:0:4}-${init_date:4:2}-${init_date:6:2} ${init_hour}:00 UTC"

# Add 6 hours
fcst_date=$(date -u -d "$base + $fcst_length hours" "+%Y%m%d")
fcst_hour=$(date -u -d "$base + $fcst_length hours" "+%H")

truth_dir="../data/icosahedral-truth"
output_dir="output/${init_date}/t${init_hour}z"

fcst_file=${output_dir}/fcst.${res}.f0${fcst_length}.nc
base_file=${truth_dir}/icosahedral_logstate_m6.${init_date}.t${init_hour}z.${res}.f000.nc
truth_file=${truth_dir}/icosahedral_logstate_m6.${fcst_date}.t${fcst_hour}z.${res}.f000.nc

python utils/plot_forecast_panel_comparison.py \
  -f ${fcst_file} \
  -t ${truth_file} \
  -z ${base_file} \
  -l 10 \
  -o forecast_surface_comparison_panel.png

