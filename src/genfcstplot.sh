#!/bin/bash

init_date=20260101
init_hour=12
fcst_length=24
# fcst_length=48
res="0p25"

output_dir="output/${init_date}/t${init_hour}z"
fcst_file=${output_dir}/fcst.0p25.f0${fcst_length}.nc

set -x

python utils/plot_icosahedral_panel.py \
  -i ${fcst_file} \
  -l 10 -s \
  -o forecast_f024_level_11.png

