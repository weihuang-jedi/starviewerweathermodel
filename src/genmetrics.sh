#!/bin/bash

set -x

init_date=20260101
init_hour=12
# fcst_length=24
fcst_length=48
res="0p25"

# Add UTC to prevent timezone offset interference
base="${init_date:0:4}-${init_date:4:2}-${init_date:6:2} ${init_hour}:00 UTC"

# Add 6 hours
fcst_date=$(date -u -d "$base + $fcst_length hours" "+%Y%m%d")
fcst_hour=$(date -u -d "$base + $fcst_length hours" "+%H")

output_dir="output/${init_date}/t${init_hour}z"
fcst_file=${output_dir}/fcst.0p25.f0${fcst_length}.nc
truth_file=../data/icosahedral-truth/icosahedral_logstate_m6.${fcst_date}.t${fcst_hour}z.0p25.f000.nc

if [ ! -f $fcst_file ]
then
    fcst_file=output/aida.${init_date}.t${init_hour}z.${res}.f0${fcst_length}.nc
fi

python utils/evaluate_aida_forecast.py \
  -f ${fcst_file} \
  -t ${truth_file} \
  -c metrics_f0${fcst_length}.csv \
  -p metrics_f0${fcst_length}_profile.png

