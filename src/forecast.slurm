#!/bin/bash
#SBATCH --job-name=aida_forecast
#SBATCH --account=epic
#SBATCH --gres=gpu:h100:1
#SBATCH --qos=gpuwf
#SBATCH --partition=u1-h100
##SBATCH --partition=u1-compute
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --time=00:05:00
#SBATCH --output=log.forecast_%J.out
#SBATCH --export=ALL

# SVGHOME=/scratch4/NAGAPE/epic/Wei.Huang/src/starviewerweathermodel
SVGHOME=/scratch5/purged/Wei.Huang/src/starviewerweathermodel
source ${SVGHOME}/svg.env
cd ${SVGHOME}/src

if [ -f checkpoints/aida_gnn.pt ]; then
   n=0
   while [ -f checkpoints/aida_gnn${n}.pt ]; do
      n=$(( n + 1 ))
   done
   mv checkpoints/aida_gnn.pt checkpoints/aida_gnn_${n}.pt
fi

gnn_chpt=checkpoints/aida_gnn_epoch_025.pt

init_date=20260101
init_hour=12
rollout_steps=20
res="0p25"

# Add UTC to prevent timezone offset interference
base="${init_date:0:4}-${init_date:4:2}-${init_date:6:2} ${init_hour}:00 UTC"

# Subtract 6 hours
m06h_date=$(date -u -d "$base - 12 hours" "+%Y%m%d")
m06h_hour=$(date -u -d "$base - 12 hours" "+%H")

# Add 6 hours
zero_date=$(date -u -d "$base - 6 hours" "+%Y%m%d")
zero_hour=$(date -u -d "$base - 6 hours" "+%H")

output_dir="output/${init_date}/t${init_hour}z"
truth_dir="../data/icosahedral-truth"
m06h_file="${truth_dir}/icosahedral_logstate_m6.${m06h_date}.t${m06h_hour}z.${res}.f000.nc"
zero_file="${truth_dir}/icosahedral_logstate_m6.${zero_date}.t${zero_hour}z.${res}.f000.nc"

set -x

# rollout forecast
time \
python scripts/run_aida_forecast.py \
  -e ../graph/graph-grid/icosahedral_edge_index_m6.pt \
  -k ${gnn_chpt} \
  -s ${rollout_steps} \
  -m ${m06h_file} \
  -z ${zero_file} \
  -o ${output_dir}/fcst.${res}.f{lead:03d}.nc

