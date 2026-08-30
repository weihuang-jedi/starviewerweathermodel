#!/bin/bash
#SBATCH --job-name=aida_forecast
#SBATCH --account=epic
#SBATCH --gres=gpu:h100:1
#SBATCH --qos=gpuwf
#SBATCH --partition=u1-h100
##SBATCH --partition=u1-compute
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --time=03:45:00
#SBATCH --output=log.forecast_%J.out
#SBATCH --export=NONE

# SVGHOME=/scratch4/NAGAPE/epic/Wei.Huang/src/starviewerweathermodel
SVGHOME=/scratch5/purged/Wei.Huang/src/starviewerweathermodel
source ${SVGHOME}/svg.env
cd ${SVGHOME}/src

if [ -f checkpoints/aida_gnn_surrogate_logstate.pt ]; then
   n=0
   while [ -f checkpoints/aida_gnn_surrogate_logstate_${n}.pt ]; do
      n=$(( n + 1 ))
   done
   mv checkpoints/aida_gnn_surrogate_logstate.pt checkpoints/aida_gnn_surrogate_logstate_${n}.pt
fi

#gnn_chpt=checkpoints/aida_gnn_epoch_003.pt
#gnn_chpt=saved_checkpoints/aida_gnn+vert_conv_amsu_iasi_hms.pt
#gnn_chpt=checkpoints/aida_gnn_epoch_006.pt
gnn_chpt=checkpoints/aida_gnn_epoch_012.pt
#gnn_chpt=saved_checkpoints/aida_gnn_2026Aug30.pt

fcst_date=20260101
fcst_hour=12

# Add UTC to prevent timezone offset interference
base="${fcst_date:0:4}-${fcst_date:4:2}-${fcst_date:6:2} ${fcst_hour}:00 UTC"

# Subtract 6 hours
m06h_date=$(date -u -d "$base - 12 hours" "+%Y%m%d")
m06h_hour=$(date -u -d "$base - 12 hours" "+%H")

# Add 6 hours
zero_date=$(date -u -d "$base - 6 hours" "+%Y%m%d")
zero_hour=$(date -u -d "$base - 6 hours" "+%H")

output_dir="output/${fcst_date}/t${fcst_hour}z"
truth_dir="../data/icosahedral-truth"
m06h_file="${truth_dir}/icosahedral_logstate_m6.${m06h_date}.t${m06h_hour}z.0p25.f000.nc"
zero_file="${truth_dir}/icosahedral_logstate_m6.${zero_date}.t${zero_hour}z.0p25.f000.nc"

set -x

# rollout forecast
time \
python scripts/run_aida_forecast.py \
  -e ../graph/graph-grid/icosahedral_edge_index_m6.pt \
  -k ${gnn_chpt} \
  -s 20 \
  -m ${m06h_file} \
  -z ${zero_file} \
  -o ${output_dir}/fcst.0p25.f{lead:03d}.nc

