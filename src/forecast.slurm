#!/bin/bash
#SBATCH --job-name=aida_forecast
#SBATCH --account=epic
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --partition=u1-h100
#SBATCH --qos=gpuwf
#SBATCH --time=00:45:00
#SBATCH --output=log.forecast_%J.out
#SBATCH --export=NONE

SVGHOME=/scratch4/NAGAPE/epic/Wei.Huang/src/starviewerweathermodel
source ${SVGHOME}/svg.env
cd ${SVGHOME}/src

if [ -f checkpoints/aida_gnn_surrogate_logstate.pt ]; then
   n=0
   while [ -f checkpoints/aida_gnn_surrogate_logstate_${n}.pt ]; do
      n=$(( n + 1 ))
   done
   mv checkpoints/aida_gnn_surrogate_logstate.pt checkpoints/aida_gnn_surrogate_logstate_${n}.pt
fi

set -x

# 1. Train surrogate
time \
python scripts/run_aida_forecast.py \
  -k checkpoints/aida_gnn_surrogate_logstate.pt \
  -m icosahedral-grid/icosahedral_logstate_m4.20240730.t00z.nc \
  -z icosahedral-grid/icosahedral_logstate_m4.20240730.t06z.nc \
  -e data/graph/icosahedral_edge_index_m4.pt \
  -s 4 \
  -o forecasts/aida_24h_terrain_forecast.20240730.nc

