#!/bin/bash
#SBATCH --job-name=aida_forecast
#SBATCH --account=epic
##SBATCH --gres=gpu:h100:1
##SBATCH --qos=gpuwf
##SBATCH --partition=u1-h100
#SBATCH --partition=u1-compute
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
gnn_chpt=checkpoints/aida_gnn.pt

set -x

# rollout forecast
time \
python scripts/run_aida_forecast.py \
  -e ../graph/graph-grid/icosahedral_edge_index_m6.pt \
  -k ${gnn_chpt} \
  -s 20 \
  -m ../data/icosahedral-truth/icosahedral_logstate_m6.20260101.t00z.0p25.f000.nc \
  -z ../data/icosahedral-truth/icosahedral_logstate_m6.20260101.t06z.0p25.f000.nc \
  -o output/aida.20260101.t12z.0p25.f{lead:03d}.nc

