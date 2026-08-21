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

#gnn_chpt=checkpoints/aida_gnn_conv_only_epoch_020.pt
gnn_chpt=checkpoints/aida_gnn_conv_only_epoch_035.pt

set -x

# rollout forecast
time \
python scripts/run_aida_forecast.py \
  -e ../graph/graph-grid/icosahedral_edge_index_m4.pt \
  -k ${gnn_chpt} \
  -s 20 \
  -m ../data/icosahedral-truth/gfs.20260101.t00z.1p00.f000.nc \
  -z ../data/icosahedral-truth/gfs.20260101.t06z.1p00.f000.nc \
  -o output/aida.20260101.t12z.1p00.f{lead:03d}.nc

