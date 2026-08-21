#!/bin/bash
#SBATCH --job-name=pack_data
#SBATCH --partition=u1-compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --account=epic
#SBATCH --output=log.pack_data_%j.out

set -x

source /scratch4/NAGAPE/epic/Wei.Huang/src/starviewerdataassimilation/svg.env

# cd /scratch4/NAGAPE/epic/Wei.Huang/src/starviewerweathermodel/data
cd /scratch5/purged/Wei.Huang/src/starviewerweathermodel/data

python pack_icosahedral_zarr.py \
  -i "icosahedral-grid/icosahedral_logstate_m6.202*.nc" \
  -o icosahedral_logstate.zarr \
  -c 4 \
  -l 3

