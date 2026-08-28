#!/bin/bash
#SBATCH --job-name=aida_stats
#SBATCH --account=epic
#SBATCH --partition=u1-compute
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --time=03:45:00
#SBATCH --output=log.stats_%J.out
#SBATCH --export=NONE

# SVGHOME=/scratch4/NAGAPE/epic/Wei.Huang/src/starviewerweathermodel
SVGHOME=/scratch5/purged/Wei.Huang/src/starviewerweathermodel
source ${SVGHOME}/svg.env
cd ${SVGHOME}/src

set -x

yyyymm=202607

time python utils/plot_monthly_mean_verification.py \
  --fcst_dir output-202607 \
  --truth_dir ../data/icosahedral-truth \
  --out_dir plots_monthly_mean/202607

