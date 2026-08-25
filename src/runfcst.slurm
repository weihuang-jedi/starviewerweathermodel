#!/bin/bash
#SBATCH --job-name=aida_forecast
#SBATCH --account=epic
#SBATCH --partition=u1-compute
#SBATCH --mem=192G
#SBATCH --nodes=1
#SBATCH --time=6:00:00
#SBATCH --output=log.forecast_%J.out
#SBATCH --export=NONE

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

gnn_chpt=checkpoints/aida_gnn.pt

# Define execution parameters
YEAR="2026"
MONTH="01"
DAYS_IN_MONTH=31
STEPS=20        # 20 steps x 6h = 120h (5-day forecast)
MAX_JOBS=4      # number of concurrent forecast rollouts via GNU xargs

# Create temporary work dir for command dispatch
WORKDIR="tmp_job_dispatch_${SLURM_JOB_ID:-manual}"
mkdir -p ${WORKDIR}
CMD_FILE="${WORKDIR}/forecast_commands.txt"
rm -f ${CMD_FILE}

echo "[ROLLOUT] Generating monthly forecast commands for ${YEAR}-${MONTH} (00, 06, 12, 18z)..."

# Export bash function for GNU xargs subshells
run_single_forecast() {
    local yyyymmdd=$1
    local cycle_hour=$2
    local gnn_ckpt=$3
    local forecast_steps=$4


    # Add UTC to prevent timezone offset interference
    base="${yyyymmdd:0:4}-${yyyymmdd:4:2}-${yyyymmdd:6:2} ${cycle_hour}:00 UTC"

    # Subtract 6 hours
    prev_date=$(date -u -d "$base - 6 hours" "+%Y%m%d")
    prev_hour=$(date -u -d "$base - 6 hours" "+%H")

    # Add 6 hours
    fcst_date=$(date -u -d "$base + 6 hours" "+%Y%m%d")
    fcst_hour=$(date -u -d "$base + 6 hours" "+%H")

    out_dir="output/${fcst_date}/t${fcst_hour}z"

    prev_file="../data/icosahedral-truth/icosahedral_logstate_m6.${prev_date}.t${prev_hour}z.0p25.f000.nc"
    zero_file="../data/icosahedral-truth/icosahedral_logstate_m6.${yyyymmdd}.t${cycle_hour}z.0p25.f000.nc"

    # Verify initial analysis condition files exist before dispatching
    if [ -f "${prev_file}" ] && [ -f "${zero_file}" ]; then
      # Check if out_dir is already created.
      if [ ! -d "${out_dir}" ]; then
        mkdir -p "${out_dir}"
        out_pattern="${out_dir}/fcst.0p25.f{lead:03d}.nc"
        
        python scripts/run_aida_forecast.py \
          -e ../graph/graph-grid/icosahedral_edge_index_m6.pt \
          -k ${gnn_ckpt} \
          -s ${forecast_steps} \
          -m ${prev_file} \
          -z ${zero_file} \
          -o ${out_pattern}
      fi
    fi
}

export -f run_single_forecast

dayinmonth=(31 28 31 30 31 30 31 31 30 31 30 31)
if (( (YEAR % 400 == 0) || (YEAR % 4 == 0 && YEAR % 100 != 0) )); then
    dayinmonth[1]=29
else
    dayinmonth[1]=28
fi

# Build initial command queue for all 124 cycles in the month
mn=$((10#$MONTH))
day=1
while [ $day -le ${dayinmonth[mn]} ]
do
    yyyymmdd="${YEAR}${MONTH}${day}"
    
    for hour in 00 06 12 18
    do
        echo "run_single_forecast ${yyyymmdd} ${hour} ${gnn_chpt} ${STEPS}" >> ${CMD_FILE}
    done
    day=$(( day + 1 ))
done

TOTAL_JOBS=$(wc -l < ${CMD_FILE})
echo "=============================================================================="
echo " STARTING PARALLEL MONTHLY FORECAST ROLLOUT ENGINE"
echo " Total Initializations : ${TOTAL_JOBS} cycles"
echo " Concurrent Workers    : ${MAX_JOBS} GNU xargs processes"
echo " Target Output Pattern : output/YYYYMMDD/tHHz/fcst.0p25.fXXX.nc"
echo "=============================================================================="

set -x

# Execute 8 jobs concurrently using GNU xargs
# time xargs -a ${CMD_FILE} -P ${MAX_JOBS} -n 1 -I {} bash -c "{}"
time xargs -a ${CMD_FILE} -P ${MAX_JOBS} -L 1 bash -c '"$@"' _

set +x
rm -rf ${WORKDIR}
echo "[SUCCESS] All monthly forecast rollouts completed!"
