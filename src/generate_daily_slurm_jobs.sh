#!/bin/bash
# ==============================================================================
# generate_daily_slurm_jobs.sh
# ------------------------------------------------------------------------------
# Generates 31 standalone SLURM batch scripts (runfcst_day01.slurm to runfcst_day31.slurm).
# Each generated script runs the 4 cycles (00z, 06z, 12z, 18z) for its specific fixed day.
# ==============================================================================

YEAR="2026"
MONTH="07"

# Leap-year handling for days in month
dayinmonth=(31 28 31 30 31 30 31 31 30 31 30 31)
if (( (YEAR % 400 == 0) || (YEAR % 4 == 0 && YEAR % 100 != 0) )); then
    dayinmonth[1]=29
else
    dayinmonth[1]=28
fi

mn=$((10#$MONTH))
total_days=${dayinmonth[$((mn - 1))]}

SCRIPT_DIR="daily_slurm_scripts"
mkdir -p ${SCRIPT_DIR}

echo "[GENERATOR] Generating ${total_days} Slurm scripts in '${SCRIPT_DIR}/'..."

for (( day=1; day<=total_days; day++ )); do
    day_str=$(printf "%02d" $day)
    yyyymmdd="${YEAR}${MONTH}${day_str}"
    script_file="${SCRIPT_DIR}/runfcst_day${day_str}.slurm"

    cat <<EOF > ${script_file}
#!/bin/bash
#SBATCH --job-name=aida_fcst_${day_str}
#SBATCH --account=epic
#SBATCH --partition=u1-compute
#SBATCH --mem=192G
#SBATCH --nodes=1
#SBATCH --time=3:00:00
#SBATCH --output=log.forecast_day${day_str}_%J.out
#SBATCH --export=NONE

SVGHOME=/scratch5/purged/Wei.Huang/src/starviewerweathermodel
source \${SVGHOME}/svg.env
cd \${SVGHOME}/src

if [ -f checkpoints/aida_gnn_surrogate_logstate.pt ]; then
    n=0
    while [ -f checkpoints/aida_gnn_surrogate_logstate_\${n}.pt ]; do
       n=\$(( n + 1 ))
    done
    mv checkpoints/aida_gnn_surrogate_logstate.pt checkpoints/aida_gnn_surrogate_logstate_\${n}.pt
fi

gnn_chpt=checkpoints/aida_gnn.pt

# Fixed date execution parameters
YEAR="${YEAR}"
MONTH="${MONTH}"
DAY="${day_str}"
YYYYMMDD="${yyyymmdd}"
STEPS=20        # 20 steps x 6h = 120h (5-day forecast)
MAX_JOBS=4      # Number of concurrent forecast rollouts via GNU xargs

WORKDIR="tmp_job_dispatch_day${day_str}_\${SLURM_JOB_ID:-manual}"
mkdir -p \${WORKDIR}
CMD_FILE="\${WORKDIR}/forecast_commands.txt"
rm -f \${CMD_FILE}

run_single_forecast() {
    local yyyymmdd=\$1
    local cycle_hour=\$2
    local gnn_ckpt=\$3
    local forecast_steps=\$4

    base="\${yyyymmdd:0:4}-\${yyyymmdd:4:2}-\${yyyymmdd:6:2} \${cycle_hour}:00 UTC"

    prev_date=\$(date -u -d "\$base - 6 hours" "+%Y%m%d")
    prev_hour=\$(date -u -d "\$base - 6 hours" "+%H")

    fcst_date=\$(date -u -d "\$base + 6 hours" "+%Y%m%d")
    fcst_hour=\$(date -u -d "\$base + 6 hours" "+%H")

    out_dir="output/\${fcst_date}/t\${fcst_hour}z"

    prev_file="../data/icosahedral-truth/icosahedral_logstate_m6.\${prev_date}.t\${prev_hour}z.0p25.f000.nc"
    zero_file="../data/icosahedral-truth/icosahedral_logstate_m6.\${yyyymmdd}.t\${cycle_hour}z.0p25.f000.nc"

    if [ -f "\${prev_file}" ] && [ -f "\${zero_file}" ]; then
      if [ ! -d "\${out_dir}" ]; then
        mkdir -p "\${out_dir}"
        out_pattern="\${out_dir}/fcst.0p25.f{lead:03d}.nc"

        python scripts/run_aida_forecast.py \\
          -e ../graph/graph-grid/icosahedral_edge_index_m6.pt \\
          -k \${gnn_ckpt} \\
          -s \${forecast_steps} \\
          -m \${prev_file} \\
          -z \${zero_file} \\
          -o \${out_pattern}
      fi
    fi
}

export -f run_single_forecast

# Add 4 fixed cycles for Day ${day_str}
for hour in 00 06 12 18
do
    echo "run_single_forecast \${YYYYMMDD} \${hour} \${gnn_chpt} \${STEPS}" >> \${CMD_FILE}
done

TOTAL_JOBS=\$(wc -l < \${CMD_FILE})
echo "=============================================================================="
echo " STARTING PARALLEL FORECAST ROLLOUT FOR DAY ${day_str} (\${YYYYMMDD})"
echo " Total Initializations : \${TOTAL_JOBS} cycles"
echo " Concurrent Workers    : \${MAX_JOBS} GNU xargs processes"
echo " Target Output Pattern : output/\${YYYYMMDD}/tHHz/fcst.0p25.fXXX.nc"
echo "=============================================================================="

set -x

time xargs -a \${CMD_FILE} -P \${MAX_JOBS} -L 1 bash -c '"\$@"' _

set +x
rm -rf \${WORKDIR}
echo "[SUCCESS] Forecast rollout for Day ${day_str} (\${YYYYMMDD}) completed!"
EOF

    sbatch ${script_file}
done

echo "[SUCCESS] Generated ${total_days} Slurm scripts under '${SCRIPT_DIR}/'."
