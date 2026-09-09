#!/usr/bin/env bash
# ==============================================================================
# run_monthly_verification.sh
# ------------------------------------------------------------------------------
# Generates a full month of 6-hourly initialization timestamps (00, 06, 12, 18Z)
# and processes verification diagnostics in parallel across 8 CPU workers.
# ==============================================================================

YEAR="2026"
MONTH="01"
DAYS_IN_MONTH=31
MAX_JOBS=8  # Number of concurrent jobs

FCST_DIR="output"
TRUTH_DIR="../data/icosahedral-truth"
OUT_DIR="plots_monthly_verification"

mkdir -p "${OUT_DIR}"
LIST_FILE="${OUT_DIR}/inits_list.txt"
rm -f "${LIST_FILE}"

echo "[GENERATOR] Building initialization list for ${YEAR}-${MONTH} (00, 06, 12, 18Z)..."

# Generate 4 cycles per day for the entire month
for day in $(seq -w 1 ${DAYS_IN_MONTH}); do
    for hour in "00" "06" "12" "18"; do
        echo "${YEAR}${MONTH}${day}${hour}" >> "${LIST_FILE}"
    done
done

TOTAL_CYCLES=$(wc -l < "${LIST_FILE}")
echo "[INFO] Total Initialization Cycles to Evaluate: ${TOTAL_CYCLES}"
echo "[INFO] Launching parallel verification using GNU xargs with ${MAX_JOBS} concurrent workers..."
echo "=============================================================================="

# GNU xargs parallel execution engine (-P 8 runs 8 processes simultaneously)
cat "${LIST_FILE}" | xargs -n 1 -P ${MAX_JOBS} -I {} python3 utils/plot_forecast_leadtime_series.py \
    --init_time {} \
    --max_lead 120 \
    --fcst_dir "${FCST_DIR}" \
    --truth_dir "${TRUTH_DIR}" \
    --out_dir "${OUT_DIR}"

echo "=============================================================================="
echo "[SUCCESS] Monthly verification complete! All plots stored in '${OUT_DIR}/'."
