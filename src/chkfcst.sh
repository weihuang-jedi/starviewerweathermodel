#!/bin/bash
# ==============================================================================
# generate_daily_slurm_jobs.sh
# ------------------------------------------------------------------------------
# Generates 31 standalone SLURM batch scripts (runfcst_day01.slurm to runfcst_day31.slurm).
# Each generated script runs the 4 cycles (00z, 06z, 12z, 18z) for its specific fixed day.
# ==============================================================================

YEAR="2026"
MONTH="01"

# Leap-year handling for days in month
dayinmonth=(31 28 31 30 31 30 31 31 30 31 30 31)
if (( (YEAR % 400 == 0) || (YEAR % 4 == 0 && YEAR % 100 != 0) )); then
    dayinmonth[1]=29
else
    dayinmonth[1]=28
fi

mn=$((10#$MONTH))
total_days=${dayinmonth[$((mn - 1))]}
fcst_length=120
fcst_interval=6

for (( day=2; day<=total_days; day++ ))
do
    echo "Working on day: ${day}..."
    day_str=$(printf "%02d" $day)
    yyyymmdd="${YEAR}${MONTH}${day_str}"

    for hour in 00 06 12 18
    do
        echo "      Working on hour: ${hour}..."
        out_dir="output/${yyyymmdd}/t${hour}z"
        fcst_hour=0
	while [[ $fcst_hour -le $fcst_length ]]
        do
	    fcst_str=$(printf "%03d" $fcst_hour)
	    fcst_hour=$(( fcst_hour + fcst_interval ))
            fcstfile="${out_dir}/fcst.0p25.f${fcst_str}.nc"
            if [ ! -f ${fcstfile} ]
	    then
                echo "fcstfile: ${fcstfile} missing. stop"
                exit 1
            fi
        done
    done
done

