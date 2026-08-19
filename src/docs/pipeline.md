Here is the complete, end-to-end shell command workflow to run the **AIDA 4D Observation-Guided Weather Assimilation and Forecasting Pipeline**.

This workflow executes every phase from downloading raw GFS GRIB2 and ETOPO topography data, interpolating into **3D terrain-following geometric height coordinates**, converting into icosahedral log-state Zarr stores, training the GNN surrogate model, and running a **24-hour multi-step forecast rollout**.

---

### Phase 1: Environment Setup & Directory Structure

```bash
# 1. Set environment paths
export SVGHOME=/scratch4/NAGAPE/epic/Wei.Huang/src/starviewerweathermodel
export DATA_DIR=${SVGHOME}/data
export SRC_DIR=${SVGHOME}/src

# 2. Activate Python conda environment
source ${SVGHOME}/svg.env

# 3. Create required directory layout
mkdir -p ${DATA_DIR}/raw_gfs
mkdir -p ${DATA_DIR}/etopo
mkdir -p ${DATA_DIR}/terrain-regular-grid
mkdir -p ${DATA_DIR}/icosahedral-grid
mkdir -p ${DATA_DIR}/obs_unified
mkdir -p ${DATA_DIR}/graph
mkdir -p ${SRC_DIR}/checkpoints
mkdir -p ${SRC_DIR}/forecasts
cd ${SRC_DIR}

```

---

### Phase 2: Topography & Raw Weather Data Fetching

```bash
# 1. Download ETOPO 2022 60-arcsecond global surface geoid topography (if not already present)
wget -N https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO2022/data/60s/60s_geoid_netcdf/ETOPO_2022_v1_60s_N90W180_geoid.nc \
  -P ${DATA_DIR}/etopo/

# 2. Fetch raw NOAA GFS GRIB2 operational analysis files for consecutive 6-hour cycles
# Initial trajectory triple: t-6h (00z), t0 (06z), t+6h (12z)
wget -N "https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/gfs.20240730/00/atmos/gfs.t00z.pgrb2.1p00.f000" -O ${DATA_DIR}/raw_gfs/gfs.20240730.t00z.grib2
wget -N "https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/gfs.20240730/06/atmos/gfs.t06z.pgrb2.1p00.f000" -O ${DATA_DIR}/raw_gfs/gfs.20240730.t06z.grib2
wget -N "https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/gfs.20240730/12/atmos/gfs.t12z.pgrb2.1p00.f000" -O ${DATA_DIR}/raw_gfs/gfs.20240730.t12z.grib2

# 3. Fetch unified observation files (prepbufr conventional + 9 satellite radiance BUFR files)
python fetch_conv_amsua_iasi_hms_atms.py \
  --date 20240730 \
  --cycle 06 \
  --output_dir ${DATA_DIR}/obs_unified/

```

---

### Phase 3: Terrain-Following Vertical Coordinate Transformation

Interpolate pressure-level GRIB2 weather fields into 3D terrain-following geometric height levels:


$$h(\text{level}, \text{lat}, \text{lon}) = H_{\text{max}} - \eta \cdot (H_{\text{max}} - H_{\text{terrain}})$$

```bash
# Transform raw GRIB2 files onto terrain-following height levels across all cycles
python interpolate_to_terrain_heights.py \
  -i ${DATA_DIR}/raw_gfs/gfs.20240730.t00z.grib2 \
  -e ${DATA_DIR}/etopo/ETOPO_2022_v1_60s_N90W180_geoid.nc \
  -o ${DATA_DIR}/terrain-regular-grid/th_gfs.20240730.t00z.nc

python interpolate_to_terrain_heights.py \
  -i ${DATA_DIR}/raw_gfs/gfs.20240730.t06z.grib2 \
  -e ${DATA_DIR}/etopo/ETOPO_2022_v1_60s_N90W180_geoid.nc \
  -o ${DATA_DIR}/terrain-regular-grid/th_gfs.20240730.t06z.nc

python interpolate_to_terrain_heights.py \
  -i ${DATA_DIR}/raw_gfs/gfs.20240730.t12z.grib2 \
  -e ${DATA_DIR}/etopo/ETOPO_2022_v1_60s_N90W180_geoid.nc \
  -o ${DATA_DIR}/terrain-regular-grid/th_gfs.20240730.t12z.nc

```

---

### Phase 4: Icosahedral Mesh Mapping & Thermodynamic Log-State Encoding

Map regular terrain-following grids onto the 1D M4 icosahedral graph mesh ($2,562$ nodes) and compute thermodynamic log-state variables $(\ln T, u, v, w, q, \ln \rho, \ln p)$:

```bash
# 1. Map to Icosahedral Log-State NetCDF files
python interpolate_to_logstate_icosahedral.py \
  -i ${DATA_DIR}/terrain-regular-grid/th_gfs.20240730.t00z.nc \
  -m ${DATA_DIR}/graph/global_icosahedral_mesh_m4.nc \
  -o ${DATA_DIR}/icosahedral-grid/icosahedral_logstate_m4.20240730.t00z.nc

python interpolate_to_logstate_icosahedral.py \
  -i ${DATA_DIR}/terrain-regular-grid/th_gfs.20240730.t06z.nc \
  -m ${DATA_DIR}/graph/global_icosahedral_mesh_m4.nc \
  -o ${DATA_DIR}/icosahedral-grid/icosahedral_logstate_m4.20240730.t06z.nc

python interpolate_to_logstate_icosahedral.py \
  -i ${DATA_DIR}/terrain-regular-grid/th_gfs.20240730.t12z.nc \
  -m ${DATA_DIR}/graph/global_icosahedral_mesh_m4.nc \
  -o ${DATA_DIR}/icosahedral-grid/icosahedral_logstate_m4.20240730.t12z.nc

# 2. Pack NetCDF collection into cloud-native compressed Zarr store
python pack_icosahedral_zarr.py \
  -i "${DATA_DIR}/icosahedral-grid/icosahedral_logstate_m4.2024*.nc" \
  -o ${DATA_DIR}/icosahedral_logstate.zarr \
  -c 32 \
  -l 3

```

---

### Phase 5: GNN Surrogate Model Training (Slurm Submission)

Train the GNN surrogate backbone using gradient accumulation over the full 9-instrument satellite radiance suite:

```bash
# 1. Submit Slurm training batch job on NVIDIA H100 GPU node
sbatch submit_training.sh

# 2. Monitor real-time training epoch loss breakdowns
tail -f log.training_*.out

```

---

### Phase 6: Autoregressive Forecast Rollout & Verification

Run a 24-hour multi-step forecast ($+6\text{h}, +12\text{h}, +18\text{h}, +24\text{h}$) from the initial analysis state pair $(X_{-6\text{h}}, X_0)$:

```bash
# 1. Execute 24-hour autoregressive rollout (4 steps x 6 hours)
python scripts/run_aida_forecast.py \
  -k checkpoints/aida_gnn_surrogate_logstate.pt \
  -m ${DATA_DIR}/icosahedral-grid/icosahedral_logstate_m4.20240730.t00z.nc \
  -z ${DATA_DIR}/icosahedral-grid/icosahedral_logstate_m4.20240730.t06z.nc \
  -e ${DATA_DIR}/graph/icosahedral_edge_index_m4.pt \
  -s 4 \
  -o forecasts/aida_24h_terrain_forecast.20240730.nc

# 2. Verify forecast trajectory stability, NaN/Inf checks, and physical error growth
python scripts/verify_aida_forecast.py \
  -f forecasts/aida_24h_terrain_forecast.20240730.nc

```
