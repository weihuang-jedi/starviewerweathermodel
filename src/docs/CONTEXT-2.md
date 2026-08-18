# AIDA GNN Surrogate Weather Model & AI-DA Assimilation Framework
**Context & State Summary — August 2026**

## 1. Project Overview & Architecture
The AIDA (AI Data Assimilation) Surrogate is a global 3D Graph Neural Network (GNN) designed for high-resolution atmospheric weather forecasting and observation-guided data assimilation. It operates natively over **unstructured icosahedral spatial meshes** (M4 topology: 2,562 nodes, 32 vertical levels) using **3D terrain-following geometric height coordinates ($z$ in meters)** and **static surface topography conditioning** (ETOPO2022 surface elevation + land-sea mask).

### Key State Vector Variables
The dynamic atmospheric state vector consists of 7 physical/logarithmic fields across 32 vertical levels:
$$\mathbf{x} = \left[ \ln T,\, u,\, v,\, w,\, q,\, \ln \rho,\, \ln p \right]^T$$
* **$\ln T$**: Logarithmic Air Temperature ($\text{K}$)
* **$u, v, w$**: Zonal ($\text{m/s}$), Meridional ($\text{m/s}$), and Lagrangian Pressure Vertical Velocity ($\text{Pa/s}$)
* **$q$**: Specific Humidity ($\text{kg/kg}$)
* **$\ln \rho$**: Logarithmic Air Density ($\text{kg/m}^3$)
* **$\ln p$**: Logarithmic Pressure ($\text{Pa}$ or $\text{hPa}$)

---

## 2. Core Model Formulation & Improvements

### Extrapolation Baseline + GNN Variational Residual Scheme
To prevent model collapse to zero predictions and eliminate extreme initial prediction biases, the GNN is formulated to predict a **variational trend correction ($\delta \mathbf{X}_{\text{GNN}}$)** over an explicit linear extrapolation baseline ($\mathbf{X}_{\text{trend}}$):

$$\mathbf{X}_{\text{pred}}(t+6\text{h}) = \mathbf{X}_0 + (\mathbf{X}_0 - \mathbf{X}_{-6\text{h}}) + \delta \mathbf{X}_{\text{GNN}}$$

1. **Guaranteed Physical Scale**: Even at step 0 or with untrained weights ($\delta \mathbf{X}_{\text{GNN}} \approx 0$), the forecast defaults to persistent trajectory continuation, ensuring realistic atmospheric temperature ($T \approx 285\text{ K}$) and pressure ($P \approx 1000\text{ hPa}$).
2. **Channel Standardization in Loss Engine**: Targets are standardized using per-channel means $\mu$ and standard deviations $\sigma$ inside `AIDASurrogateLoss` to balance gradients across disparate variable scales ($q \sim 0.005$ vs. $\ln p \sim 10.5$).
3. **Observation Vertical $p \to z$ Interpolation**: Conventional observations recorded on isobaric pressure levels ($p$ in $\text{hPa}$) are interpolated 1D log-linearly onto the 3D model grid height levels ($z$ in meters) using `h_3d` and background pressure profiles.

---

## 3. Diagnostics & Solved Issues

| Issue / Symptom | Root Cause | Technical Solution |
| :--- | :--- | :--- |
| **`IndexError` during backward interpolation** | `NearestNDInterpolator` returning float values instead of integer array indices. | Passed `padded_vals` directly into `NearestNDInterpolator(src_points, padded_vals)`. |
| **`loss_base = inf / nan` during training** | Unbounded `exp(ln_rho)` exponential overflow & Equator Coriolis division by zero ($f \to 0$). | Clamped density $\rho \in [10^{-4}, 2.0]\text{ kg/m}^3$ and enforced minimum $f_{\text{coriolis}}$ magnitude threshold ($2 \times 10^{-5}$). |
| **Forecast evaluated to zero ($T \approx 1.0\text{ K}$)** | Neural network predicting absolute zero state from un-normalized outputs. | Introduced linear trend baseline ($X_{\text{trend}}$) + residual GNN update ($\delta X_{\text{GNN}}$) in `models/gnn.py` and `scripts/run_aida_forecast.py`. |
| **Polar plot distortion ("scattered squares")** | Standard `plt.scatter()` using constant pixel sizes near converging polar meridians. | Interpolated unstructured nodes to a 2D regular grid (`griddata`) and rendered continuous contours with `pcolormesh`. |

---

## 4. Current Performance Metrics (Epoch 5 Validation)

Evaluated at $+12\text{h}$ lead time (`output/aida.20260101.t12z.1p00.f012.nc`) against GFS truth (`gfs.20260101.t18z.1p00.f000.nc`):

```text
=====================================================================================
 METRIC SUMMARY TABLE | Lead Time: +12h
=====================================================================================
 Var   | Level   | Mean Truth   | RMSE         | BIAS         | ACC
-------------------------------------------------------------------------------------
 T     | L01    |     285.4272 |      10.1985 |      +0.4183 |   0.7862
 T     | L17    |     275.6626 |       3.1142 |      +0.1078 |   0.9675
 T     | L32    |     215.5189 |       2.2888 |      +0.0401 |   0.9674
 P     | L01    |     998.4570 |       4.9148 |      -0.2104 |   0.8973
 P     | L17    |     746.1491 |       3.2163 |      -0.1638 |   0.9772
 P     | L32    |      92.4138 |       0.3168 |      -0.0025 |   0.9928
 RHO   | L01    |       1.2219 |       0.0471 |      -0.0012 |   0.7833
 RHO   | L17    |       0.9442 |       0.0116 |      -0.0006 |   0.9255


Bias Alignment: Temperature and pressure biases are centered near zero ($\Delta T_{\text{bias}} \approx +0.04\text{ K}$, $\Delta P_{\text{bias}} \approx -0.16\text{ hPa}$).Spatial Pattern Correlation: Anomaly Correlation Coefficients (ACC) reach $0.96 - 0.99$ in mid and upper atmospheric levels.5. Repository Utility Scripts ReferenceTraining Engine: python -u scripts/train_aida_surrogate.py --config configs/config.yamlAutoregressive Rollout: python scripts/run_aida_forecast.py -k checkpoints/aida_gnn_surrogate_logstate_epoch_005.pt -e ../graph/graph-grid/icosahedral_edge_index_m4.pt -s 4 -m ... -z ... -o 'output/aida.20260101.t12z.1p00.f{lead:03d}.nc'Verification Evaluator: python utils/evaluate_aida_forecast.py -f output/aida.20260101.t12z.1p00.f012.nc -t ../data/icosahedral-truth/gfs.20260101.t18z.1p00.f000.nc -c metrics_f012.csv -p metrics_f012_profile.png4x7 Diagnostic Panel Plotter: python utils/plot_forecast_panel_comparison.py -f output/aida.20260101.t12z.1p00.f012.nc -t ../data/icosahedral-truth/gfs.20260101.t18z.1p00.f000.nc -z ../data/icosahedral-truth/gfs.20260101.t06z.1p00.f000.nc -l 0 -o surface_panel.png3x5 Lead-Time Progression Plotter: python utils/plot_forecast_leadtime_series.py -v T -l 0 -o leadtime_series_T_surface.pngIcosahedral $\to$ Regular Grid Reverse Interpolator: python utils/interpolate_icosahedral_to_terrain.py -i icosahedral_file.nc -r reference_regular_template.nc -o output_regular.nc6. Next Steps & Ongoing ObjectivesBoundary Layer Refinement: Resume training from Epoch 5 up to Epoch 20 to reduce near-surface temperature and wind RMSE ($U, V$).Satellite Radiance Operator Assimilation: Enable multi-sensor forward radiance operators incrementally in configs/config.yaml (e.g., w_rad_amsua: 0.01, w_rad_atms: 0.01, w_rad_iasi: 0.01).
<FollowUp label="Want me to save this summary directly into CONTEXT.md?" query="Write the summary

