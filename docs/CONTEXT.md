
# AIDA Atmospheric GNN Model & GraphCast AI-DA Framework
**Context, High-Resolution Migration & Debugging Summary — August 2026**

---

## 1. Project Overview & M6 Grid Migration
The **AIDA (AI Data Assimilation)** and **GraphCast Surrogate** framework is a 3D Graph Neural Network designed for high-resolution atmospheric forecasting and observation assimilation. The framework operates on an **unstructured 3D terrain-following icosahedral grid** using **3D geometric height coordinates ($z$ in meters)** and **static surface topography** (elevation + land-sea mask).

### Key Dataset & Grid Resolution Transition
* **Legacy Grid (M4)**: 2,562 horizontal spatial nodes across 32 vertical levels ($N_{\text{flat}} = 81,984$).
* **Updated Grid (M6)**: **40,962 horizontal spatial nodes** across 32 vertical levels ($N_{\text{flat}} = 1,310,784$).
  * Provides $\approx 0.25^\circ$ global resolution matching operational GFS data scales.
  * Zarr storage path: `/scratch5/purged/Wei.Huang/src/starviewerweathermodel/data/icosahedral_logstate.zarr`

### Primary Atmospheric State Vector (6–7 Variables)
$$\mathbf{x} = \left[ p,\, q,\, T,\, u,\, v,\, w \right]^T \quad \text{or} \quad \left[ \ln T,\, u,\, v,\, w,\, q,\, \ln \rho,\, \ln p \right]^T$$

---

## 2. Solved Engineering Bottlenecks & Resolved Errors

During the M6 high-resolution dataset packaging and model training rollout, several memory, dimensionality, and tensor stride issues were diagnosed and resolved:

| Component / Issue | Root Cause | Engineering Solution |
| :--- | :--- | :--- |
| **Zarr Packing OOM (`slurm step OOM Killed`)** | Large temporal chunking (`-c 32`) forced Dask to hold multi-gigabyte memory blocks during `xr.open_mfdataset`. | Rewrote `pack_icosahedral_zarr.py` to stream files incrementally via `append_dim='time'` with batch size `-c 4`. |
| **DataLoader `RuntimeError` (`shape '[1310784, 12]' invalid for size 10486272`)** | Hardcoded M6 node dimensions ($40,962$) applied against remaining M4 store buffers ($2,562$). | Re-pointed `config.yaml` to the new M6 Zarr store and dynamically computed dataset tensor shapes. |
| **Non-Contiguous Memory Tensor Crash (`.view()` error)** | Slicing and transposing dataset arrays broke contiguous memory strides in PyTorch. | Replaced standard `.view()` with non-contiguous-safe `.reshape()` across `training.py` and Lightning modules. |
| **Model Forward `RecursionError`** | Line 196 in `forward()` called `self.forward()` instead of passing inputs to `self.model()`. | Fixed recursive loop call in `models/graphcast_lightning_direct.py` to target the underlying `DeepGraphCastModel`. |
| **Static Tensor Shape Mismatch (`1310784` vs `873856`)** | Static topography tensor was missing 3D level expansion logic across the 32 vertical levels. | Built dynamic node-level expansion logic: `static_flat = static_flat.unsqueeze(2).expand(-1, -1, 32, -1).reshape(B, N_flat, C)`. |

---

## 3. Core Technical Architecture & Training Pipeline

### Model Design
1. **Nodes-First Memory Layout**: Inputs formatted as `(Batch, N_nodes * N_levels, C_in)`.
2. **Channel Configuration ($C_{\text{in}} = 15$)**:
   * **12 Dynamic History Features**: 2 timesteps ($t_{-1}, t_0$) $\times$ 6 variables ($p, q, T, u, v, w$).
   * **2 Surface Static Features**: Normalized surface elevation ($\text{km}$) + Land-Sea Mask.
   * **1 3D Vertical Terrain Coordinate**: Terrain-following decay function $z_{\text{3D}}(k) = z_{\text{flat}}(k) + e^{-k/10} \cdot z_{\text{elev}}$.
3. **Loss Engine**:
   * Standardized MSE loss in normalized space using dataset empirical means ($\mu$) and standard deviations ($\sigma$).
   * Weighted variable loss multipliers ($w_U = 2.5, w_V = 2.5$) to prioritize kinetic wind energy convergence.

---

## 4. Execution Commands

### Stream NetCDF to Compressed Zarr (M6 Grid)
```bash
python pack_icosahedral_zarr.py \
  -i "icosahedral-grid/icosahedral_logstate_m6.202*.nc" \
  -o icosahedral_logstate.zarr \
  -c 4 -l 3


5. Next StepsVerify GPU Execution: Move from single-CPU test runs to multi-GPU Slurm execution (accelerator="gpu").Loss Weight Tuning: Monitor near-surface wind ($u, v$) and humidity ($q$) residual loss profiles over extended epochs.Observation Ingestion: Re-enable satellite radiance operator loss functions for real-time data assimilation experiments.
<FollowUp label="Save CONTEXT.md directly to repository root?" query="Save the markdown text above into a CONTEXT.md file in the root directory."/>
