To scale the 3D momentum residual loss ($L_{\text{momentum}}$) relative to the state reconstruction loss ($L_{\text{mse}}$), adjust `w_dynamics` inside **`configs/config.yaml`** under the `loss_weights` block.

### `configs/config.yaml`

```yaml
loss_weights:
  w_mse: 1.0           # Baseline State MSE weight
  w_conv: 0.05         # Conventional observation weight
  w_wind_ke: 0.15      # Kinetic energy conservation weight
  w_wind_dir: 0.10     # Wind vector cosine direction alignment weight
  w_laplacian_p: 0.01  # Pressure smoothness regularization weight

  # =========================================================================
  # 3D NAVIER-STOKES MOMENTUM RESIDUAL LOSS WEIGHT (w_dynamics)
  # Controls the physical momentum conservation penalty relative to w_mse.
  # Recommended range: 0.001 (loose constraint) to 0.05 (strict physical constraint).
  # Set to 0.0 to disable momentum residual computation entirely.
  # =========================================================================
  w_dynamics: 0.005

```

---

### Tuning Recommendations

| Relative Scale | `w_dynamics` Value | Behavior / Effect |
| --- | --- | --- |
| **Disabled** | `0.0` | Pure empirical state learning; skips dynamic residual graph computations for maximum throughput. |
| **Gentle Regularization** *(Default)* | `0.001` - `0.005` | Prevents physically non-viable momentum spikes without hindering MSE convergence speed. |
| **Balanced Physical DA** | `0.01` | Equalized gradient contributions between state trajectory reconstruction and Navier-Stokes momentum balance. |
| **Strict Conservation** | `0.02` - `0.05` | Enforces strong geostrophic / ageostrophic balance constraints; ideal for long-rollout stability. |

---

### Verify Scaling During Training

When you run `python -u scripts/train_aida_surrogate.py --config configs/config.yaml`, monitor the epoch loss breakdown in stdout to ensure `MOMENTUM` loss matches your desired order of magnitude:

```text
  ├─ [Epoch 007] Batch 0100/2924 (  3.4%) | Total Loss: 1.24501e-02 | State MSE: 8.95012e-03 | MOMENTUM: 1.45021e-01 | Speed: 1.85 batch/s

```

If `MOMENTUM` dominates `Total Loss` by more than an order of magnitude (e.g., $> 1.0$), reduce `w_dynamics` by a factor of 5 to 10.
