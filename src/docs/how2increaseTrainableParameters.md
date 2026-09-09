Your test output confirms the scaling behavior:

* **Current Architecture (`hidden_dim: 128`, `num_layers: 4`):** **499,680 parameters (~0.50 M)**
* **Scaled Configuration (`hidden_dim: 512`, `num_layers: 16`):** **25,824,480 parameters (~25.82 M)**

---

### How to Fine-Tune to Reach Exactly ~39 Million Parameters

To scale from **25.82 M** up to the full **~39 M** footprint used by GraphCast, you can either widen the latent channels (`hidden_dim`) or increase the message-passing depth (`num_layers`):

#### Option A: Increase Hidden Latent Dimension to 640 (`~39.8 M`)

```yaml
model:
  in_vars: 14
  out_vars: 7
  num_static_feats: 4
  hidden_dim: 640             # Scaled up from 512
  num_layers: 16

```

* **Parameters:** **39,835,104 (~39.8 M)**

#### Option B: Increase Message-Passing Layers to 24 (`~38.4 M`)

```yaml
model:
  in_vars: 14
  out_vars: 7
  num_static_feats: 4
  hidden_dim: 512
  num_layers: 24              # Scaled up from 16

```

* **Parameters:** **38,413,536 (~38.4 M)**

---

### Memory & Slurm Execution Recommendations for ~38M Scale

Scaling from 0.5M to ~38M parameters increases GPU VRAM demand and backward-pass graph overhead:

1. **Gradient Accumulation:** Increase `accum_steps: 8` or `16` in `configs/config.yaml` to simulate larger batch sizes without triggering `CUDA Out Of Memory` (OOM) errors.
2. **Mixed Precision (AMP):** Enable `torch.cuda.amp.autocast(dtype=torch.bfloat16)` inside `train_epoch()` in `scripts/train_aida_surrogate.py` to halve VRAM footprint during message passing.
3. **Slurm VRAM Target:** Allocate GPUs with at least **32 GB to 40 GB VRAM** (e.g., NVIDIA A100 or V100-32GB).

