To transition the AIDA pipeline from a flat constant geometric height ($z$) coordinate to a **terrain-following height coordinate system** $h(lat, lon, level) = H_{\text{max}} - \eta \cdot (H_{\text{max}} - H_{\text{terrain}})$, changes are required across **three primary files**:

---

## Architecture Change Summary

| File | Primary Change | Purpose |
| --- | --- | --- |
| **`models/gnn.py`** | Add static terrain features (`elevation`, `land_sea_mask`, `h_terrain`) to the GNN node input. | Informs graph message-passing layers about local topography and boundary layer height. |
| **`models/dataset.py`** | Load 3D terrain heights $h$ and surface elevation $H_{\text{terrain}}$ alongside log-state variables ($X_{-6\text{h}}, X_0 \to X_{+6\text{h}}$). | Serves 3D terrain-following coordinate fields to the GNN and satellite radiance loss operators. |
| **`models/loss.py`** / Forward Radiance Operators | Pass dynamic 3D geometric heights $h$ into satellite forward radiance operators $H(\mathbf{x})$. | Replaces static flat-height pressure/altitude levels with actual 3D heights when evaluating satellite optical depth & weighting functions. |

---

### Step 1: Update `models/gnn.py`

Modify `IcosahedralGNNSurrogate` to accept static surface topography features (`elevation`, `land_sea_mask`) at the node level along with the dynamic log-state variables:

```python
# In models/gnn.py

import torch
import torch.nn as nn

class IcosahedralGNNSurrogate(nn.Module):
    def __init__(
        self,
        in_vars: int = 14,         # Dynamic variables (e.g., 7 vars x 2 timesteps for 4D forecast)
        out_vars: int = 7,        # Predicted 7 log-state variables
        num_static_feats: int = 2, # Static node features: elevation, land_sea_mask
        hidden_dim: int = 128,
        num_levels: int = 32,
        num_layers: int = 4
    ):
        super().__init__()
        self.num_levels = num_levels
        self.hidden_dim = hidden_dim

        # Input dimension: Dynamic State [In_Vars, Levels] + Static Topography [Num_Static]
        total_in_dim = (in_vars * num_levels) + num_static_feats

        self.encoder = nn.Linear(total_in_dim, hidden_dim)
        
        # GNN message passing layers...
        self.gnn_layers = nn.ModuleList([
            # GraphConv / Interaction blocks
        ])

        self.decoder = nn.Linear(hidden_dim, out_vars * num_levels)

    def forward(self, x_dynamic: torch.Tensor, edge_index: torch.Tensor, static_topo: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x_dynamic: [Batch, In_Vars, Levels=32, Nodes]
            edge_index: [2, Num_Edges]
            static_topo: [Batch, Num_Static=2, Nodes] - Topography elevation + Land/Sea mask
        """
        batch_size, num_vars, num_levels, num_nodes = x_dynamic.shape

        # Flatten vertical levels: [Batch, In_Vars * Levels, Nodes]
        x_flat = x_dynamic.view(batch_size, num_vars * num_levels, num_nodes)

        # Append 2D static terrain features if provided
        if static_topo is not None:
            x_flat = torch.cat([x_flat, static_topo], dim=1)  # [Batch, (In_Vars*Levels)+2, Nodes]

        x_flat = x_flat.permute(0, 2, 1)  # [Batch, Nodes, Channels]
        feat = self.encoder(x_flat)       # [Batch, Nodes, Hidden_Dim]

        # Process through GNN message-passing blocks
        for gnn in self.gnn_layers:
            feat = gnn(feat, edge_index)

        out_flat = self.decoder(feat)     # [Batch, Nodes, Out_Vars * Levels]
        out_flat = out_flat.permute(0, 2, 1).view(batch_size, out_vars, num_levels, num_nodes)

        return out_flat

```

---

### Step 2: Update `models/dataset.py`

Update `LogState4DForecastDataset` to load `h_icosahedral` (3D height) and `h_terrain_icosahedral` (2D surface elevation) from `icosahedral_logstate.zarr`:

```python
# In models/dataset.py

class LogState4DForecastDataset(Dataset):
    def __init__(self, zarr_path: str, obs_dir: str = None):
        super().__init__()
        self.ds = xr.open_zarr(zarr_path)

        # Dynamic atmospheric variables
        self.var_names = [
            'ln_t_icosahedral', 'u_icosahedral', 'v_icosahedral',
            'w_icosahedral', 'q_icosahedral', 'ln_rho_icosahedral', 'ln_p_icosahedral'
        ]

    def __getitem__(self, idx):
        t_idx = self.valid_indices[idx]
        idx_minus6, idx_zero, idx_plus6 = t_idx - 1, t_idx, t_idx + 1

        # 1. Extract 3D dynamic state tensors [7, 32, 2562]
        x_minus6 = np.stack([self.ds[v].isel(time=idx_minus6).values for v in self.var_names], axis=0)
        x_zero   = np.stack([self.ds[v].isel(time=idx_zero).values for v in self.var_names], axis=0)
        target   = np.stack([self.ds[v].isel(time=idx_plus6).values for v in self.var_names], axis=0)

        x_trajectory = np.concatenate([x_minus6, x_zero], axis=0).astype(np.float32)

        # 2. Extract 3D terrain-following heights [32, 2562] and 2D surface topography [2562]
        h_3d = self.ds['h_icosahedral'].isel(time=idx_zero).values.astype(np.float32)
        h_terrain = self.ds['h_terrain_icosahedral'].isel(time=idx_zero).values.astype(np.float32)
        ls_mask = self.ds['land_sea_mask'].values.astype(np.float32)

        # Pack static terrain inputs: elevation (normalized) and land-sea mask [2, 2562]
        static_topo = np.stack([h_terrain / 10000.0, ls_mask], axis=0)

        item = {
            'input_trajectory': torch.from_numpy(x_trajectory),
            'target_state': torch.from_numpy(target.astype(np.float32)),
            'h_3d': torch.from_numpy(h_3d),                   # [32, Nodes]
            'static_topo': torch.from_numpy(static_topo),     # [2, Nodes]
        }

        # Load future observations O_plus6...
        item.update(self._load_observations_for_time(self.times[idx_plus6]))
        return item

```

---

### Step 3: Update Satellite Forward Radiance Operators (`models/*.py`)

In satellite forward radiance operators (e.g., `models/amsua.py`, `models/iasi.py`, `models/seviri.py`), replace flat height profiles with the dynamic 3D terrain-following height tensor `h_3d` when computing level-to-channel weighting functions:

```python
# Example update in models/seviri.py (or amsua.py, iasi.py, etc.)

class DifferentiableSEVIRIOperator(nn.Module):
    def __init__(self, num_levels: int = 32):
        super().__init__()
        self.num_levels = num_levels

    def forward(self, temp_k: torch.Tensor, press_pa: torch.Tensor, h_3d: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            temp_k: Temperature [Batch, Levels=32, Nodes]
            press_pa: Pressure [Batch, Levels=32, Nodes]
            h_3d: Dynamic Terrain-Following Heights [Batch, Levels=32, Nodes]
        """
        press_pa = torch.clamp(press_pa, min=10.0)
        ln_press = torch.log(press_pa)

        # Incorporate terrain-following height z-variation if provided
        if h_3d is not None:
            # Adjust weighting kernels based on actual 3D geometric height h_3d
            weight_logits = -0.5 * ((h_3d.unsqueeze(1) - self.peak_heights.view(1, -1, 1, 1)) / self.sigma_heights.view(1, -1, 1, 1)) ** 2
        else:
            # Standard log-pressure weighting fallback
            weight_logits = -0.5 * ((ln_press.unsqueeze(1) - torch.log(self.peak_p).view(1, -1, 1, 1)) / self.sigma_ln_p.view(1, -1, 1, 1)) ** 2

        weights = F.softmax(weight_logits, dim=2)
        tb_sim = torch.sum(temp_k.unsqueeze(1) * weights, dim=2)
        return tb_sim

```

---

### Step 4: Update `scripts/train_aida_surrogate.py`

In `train_epoch()`, extract `static_topo` and `h_3d` from the dataloader batch and pass them to the model and loss calculations:

```python
# In scripts/train_aida_surrogate.py

for batch_data in dataloader:
    x_batch = batch_data['input_trajectory'].to(device)
    y_batch = batch_data['target_state'].to(device)
    static_topo = batch_data['static_topo'].to(device)  # [Batch, 2, Nodes]
    h_3d = batch_data['h_3d'].to(device)                # [Batch, 32, Nodes]

    # GNN Forward pass with terrain features
    pred = model(x_batch, edge_index, static_topo=static_topo)

    # Evaluate Satellite Radiance Loss passing dynamic 3D terrain heights (h_3d)
    tb_seviri_sim = seviri_op(t_k_perm, p_pa, h_3d=h_3d)

```

---

## Implementation Roadmap

```
1. models/gnn.py      -> Add static_topo concatenation to encoder
2. models/dataset.py  -> Add 'h_3d' and 'static_topo' extraction from Zarr
3. models/*.py        -> Pass h_3d to satellite forward radiance weighting kernels
4. train_aida_surrogate.py -> Feed static_topo and h_3d through training loop

```

With these updates, AIDA directly ingests surface elevation and 3D terrain-following geometry across all model layers and forward radiance operators.
