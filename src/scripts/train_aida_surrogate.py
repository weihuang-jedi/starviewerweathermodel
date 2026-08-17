#!/usr/bin/env python3
"""
scripts/train_aida_surrogate.py
-------------------------------
AIDA GNN Surrogate Model Training Script for Icosahedral Atmospheric Grids.
Supports terrain-following 3D height coordinates, static topography conditioning,
4D observation-guided forecast dataset ingestion, conditional satellite operator execution,
and verbose, real-time stdout progress flushing.
"""

import argparse
import os
import sys
import time
import yaml

# Ensure parent directory is in Python path for 'models' package imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models import (
    LogState4DForecastDataset,
    SyntheticAIDAStateDataset,
    generate_or_load_edge_index,
    IcosahedralGNNSurrogate,
    AIDASurrogateLoss,
    M4MeshOperators,
    build_icosahedral_differential_operators,
)
from models.amsua import DifferentiableAMSUAOperator
from models.iasi import DifferentiableIASIOperator
from models.hms import DifferentiableHMSOperator
from models.atms import DifferentiableATMSOperator
from models.cris import DifferentiableCrISOperator
from models.seviri import DifferentiableSEVIRIOperator
from models.gsrasr import DifferentiableGSRASROperator
from models.gsrcsr import DifferentiableGSRCSROperator
from models.ahicsr import DifferentiableAHICSROperator


def load_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"[ERROR] Config file not found at: '{config_path}'")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def save_checkpoint(filepath: str, model, optimizer, epoch: int, cfg: dict, criterion):
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    stats_dict = {}
    for attr in ["mu_ln_t", "std_ln_t", "mu_ln_rho", "std_ln_rho", "mu_ln_p", "std_ln_p"]:
        if hasattr(criterion, attr):
            stats_dict[attr] = getattr(criterion, attr)

    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": cfg,
        "stats": stats_dict
    }, filepath)
    print(f"[CHECKPOINT] Successfully saved epoch {epoch} state to '{filepath}'", flush=True)


def train_epoch(
    epoch: int, total_epochs: int, model, dataloader, optimizer, criterion, device,
    edge_index, graph_mesh_ops, amsua_op, amsua_obs_err, iasi_op, iasi_obs_err,
    hms_op, hms_obs_err, atms_op, atms_obs_err, cris_op, cris_obs_err,
    seviri_op, seviri_obs_err, gsrasr_op, gsrasr_obs_err, gsrcsr_op, gsrcsr_obs_err,
    ahicsr_op, ahicsr_obs_err, loss_cfg, accum_steps: int = 4, log_batch_freq: int = 50
):
    model.train()
    epoch_losses = {}
    num_batches = len(dataloader)
    optimizer.zero_grad()

    start_time = time.time()
    batch_start_time = time.time()

    print(f"\n[TRAIN] >>> Starting Epoch {epoch:03d}/{total_epochs:03d} | Total Batches: {num_batches} <<<", flush=True)

    for batch_idx, batch_data in enumerate(dataloader):

        if isinstance(batch_data, dict):
            x_batch = batch_data.get('input_trajectory', batch_data.get('background')).to(device)
            y_batch = batch_data.get('target_state', batch_data.get('target')).to(device)
            valid_mask = batch_data.get('valid_mask', None)
            if valid_mask is not None:
                valid_mask = valid_mask.to(device)

            static_topo = batch_data.get('static_topo', None)
            if static_topo is not None:
                static_topo = static_topo.to(device)

            h_3d = batch_data.get('h_3d', None)
            if h_3d is not None:
                h_3d = h_3d.to(device)

            obs_amsua_tb = batch_data.get('obs_amsua_tb', None)
            obs_amsua_mask = batch_data.get('obs_amsua_mask', None)
            obs_iasi_tb = batch_data.get('obs_iasi_tb', None)
            obs_iasi_mask = batch_data.get('obs_iasi_mask', None)
            obs_hms_tb = batch_data.get('obs_hms_tb', None)
            obs_hms_mask = batch_data.get('obs_hms_mask', None)
            obs_atms_tb = batch_data.get('obs_atms_tb', None)
            obs_atms_mask = batch_data.get('obs_atms_mask', None)
            obs_cris_tb = batch_data.get('obs_cris_tb', None)
            obs_cris_mask = batch_data.get('obs_cris_mask', None)
            obs_seviri_tb = batch_data.get('obs_seviri_tb', None)
            obs_seviri_mask = batch_data.get('obs_seviri_mask', None)
            obs_gsrasr_tb = batch_data.get('obs_gsrasr_tb', None)
            obs_gsrasr_mask = batch_data.get('obs_gsrasr_mask', None)
            obs_gsrcsr_tb = batch_data.get('obs_gsrcsr_tb', None)
            obs_gsrcsr_mask = batch_data.get('obs_gsrcsr_mask', None)
            obs_ahicsr_tb = batch_data.get('obs_ahicsr_tb', None)
            obs_ahicsr_mask = batch_data.get('obs_ahicsr_mask', None)
            obs_conv_val = batch_data.get('obs_conv_val', None)
            obs_conv_mask = batch_data.get('obs_conv_mask', None)
        else:
            x_batch = batch_data[0].to(device)
            y_batch = batch_data[1].to(device)
            valid_mask, static_topo, h_3d = None, None, None
            obs_amsua_tb, obs_amsua_mask = None, None
            obs_iasi_tb, obs_iasi_mask = None, None
            obs_hms_tb, obs_hms_mask = None, None
            obs_atms_tb, obs_atms_mask = None, None
            obs_cris_tb, obs_cris_mask = None, None
            obs_seviri_tb, obs_seviri_mask = None, None
            obs_gsrasr_tb, obs_gsrasr_mask = None, None
            obs_gsrcsr_tb, obs_gsrcsr_mask = None, None
            obs_ahicsr_tb, obs_ahicsr_mask = None, None
            obs_conv_val, obs_conv_mask = None, None

        # GPU Input Sanitization
        x_batch = torch.nan_to_num(x_batch, nan=0.0, posinf=10.0, neginf=-10.0)
        y_batch = torch.nan_to_num(y_batch, nan=0.0, posinf=10.0, neginf=-10.0)
        if static_topo is not None:
            static_topo = torch.nan_to_num(static_topo, nan=0.0, posinf=1.0, neginf=0.0)

        # GNN Forward Pass
        pred = model(x_batch, edge_index, static_topo=static_topo)
        pred = torch.nan_to_num(pred, nan=0.0, posinf=10.0, neginf=-10.0)

        # 1. Base Physical State Reconstruction Loss
        loss, metrics = criterion(
            pred=pred,
            target=y_batch,
            edge_index=edge_index,
            graph_mesh_ops=graph_mesh_ops,
            valid_mask=valid_mask
        )
        total_loss = loss

        # Un-normalize physical profile fields for satellite forward operators
        std_t = getattr(criterion, "std_ln_t", 1.0)
        mu_t = getattr(criterion, "mu_ln_t", 0.0)
        std_p = getattr(criterion, "std_ln_p", 1.0)
        mu_p = getattr(criterion, "mu_ln_p", 0.0)

        ln_T_phys = pred[:, 0, :, :] * std_t + mu_t
        ln_p_phys = pred[:, 6, :, :] * std_p + mu_p

        t_k = torch.clamp(torch.exp(ln_T_phys), min=180.0, max=330.0)
        p_hpa = torch.clamp(torch.exp(ln_p_phys) / 100.0, min=0.01, max=1050.0)
        p_pa = p_hpa * 100.0

        # 2. Conventional Observation Loss
        w_conv = loss_cfg.get("w_conv", 0.05)
        if w_conv > 0.0 and obs_conv_val is not None:
            conv_val = torch.nan_to_num(obs_conv_val.to(device), nan=0.0)
            if obs_conv_mask is not None:
                conv_m = torch.nan_to_num(obs_conv_mask.to(device), nan=0.0)
                loss_conv = torch.sum(((pred - conv_val) ** 2) * conv_m) / (torch.sum(conv_m) + 1e-8)
            else:
                loss_conv = F.mse_loss(pred, conv_val)
        else:
            loss_conv = torch.tensor(0.0, device=device)

        total_loss += (w_conv * loss_conv)
        metrics["loss_conv"] = loss_conv.item()

        op_kwargs = {"h_3d": h_3d} if h_3d is not None else {}

        # ---------------------------------------------------------------------
        # CONDITIONAL SATELLITE RADIANCE LOSSES (Only run if weight > 0.0)
        # ---------------------------------------------------------------------
        # AMSU-A
        w_rad_amsua = loss_cfg.get("w_rad_amsua", loss_cfg.get("w_rad", 0.0))
        if w_rad_amsua > 0.0 and obs_amsua_tb is not None:
            tb_sim = torch.nan_to_num(amsua_op(t_k, p_hpa, **op_kwargs), nan=240.0)
            tb_obs = torch.nan_to_num(obs_amsua_tb.to(device), nan=240.0)
            if tb_sim.shape[1] != 15 and tb_sim.shape[2] == 15:
                tb_sim = tb_sim.permute(0, 2, 1)
            if tb_obs.shape[1] != 15 and tb_obs.shape[2] == 15:
                tb_obs = tb_obs.permute(0, 2, 1)
            err = amsua_obs_err.view(1, 15, 1)
            innov = (tb_obs - tb_sim) / err
            if obs_amsua_mask is not None:
                m = torch.nan_to_num(obs_amsua_mask.to(device), nan=0.0)
                if m.shape[1] != 15 and m.shape[2] == 15:
                    m = m.permute(0, 2, 1)
                loss_rad_amsua = torch.sum((innov ** 2) * m) / (15.0 * torch.sum(m) + 1e-8)
            else:
                loss_rad_amsua = torch.mean(innov ** 2) / 15.0
        else:
            loss_rad_amsua = torch.tensor(0.0, device=device)
        total_loss += (w_rad_amsua * loss_rad_amsua)
        metrics["loss_rad_amsua"] = loss_rad_amsua.item()

        # IASI
        w_rad_iasi = loss_cfg.get("w_rad_iasi", 0.0)
        if w_rad_iasi > 0.0 and obs_iasi_tb is not None:
            tb_sim = torch.nan_to_num(iasi_op(t_k, p_pa, **op_kwargs), nan=240.0)
            tb_obs = torch.nan_to_num(obs_iasi_tb.to(device), nan=240.0)
            if tb_obs.shape[1] != 30 and tb_obs.shape[2] == 30:
                tb_obs = tb_obs.permute(0, 2, 1)
            if tb_sim.shape[1] != 30 and tb_sim.shape[2] == 30:
                tb_sim = tb_sim.permute(0, 2, 1)
            err = iasi_obs_err.view(1, 30, 1)
            innov = (tb_obs - tb_sim) / err
            if obs_iasi_mask is not None:
                m = torch.nan_to_num(obs_iasi_mask.to(device), nan=0.0)
                if m.shape[1] != 30 and m.shape[2] == 30:
                    m = m.permute(0, 2, 1)
                loss_rad_iasi = torch.sum((innov ** 2) * m) / (30.0 * torch.sum(m) + 1e-8)
            else:
                loss_rad_iasi = torch.mean(innov ** 2) / 30.0
        else:
            loss_rad_iasi = torch.tensor(0.0, device=device)
        total_loss += (w_rad_iasi * loss_rad_iasi)
        metrics["loss_rad_iasi"] = loss_rad_iasi.item()

        metrics["loss_total"] = total_loss.item()

        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"  [WARNING] Batch {batch_idx + 1}/{num_batches} yielded non-finite loss. Skipping step...", flush=True)
            optimizer.zero_grad()
            continue

        loss_accum = total_loss / accum_steps
        loss_accum.backward()

        if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == num_batches:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        if not epoch_losses:
            epoch_losses = {k: 0.0 for k in metrics.keys()}

        for k, v in metrics.items():
            epoch_losses[k] += v / num_batches

        # Real-time progress output flushing
        if (batch_idx + 1) % log_batch_freq == 0 or (batch_idx + 1) == num_batches:
            elapsed_batch_time = time.time() - batch_start_time
            rate = log_batch_freq / elapsed_batch_time if elapsed_batch_time > 0 else 0.0
            pct = ((batch_idx + 1) / num_batches) * 100.0

            print(
                f"  ├─ [Epoch {epoch:03d}] Batch {batch_idx + 1:04d}/{num_batches:04d} ({pct:5.1f}%) | "
                f"Total Loss: {metrics['loss_total']:.5e} | State MSE: {metrics.get('loss_mse', 0.0):.5e} | "
                f"Speed: {rate:.2f} batch/s",
                flush=True
            )
            batch_start_time = time.time()

    total_epoch_time = time.time() - start_time
    print(f"[TRAIN] Epoch {epoch:03d} Completed in {total_epoch_time:.2f}s", flush=True)
    return epoch_losses


def train_model(cfg: dict):
    paths = cfg["paths"]
    mesh_cfg = cfg["mesh"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    loss_cfg = cfg["loss_weights"]

    print("=" * 80, flush=True)
    print(f"[TRAIN] Initializing AIDA GNN Surrogate Model Training", flush=True)
    print(f"[TRAIN] Epochs: {train_cfg['epochs']} | Batch Size: {train_cfg['batch_size']} | Learning Rate: {train_cfg['lr']}", flush=True)
    print("=" * 80, flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[TRAIN] Operating on compute device: {device}", flush=True)

    zarr_path = paths["zarr_path"]
    if zarr_path and os.path.exists(zarr_path):
        print(f"[TRAIN] Loading Zarr dataset store from: '{zarr_path}'...", flush=True)
        obs_dir = paths.get("obs_dir", None)
        if obs_dir and os.path.exists(obs_dir):
            print(f"[TRAIN] Loading observation directory from: '{obs_dir}'...", flush=True)
            dataset = LogState4DForecastDataset(zarr_path=zarr_path, obs_dir=obs_dir)
        else:
            dataset = LogState4DForecastDataset(zarr_path=zarr_path)
        num_nodes = dataset.num_nodes
        lat_deg = torch.tensor(dataset.latitudes, dtype=torch.float32) if hasattr(dataset, "latitudes") else torch.linspace(-90, 90, num_nodes)
        lon_deg = torch.tensor(dataset.longitudes, dtype=torch.float32) if hasattr(dataset, "longitudes") else torch.linspace(-180, 180, num_nodes)
    else:
        print("[TRAIN] Zarr path not found. Initializing Synthetic Dataset generator...", flush=True)
        dataset = SyntheticAIDAStateDataset(
            num_samples=mesh_cfg["samples"],
            num_nodes=mesh_cfg["num_nodes"],
            num_levels=mesh_cfg["num_levels"]
        )
        num_nodes = mesh_cfg["num_nodes"]
        lat_deg = torch.linspace(-90, 90, num_nodes)
        lon_deg = torch.linspace(-180, 180, num_nodes)

    dataloader = DataLoader(dataset, batch_size=train_cfg["batch_size"], shuffle=True)
    print(f"[TRAIN] DataLoader created with {len(dataloader)} total batches.", flush=True)

    print("[GRAPH] Setting up mesh connectivity edge index...", flush=True)
    edge_index = generate_or_load_edge_index(
        num_nodes=num_nodes,
        edge_file=paths["edges_path"]
    ).to(device)

    print("[GRAPH] Pre-computing M4 mesh sparse differential operators (Grad/Div)...", flush=True)
    Gx_sparse, Gy_sparse = build_icosahedral_differential_operators(
        lat_deg=lat_deg,
        lon_deg=lon_deg,
        edge_index=edge_index.cpu()
    )
    graph_mesh_ops = M4MeshOperators(
        Gx_sparse=Gx_sparse,
        Gy_sparse=Gy_sparse,
        lat_deg=lat_deg
    ).to(device)

    num_levels = mesh_cfg.get("num_levels", 32)
    in_vars = model_cfg.get("in_vars", 14)
    out_vars = model_cfg.get("out_vars", 7)
    num_static_feats = model_cfg.get("num_static_feats", 2)

    print("[MODEL] Building Icosahedral GNN Surrogate Network...", flush=True)
    model = IcosahedralGNNSurrogate(
        in_vars=in_vars,
        out_vars=out_vars,
        num_static_feats=num_static_feats,
        hidden_dim=model_cfg["hidden_dim"],
        num_levels=num_levels,
        num_layers=model_cfg.get("num_layers", 4)
    ).to(device)

    criterion = AIDASurrogateLoss(num_levels=num_levels, **loss_cfg).to(device)

    # Initialize Radiance Operators
    print("[MODEL] Initializing Satellite Radiance Operators...", flush=True)
    amsua_op = DifferentiableAMSUAOperator().to(device)
    amsua_obs_err = torch.tensor([
        2.5, 2.2, 1.2, 0.6, 0.3, 0.25, 0.25, 0.25,
        0.25, 0.35, 0.55, 0.8, 1.2, 1.8, 3.5
    ], dtype=torch.float32, device=device)

    iasi_op = DifferentiableIASIOperator(num_levels=num_levels).to(device)
    iasi_obs_err = iasi_op.obs_errors.to(device)

    hms_op = DifferentiableHMSOperator(num_levels=num_levels).to(device)
    hms_obs_err = hms_op.obs_errors.to(device)

    atms_op = DifferentiableATMSOperator(num_levels=num_levels).to(device)
    atms_obs_err = atms_op.obs_errors.to(device)

    cris_op = DifferentiableCrISOperator(num_levels=num_levels).to(device)
    cris_obs_err = cris_op.obs_errors.to(device)

    seviri_op = DifferentiableSEVIRIOperator(num_levels=num_levels).to(device)
    seviri_obs_err = seviri_op.obs_errors.to(device)

    gsrasr_op = DifferentiableGSRASROperator(num_levels=num_levels).to(device)
    gsrasr_obs_err = gsrasr_op.obs_errors.to(device)

    gsrcsr_op = DifferentiableGSRCSROperator(num_levels=num_levels).to(device)
    gsrcsr_obs_err = gsrcsr_op.obs_errors.to(device)

    ahicsr_op = DifferentiableAHICSROperator(num_levels=num_levels).to(device)
    ahicsr_obs_err = ahicsr_op.obs_errors.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg.get("weight_decay", 1e-4)
    )

    checkpoint_path = paths["checkpoint_path"]
    save_interval = train_cfg.get("save_interval", 5)
    epochs = train_cfg["epochs"]
    log_interval = train_cfg["log_interval"]
    accum_steps = train_cfg.get("accum_steps", 4)
    log_batch_freq = train_cfg.get("log_batch_freq", 50)

    print(f"[TRAIN] Active Satellite Weights: AMSU-A={loss_cfg.get('w_rad_amsua', 0.0)}, "
          f"IASI={loss_cfg.get('w_rad_iasi', 0.0)}, ATMS={loss_cfg.get('w_rad_atms', 0.0)}", flush=True)

    print("\n[TRAIN] Beginning Main Training Loop...", flush=True)

    # Add Cosine Annealing Scheduler with Warmup
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=train_cfg["epochs"],
        eta_min=1e-6
    )

    start_epoch = 1
    resume_ckpt = train_cfg.get("resume_checkpoint", None)
    if resume_ckpt and os.path.exists(resume_ckpt):
        print(f"[TRAIN] Resuming training from checkpoint: '{resume_ckpt}'", flush=True)
        ckpt_data = torch.load(resume_ckpt, map_location=device)
        model.load_state_dict(ckpt_data['model_state_dict'])
        if 'optimizer_state_dict' in ckpt_data:
            optimizer.load_state_dict(ckpt_data['optimizer_state_dict'])
        start_epoch = ckpt_data.get('epoch', 0) + 1
        print(f"[TRAIN] Successfully loaded state! Resuming from Epoch {start_epoch:03d}...", flush=True)

    for epoch in range(1, epochs + 1):
        epoch_losses = train_epoch(
            epoch=epoch,
            total_epochs=epochs,
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            edge_index=edge_index,
            graph_mesh_ops=graph_mesh_ops,
            amsua_op=amsua_op, amsua_obs_err=amsua_obs_err,
            iasi_op=iasi_op, iasi_obs_err=iasi_obs_err,
            hms_op=hms_op, hms_obs_err=hms_obs_err,
            atms_op=atms_op, atms_obs_err=atms_obs_err,
            cris_op=cris_op, cris_obs_err=cris_obs_err,
            seviri_op=seviri_op, seviri_obs_err=seviri_obs_err,
            gsrasr_op=gsrasr_op, gsrasr_obs_err=gsrasr_obs_err,
            gsrcsr_op=gsrcsr_op, gsrcsr_obs_err=gsrcsr_obs_err,
            ahicsr_op=ahicsr_op, ahicsr_obs_err=ahicsr_obs_err,
            loss_cfg=loss_cfg,
            accum_steps=accum_steps,
            log_batch_freq=log_batch_freq
        )

        if epoch % log_interval == 0 or epoch == epochs:
            print(
                f"\n" + "=" * 110 + "\n"
                f" EPOCH {epoch:03d}/{epochs:03d} SUMMARY LOSS BREAKDOWN\n"
                f" " + "-" * 108 + "\n"
                f"  TOTAL LOSS    : {epoch_losses.get('loss_total', 0.0):12.5e} | STATE MSE     : {epoch_losses.get('loss_mse', 0.0):12.5e}\n"
                f"  CONV OBS LOSS : {epoch_losses.get('loss_conv', 0.0):12.5e} | LAPLACIAN P   : {epoch_losses.get('loss_laplacian_p', 0.0):12.5e}\n"
                f"  AMSU-A RAD    : {epoch_losses.get('loss_rad_amsua', 0.0):12.5e} | IASI RAD      : {epoch_losses.get('loss_rad_iasi', 0.0):12.5e}\n"
                f"  DYNAMICS LOSS : {epoch_losses.get('loss_dynamics_total', 0.0):12.5e}\n"
                f"=" * 110 + "\n",
                flush=True
            )

        if epoch % save_interval == 0 or epoch == epochs:
            base_name, ext = os.path.splitext(checkpoint_path)
            epoch_ckpt_path = f"{base_name}_epoch_{epoch:03d}{ext}"
            save_checkpoint(epoch_ckpt_path, model, optimizer, epoch, cfg, criterion)
            save_checkpoint(checkpoint_path, model, optimizer, epoch, cfg, criterion)


def main():
    parser = argparse.ArgumentParser(description="Train AIDA GNN Surrogate Model via YAML Config")
    parser.add_argument("-c", "--config", type=str, default="configs/config.yaml", help="Path to YAML config")
    args = parser.parse_args()
    cfg = load_config(args.config)
    train_model(cfg)


if __name__ == "__main__":
    main()
