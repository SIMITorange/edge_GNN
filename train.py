"""
Training entrypoint for the edge-based GNN surrogate.

Inputs:
    - HDF5 mesh dataset (config.paths.data_h5).
    - Configurable hyperparameters in config.py (can override via CLI flags).
Outputs:
    - Model checkpoints, normalization stats, and metric history under artifacts/ and logs/.
Usage:
    python train.py --config config.py
"""

import argparse
import json
import os
import random
import sys
from typing import cast

# Ensure the project root directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from config import get_default_configs
from src.data import FourierFeatureMapper, MeshGraphDataset, build_splits, collate_graphs, fit_normalizer
from src.models import build_model
from src.training import CompositeLoss, Trainer
from torch_geometric.data import Data


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Train edge-based GNN surrogate.")
    parser.add_argument("--data", type=str, default=None, help="Path to meshgraph_data.h5 (overrides config).")
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu override.")
    args, _ = parser.parse_known_args()
    return args


def main():
    args = parse_args()
    paths, data_cfg, model_cfg, loss_cfg, train_cfg = get_default_configs()
    if args.data:
        paths.data_h5 = args.data
    if args.device:
        train_cfg.device = args.device

    # Device selection with CPU compatibility.
    if train_cfg.device.lower() == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "="*80)
    print("🚀 Starting Edge-GNN Training")
    print("="*80)
    print(f"Device: {device}")
    set_seed(train_cfg.seed)
    print(f"Seed: {train_cfg.seed}")

    os.makedirs(paths.artifact_dir, exist_ok=True)
    os.makedirs(paths.log_dir, exist_ok=True)
    os.makedirs(paths.plot_dir, exist_ok=True)

    print("\n📂 Stage 1: Loading Data and Building Dataset")
    print("-" * 80)

    # Fourier mapper for coordinates.
    fourier_mapper = None
    if data_cfg.use_fourier:
        fourier_mapper = FourierFeatureMapper(num_features=data_cfg.fourier_features, sigma=data_cfg.fourier_sigma)
        print(f"✓ Fourier mapper created: {data_cfg.fourier_features} features, σ={data_cfg.fourier_sigma}")

    # Initial dataset without normalization to compute stats.
    print("  Loading HDF5 dataset...")
    base_dataset = MeshGraphDataset(
        h5_path=paths.data_h5,
        target_columns=data_cfg.target_columns,
        prediction_targets=data_cfg.prediction_targets,
        input_features=data_cfg.input_features,
        fourier_mapper=fourier_mapper,
        normalizer=None,
    )
    print(f"✓ Loaded {len(base_dataset)} samples from {paths.data_h5}")

    train_ds, val_ds, test_ds = build_splits(
        base_dataset,
        train=data_cfg.train_split,
        val=data_cfg.val_split,
        seed=train_cfg.seed,
    )
    print(f"✓ Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    # Fit normalization on the training split.
    # Optimized strategy map for better physical field learning
    strategy_map = {
        "x": "minmax",  # Spatial coordinates: preserve domain bounds
        "y": "minmax",
        "doping": "log_standard",  # Log for exponential doping profile
        "vds": "minmax",  # Vds as relative potential
        "ElectrostaticPotential": "robust",  # Robust for outlier-resistant scaling
        "ElectricField_x": "robust",  # Field components: handle sharp peaks
        "ElectricField_y": "robust",
        "SpaceCharge": "robust",  # SpaceCharge: similar sharp features
    }
    print("\n📊 Stage 2: Fitting Normalization")
    print("-" * 80)
    print(f"  Normalization strategy: {strategy_map}")
    normalizer = fit_normalizer(train_ds, strategy_map=strategy_map)
    print(f"✓ Normalization fitted on {len(train_ds)} training samples")

    # Attach normalizer and Fourier mapper to all splits.
    train_ds.normalizer = normalizer
    val_ds.normalizer = normalizer
    test_ds.normalizer = normalizer
    train_ds.fourier_mapper = fourier_mapper
    val_ds.fourier_mapper = fourier_mapper
    test_ds.fourier_mapper = fourier_mapper

    # Determine input dimension from first sample to ensure correct feature count
    # This accounts for base features + Fourier features if they are applied
    sample_data = cast(Data, train_ds[0])
    input_dim = int(sample_data.x.shape[1])  # type: ignore
    
    print("\n🏗️  Stage 3: Building Model")
    print("-" * 80)
    print(f"  Input dimension: {input_dim}")
    print(f"  Hidden dimension: {model_cfg.hidden_dim}")
    print(f"  Number of layers: {model_cfg.num_layers}")
    print(f"  Decoder heads: {model_cfg.heads}")
    
    model = build_model(input_dim=input_dim, target_names=data_cfg.prediction_targets, model_cfg=model_cfg)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Model built: {total_params:,} total parameters ({trainable_params:,} trainable)")

    loss_fn = CompositeLoss(
        target_order=data_cfg.prediction_targets,
        l1_weight=loss_cfg.l1_weight,
        relative_l1_weight=loss_cfg.relative_l1_weight,
        smoothness_weight=loss_cfg.smoothness_weight,
        gradient_consistency_weight=loss_cfg.gradient_consistency_weight,
        l2_weight=loss_cfg.l2_weight,
        curvature_weight=loss_cfg.curvature_weight,
    )
    print(f"✓ Loss function: L1={loss_cfg.l1_weight}, RelL1={loss_cfg.relative_l1_weight}, "
          f"Smooth={loss_cfg.smoothness_weight}, GradConsist={loss_cfg.gradient_consistency_weight}, "
          f"Curv={loss_cfg.curvature_weight}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    print(f"✓ Optimizer: AdamW(lr={train_cfg.lr}, weight_decay={train_cfg.weight_decay})")

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=data_cfg.num_workers,
        pin_memory=data_cfg.pin_memory,
        collate_fn=collate_graphs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=data_cfg.num_workers,
        pin_memory=data_cfg.pin_memory,
        collate_fn=collate_graphs,
    )

    amp_enabled = train_cfg.amp and device.type == "cuda"

    print("\n⚙️  Stage 4: Initializing Trainer")
    print("-" * 80)
    print(f"  Epochs: {train_cfg.epochs}")
    print(f"  Early stop patience: {train_cfg.early_stop_patience}")
    print(f"  Learning rate scheduler: {train_cfg.scheduler_type} (warmup={train_cfg.warmup_epochs} epochs)")
    print(f"  AMP: {'enabled (CUDA)' if amp_enabled else 'disabled'}")
    
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        log_dir=paths.log_dir,
        amp=amp_enabled,
        grad_clip=train_cfg.grad_clip,
        use_warmup=train_cfg.use_warmup,
        warmup_epochs=train_cfg.warmup_epochs,
        total_epochs=train_cfg.epochs,
        scheduler_type=train_cfg.scheduler_type,
        min_lr=train_cfg.min_lr,
    )
    print(f"✓ Trainer initialized")

    print("\n🚀 Stage 5: Starting Training")
    print("="*80 + "\n")

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=train_cfg.epochs,
        checkpoint_path=paths.checkpoint_path,
        save_every=train_cfg.save_every,
        early_stop_patience=train_cfg.early_stop_patience,
        metrics_path=paths.metrics_path,
    )

    # Persist normalization stats for inference.
    with open(paths.normalization_path, "w") as f:
        json.dump(normalizer.state_dict(), f, indent=2)

    # Save configs for reproducibility.
    with open(os.path.join(paths.artifact_dir, "config_snapshot.json"), "w") as f:
        json.dump(
            {
                "paths": vars(paths),
                "data": vars(data_cfg),
                "model": vars(model_cfg),
                "loss": vars(loss_cfg),
                "train": vars(train_cfg),
            },
            f,
            indent=2,
        )

    print(f"Training complete. Best checkpoint at {paths.checkpoint_path}")
    print(f"Metrics saved to {paths.metrics_path}")


if __name__ == "__main__":
    main()
