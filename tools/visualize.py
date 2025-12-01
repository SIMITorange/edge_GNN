"""
Visualization helpers for edge_GNN.

Inputs:
    - artifacts/metrics.pkl (loss curves)
    - outputs/inference.npy (pred/target/pos from inference.py)
    - checkpoint + config for model summary
Outputs:
    - PNG plots saved under outputs/
Usage:
    python tools/visualize.py --metrics artifacts/metrics.pkl --pred outputs/inference.npz
"""
import sys
import argparse
import os

# Ensure the project root directory is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import pickle
from typing import Dict, List, cast

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch_geometric.data import Data, Batch

from config import get_default_configs
from src.models import build_model


def plot_losses(metrics_path: str, out_dir: str):
    with open(metrics_path, "rb") as f:
        history = pickle.load(f)
    train = history.get("train", {})
    val = history.get("val", {})

    def save_curve(key: str, title: str, filename: str):
        if key not in train:
            return
        plt.figure(figsize=(6, 4))
        plt.plot(train[key], label="train")
        if key in val:
            plt.plot(val[key], label="val")
        plt.xlabel("Epoch")
        plt.ylabel(key)
        plt.legend()
        plt.title(title)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, filename)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Saved {title} to {out_path}")

    save_curve("loss/total", "Total Loss", "loss_total.png")
    # Per-target L1 and relative curves if available.
    for key in train.keys():
        if key == "loss/total":
            continue
        save_curve(key, key.replace("/", " "), f"{key.replace('/', '_')}.png")


def plot_field_maps(pred_np: Dict[str, np.ndarray], target_np: Dict[str, np.ndarray], pos: np.ndarray, out_dir: str, keys: List[str]):
    """Scatter plots for predicted vs. true fields on coordinates."""

    os.makedirs(out_dir, exist_ok=True)
    x, y = pos[:, 0], pos[:, 1]
    for key in keys:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        sc1 = plt.scatter(x, y, c=target_np[key], s=1, cmap="viridis")
        plt.colorbar(sc1)
        plt.title(f"GT {key}")
        plt.subplot(1, 2, 2)
        sc2 = plt.scatter(x, y, c=pred_np[key], s=1, cmap="viridis")
        plt.colorbar(sc2)
        plt.title(f"Pred {key}")
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"{key}_comparison.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Saved {key} comparison to {out_path}")


def plot_histograms(pred_np: Dict[str, np.ndarray], target_np: Dict[str, np.ndarray], out_dir: str, keys: List[str]):
    os.makedirs(out_dir, exist_ok=True)
    for key in keys:
        plt.figure(figsize=(6, 4))
        sns.kdeplot(target_np[key].flatten(), label="gt", fill=True)
        sns.kdeplot(pred_np[key].flatten(), label="pred", fill=True)
        plt.title(f"Distribution: {key}")
        plt.legend()
        out_path = os.path.join(out_dir, f"{key}_hist.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Saved {key} histogram to {out_path}")


def export_model_summary(out_path: str, input_dim: int, target_names: List[str]):
    paths, data_cfg, model_cfg, _, _ = get_default_configs()
    model = build_model(input_dim=input_dim, target_names=target_names, model_cfg=model_cfg)
    
    # Count model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Create summary text
    text = f"Model Architecture Summary\n"
    text += f"=" * 50 + "\n"
    text += f"\n{str(model)}\n\n"
    text += f"=" * 50 + "\n"
    text += f"Total Parameters: {total_params:,}\n"
    text += f"Trainable Parameters: {trainable_params:,}\n"
    text += f"Input Dimension: {input_dim}\n"
    text += f"Prediction Targets: {', '.join(target_names)}\n"
    
    with open(out_path, "w") as f:
        f.write(text)
    print(f"Saved model summary to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot training curves and prediction maps.")
    parser.add_argument("--metrics", type=str, default="artifacts/metrics.pkl", help="Path to metrics.pkl from training.")
    parser.add_argument("--pred", type=str, default="outputs/inference.npz", help="Path to npz saved by inference.py.")
    parser.add_argument("--out", type=str, default="outputs", help="Directory to store plots.")
    parser.add_argument("--group", type=str, default="n43", help="Group ID for inference (e.g., n1, n43).")
    parser.add_argument("--sheet", type=int, default=0, help="Sheet index within the group for inference.")
    parser.add_argument("--checkpoint", type=str, default="artifacts/edge_gnn.ckpt", help="Path to model checkpoint.")
    parser.add_argument("--norm", type=str, default="artifacts/normalization.json", help="Path to normalization JSON.")
    args = parser.parse_args()

    # Auto-run inference if prediction file doesn't exist
    if not os.path.exists(args.pred):
        print(f"Prediction file {args.pred} not found. Running inference...")
        try:
            import subprocess
            import sys
            result = subprocess.run(
                [sys.executable, "inference.py", 
                 "--group", args.group, 
                 "--sheet", str(args.sheet),
                 "--checkpoint", args.checkpoint,
                 "--norm", args.norm,
                 "--out", args.pred],
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                print(f"Inference failed with error:\n{result.stderr}")
                print("Continuing with visualization of other outputs...")
            else:
                print(result.stdout)
        except Exception as e:
            print(f"Failed to run inference: {e}")
            print("Continuing with visualization of other outputs...")

    if os.path.exists(args.metrics):
        plot_losses(args.metrics, out_dir=args.out)
    else:
        print(f"Metrics file {args.metrics} not found; skipping loss plot.")

    if os.path.exists(args.pred):
        payload = np.load(args.pred, allow_pickle=True)
        pred = payload["pred"].item()
        target = payload["target"].item()
        pos = payload["pos"]
        plot_field_maps(pred, target, pos, args.out, keys=list(pred.keys()))
        plot_histograms(pred, target, args.out, keys=list(pred.keys()))
    else:
        print(f"Prediction file {args.pred} not found; skipping field plots.")

    # Save architecture summary for documentation.
    # Use the same input_dim calculation as train.py and inference.py
    paths, data_cfg, model_cfg, _, _ = get_default_configs()
    fourier_mapper = None
    if data_cfg.use_fourier:
        from src.data import FourierFeatureMapper
        fourier_mapper = FourierFeatureMapper(num_features=data_cfg.fourier_features, sigma=data_cfg.fourier_sigma)
    
    from src.data import MeshGraphDataset
    dataset = MeshGraphDataset(
        h5_path=paths.data_h5,
        target_columns=data_cfg.target_columns,
        prediction_targets=data_cfg.prediction_targets,
        input_features=data_cfg.input_features,
        fourier_mapper=fourier_mapper,
        normalizer=None,
    )
    sample_data = cast(Data, dataset[0])
    input_dim = sample_data.x.shape[1]
    
    export_model_summary(
        out_path=os.path.join(args.out, "model_summary.txt"),
        input_dim=input_dim,
        target_names=list(get_default_configs()[1].prediction_targets),
    )


if __name__ == "__main__":
    main()
