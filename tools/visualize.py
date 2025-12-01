"""
Visualization helpers for edge_GNN.

Inputs:
    - artifacts/metrics.pkl (loss curves)
    - outputs/inference.npy (pred/target/pos from inference.py)
    - checkpoint + config for model summary
Outputs:
    - PNG plots saved under outputs/
Usage:
    python tools/visualize.py --metrics artifacts/metrics.pkl --pred outputs/inference.npy
"""

import argparse
import os
import pickle
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torchinfo import summary

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
    from torch_geometric.data import Data

    dummy = Data(
        x=torch.randn((4, input_dim)),
        edge_index=torch.tensor([[0, 1, 2, 3, 0, 2], [1, 0, 3, 2, 2, 0]], dtype=torch.long),
        pos=torch.randn((4, 2)),
    )
    text = summary(model, input_data=(dummy,), verbose=0).to_string()
    with open(out_path, "w") as f:
        f.write(text)
    print(f"Saved model summary to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot training curves and prediction maps.")
    parser.add_argument("--metrics", type=str, default="artifacts/metrics.pkl", help="Path to metrics.pkl from training.")
    parser.add_argument("--pred", type=str, default="outputs/inference.npy", help="Path to npy saved by inference.py.")
    parser.add_argument("--out", type=str, default="outputs", help="Directory to store plots.")
    args = parser.parse_args()

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
    export_model_summary(
        out_path=os.path.join(args.out, "model_summary.txt"),
        input_dim=len(get_default_configs()[1].input_features) + 4 * get_default_configs()[1].fourier_features,
        target_names=list(get_default_configs()[1].prediction_targets),
    )


if __name__ == "__main__":
    main()
