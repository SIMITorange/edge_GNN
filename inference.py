"""
Inference script for the trained edge-based GNN.

Inputs:
    - meshgraph_data.h5
    - checkpoint (.ckpt) and normalization.json saved after training
Outputs:
    - Numpy arrays of predicted fields plus optional PNG visualizations (handled by tools/visualize.py)
Usage:
    python inference.py --group n43 --sheet 0 --checkpoint artifacts/edge_gnn.ckpt --norm artifacts/normalization.json
"""

import argparse
import json
import os
import sys
from typing import Dict, cast

# Ensure the project root directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import seaborn as sns

from config import get_default_configs
from src.data import FourierFeatureMapper, MeshGraphDataset, collate_graphs
from src.models import build_model
from src.data.normalization import Normalizer


def load_normalizer(path: str) -> Normalizer:
    with open(path, "r") as f:
        state = json.load(f)
    normalizer = Normalizer()
    normalizer.load_state_dict(state)
    return normalizer


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on a single graph sheet.")
    parser.add_argument("--group", type=str, required=True, help="Group ID (e.g., n1).")
    parser.add_argument("--sheet", type=int, default=0, help="Sheet index within the group.")
    parser.add_argument("--checkpoint", type=str, default="artifacts/edge_gnn.ckpt", help="Path to checkpoint.")
    parser.add_argument("--norm", type=str, default="artifacts/normalization.json", help="Path to normalization JSON.")
    parser.add_argument("--data", type=str, default=None, help="HDF5 path override.")
    parser.add_argument("--out", type=str, default="outputs/inference.npz", help="Output .npz for predictions.")
    parser.add_argument("--device", type=str, default=None, help="Force device: cpu or cuda.")
    args, _ = parser.parse_known_args()
    return args


def main():
    args = parse_args()
    paths, data_cfg, model_cfg, _, train_cfg = get_default_configs()
    if args.data:
        paths.data_h5 = args.data
    if args.device:
        train_cfg.device = args.device

    # Device selection with CPU fallback.
    if train_cfg.device.lower() == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    normalizer = load_normalizer(args.norm)

    fourier_mapper = None
    if data_cfg.use_fourier:
        fourier_mapper = FourierFeatureMapper(num_features=data_cfg.fourier_features, sigma=data_cfg.fourier_sigma)

    dataset = MeshGraphDataset(
        h5_path=paths.data_h5,
        target_columns=data_cfg.target_columns,
        prediction_targets=data_cfg.prediction_targets,
        input_features=data_cfg.input_features,
        split_indices=None,
        fourier_mapper=fourier_mapper,
        normalizer=normalizer,
    )
    # Filter to the requested sample.
    sample_ids = [i for i, s in enumerate(dataset.sample_index) if s.group == args.group and s.sheet == args.sheet]
    if not sample_ids:
        raise ValueError(f"No sample found for group={args.group}, sheet={args.sheet}")
    dataset.sample_index = [dataset.sample_index[sample_ids[0]]]

    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_graphs)

    # Determine input dimension from first sample to ensure correct feature count
    sample_data = cast(Data, dataset[0])
    input_dim = sample_data.x.shape[1]
    
    model = build_model(input_dim=input_dim, target_names=data_cfg.prediction_targets, model_cfg=model_cfg)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred_dict, stacked = model(batch)
            # Denormalize predictions.
            outputs: Dict[str, np.ndarray] = {}
            for i, name in enumerate(data_cfg.prediction_targets):
                outputs[name] = normalizer.inverse(name, pred_dict[name].cpu().squeeze(-1)).numpy()
            targets = {}
            for i, name in enumerate(data_cfg.prediction_targets):
                targets[name] = normalizer.inverse(name, batch.y[:, i].cpu().squeeze(-1)).numpy()

            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            np.savez_compressed(
                args.out,
                pred=outputs,
                target=targets,
                pos=batch.pos.cpu().numpy(),
                edge_index=batch.edge_index.cpu().numpy(),
                group=batch.group.cpu().numpy() if isinstance(batch.group, torch.Tensor) else batch.group,
                sheet=batch.sheet.cpu().numpy() if isinstance(batch.sheet, torch.Tensor) else batch.sheet,
            )
            print(f"Inference complete. Saved predictions to {args.out}")


if __name__ == "__main__":
    main()
