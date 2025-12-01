"""
HDF5 -> PyG dataset loader.

Inputs:
    - HDF5 file produced by the user's preprocessing script (groups: nXX/{pos, edge_index, fields}).
Outputs:
    - torch_geometric.data.Data objects with node features, targets, and metadata per sheet.
Purpose:
    Handles Vds extraction (max potential), doping as input, Fourier lifting of coordinates, and normalization.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import torch
from torch_geometric.data import Data, Dataset

from .fourier import FourierFeatureMapper
from .normalization import Normalizer


@dataclass
class SampleIndex:
    """Identifies a single sheet within a device group."""

    group: str
    sheet: int
    sheet_name: Optional[str] = None


def _compute_split_indices(n: int, train: float, val: float, seed: int = 42):
    """Return index lists for train/val/test."""

    ids = list(range(n))
    random.Random(seed).shuffle(ids)
    train_n = int(train * n)
    val_n = int(val * n)
    train_ids = ids[:train_n]
    val_ids = ids[train_n : train_n + val_n]
    test_ids = ids[train_n + val_n :]
    return train_ids, val_ids, test_ids


class MeshGraphDataset(Dataset):
    """Dataset that lazily reads graph samples from an HDF5 file."""

    def __init__(
        self,
        h5_path: str,
        target_columns: Sequence[str],
        prediction_targets: Sequence[str],
        input_features: Sequence[str],
        split_indices: Optional[Sequence[int]] = None,
        fourier_mapper: Optional[FourierFeatureMapper] = None,
        normalizer: Optional[Normalizer] = None,
    ):
        """
        Args:
            h5_path: path to meshgraph_data.h5.
            target_columns: column order used when building the HDF5.
            prediction_targets: subset of columns to regress.
            input_features: node-level input feature order (subset of: x, y, doping, vds).
            split_indices: optional subset of sample indices to expose (for train/val/test splits).
            fourier_mapper: optional Fourier encoder applied to normalized coordinates.
            normalizer: optional Normalizer to scale inputs/targets; may be set after init.
        """

        super().__init__(root=".")
        self.h5_path = h5_path
        self.target_columns = list(target_columns)
        self.prediction_targets = list(prediction_targets)
        self.input_features = list(input_features)
        self.target_idx = {c: i for i, c in enumerate(self.target_columns)}
        self.sample_index: List[SampleIndex] = self._build_index()
        if split_indices is not None:
            self.sample_index = [self.sample_index[i] for i in split_indices]
        self.fourier_mapper = fourier_mapper
        self.normalizer = normalizer
        # Lazily opened HDF5 handle; per-worker to avoid reopening every __getitem__.
        self._h5 = None

    # Dataset abstract methods ------------------------------------------------
    @property
    def raw_file_names(self) -> List[str]:
        return []

    @property
    def processed_file_names(self) -> List[str]:
        return []

    def len(self) -> int:
        return len(self.sample_index)

    def get(self, idx: int) -> Data:
        sample = self.sample_index[idx]
        pos, feat_dict, target_dict, edge_index, vds_scalar = self._load_sample(sample)
        pos_normed = pos

        # Normalize and build feature matrix.
        if self.normalizer:
            normed_feats = []
            for key in self.input_features:
                normed_feats.append(self.normalizer.transform(key, feat_dict[key]))
            feat_tensor = torch.stack(normed_feats, dim=-1)

            normed_targets = []
            for key in self.prediction_targets:
                normed_targets.append(self.normalizer.transform(key, target_dict[key]))
            target_tensor = torch.stack(normed_targets, dim=-1)

            # Normalize position for Fourier mapping if needed.
            pos_normed = torch.stack(
                [
                    self.normalizer.transform("x", feat_dict["x"]),
                    self.normalizer.transform("y", feat_dict["y"]),
                ],
                dim=-1,
            )
        else:
            feat_tensor = torch.stack([feat_dict[k] for k in self.input_features], dim=-1)
            target_tensor = torch.stack([target_dict[k] for k in self.prediction_targets], dim=-1)

        # Fourier lifting of coordinates appended to node features.
        if self.fourier_mapper:
            fourier_feats = self.fourier_mapper(pos_normed)
            feat_tensor = torch.cat([feat_tensor, fourier_feats], dim=-1)

        data = Data(
            x=feat_tensor,
            pos=pos,
            edge_index=edge_index,
            y=target_tensor,
            num_nodes=pos.shape[0],
        )
        data.vds = torch.tensor(vds_scalar, dtype=torch.float32)
        data.group = sample.group
        data.sheet = sample.sheet
        data.sheet_name = sample.sheet_name if sample.sheet_name is not None else f"sheet_{sample.sheet}"
        return data

    # Internal helpers -------------------------------------------------------
    def _ensure_h5(self):
        """Open and cache an HDF5 handle per worker/process."""

        if getattr(self, "_h5", None) is None or not getattr(self._h5, "id", None) or not self._h5.id.valid:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def _build_index(self) -> List[SampleIndex]:
        """Scan the HDF5 and enumerate all (group, sheet) pairs."""

        indices: List[SampleIndex] = []
        with h5py.File(self.h5_path, "r") as h5f:
            for group_name, grp in h5f.items():
                fields = grp["fields"]
                num_sheets = fields.shape[0]
                # Sheet names (if stored as attributes) are optional.
                sheet_names = grp.attrs.get("sheet_names", [f"sheet_{i}" for i in range(num_sheets)])
                for sheet_id in range(num_sheets):
                    sheet_label = sheet_names[sheet_id] if sheet_id < len(sheet_names) else f"sheet_{sheet_id}"
                    indices.append(SampleIndex(group=group_name, sheet=sheet_id, sheet_name=sheet_label))
        return indices

    def _load_sample(self, sample: SampleIndex) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor, float]:
        """Load a single sample from disk without normalization."""

        h5f = self._ensure_h5()
        grp = h5f[sample.group]
        pos = torch.from_numpy(grp["pos"][:]).float()  # [N, 2]
        edge_index = torch.from_numpy(grp["edge_index"][:]).long()  # [2, E]
        fields = torch.from_numpy(grp["fields"][sample.sheet])  # [N, F]

        # Map columns.
        feat_dict: Dict[str, torch.Tensor] = {
            "x": pos[:, 0],
            "y": pos[:, 1],
        }
        # Doping as input.
        doping_idx = self.target_idx["DopingConcentration"]
        feat_dict["doping"] = fields[:, doping_idx]

        # Targets.
        target_dict: Dict[str, torch.Tensor] = {}
        for key in self.prediction_targets:
            target_dict[key] = fields[:, self.target_idx[key]]

        vds_scalar = target_dict["ElectrostaticPotential"].max().item()
        feat_dict["vds"] = torch.full_like(feat_dict["x"], fill_value=vds_scalar)

        return pos, feat_dict, target_dict, edge_index, vds_scalar

    def __getstate__(self):
        """Drop open file handles when pickling for DataLoader workers."""

        state = self.__dict__.copy()
        state["_h5"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._h5 = None

    def __del__(self):
        try:
            if getattr(self, "_h5", None) is not None:
                self._h5.close()
        except Exception:
            pass


def build_splits(dataset: MeshGraphDataset, train: float, val: float, seed: int = 42):
    """Create train/val/test subset instances."""

    train_ids, val_ids, test_ids = _compute_split_indices(len(dataset.sample_index), train, val, seed)
    train_ds = MeshGraphDataset(
        dataset.h5_path,
        dataset.target_columns,
        dataset.prediction_targets,
        dataset.input_features,
        split_indices=train_ids,
        fourier_mapper=dataset.fourier_mapper,
        normalizer=dataset.normalizer,
    )
    val_ds = MeshGraphDataset(
        dataset.h5_path,
        dataset.target_columns,
        dataset.prediction_targets,
        dataset.input_features,
        split_indices=val_ids,
        fourier_mapper=dataset.fourier_mapper,
        normalizer=dataset.normalizer,
    )
    test_ds = MeshGraphDataset(
        dataset.h5_path,
        dataset.target_columns,
        dataset.prediction_targets,
        dataset.input_features,
        split_indices=test_ids,
        fourier_mapper=dataset.fourier_mapper,
        normalizer=dataset.normalizer,
    )
    return train_ds, val_ds, test_ds


def fit_normalizer(
    dataset: MeshGraphDataset,
    strategy_map: Dict[str, str],
) -> Normalizer:
    """
    Compute normalization statistics over the dataset (typically the train split).

    Args:
        dataset: MeshGraphDataset without an attached normalizer.
        strategy_map: dict specifying normalization per key.
    Returns:
        Fitted Normalizer instance.
    """

    accum: Dict[str, List[torch.Tensor]] = {}
    keys: List[str] = list(set(dataset.input_features + dataset.prediction_targets))
    for key in keys:
        accum[key] = []

    for sample in dataset.sample_index:
        pos, feat_dict, target_dict, _, vds_scalar = dataset._load_sample(sample)
        # Node-level features.
        accum["x"].append(pos[:, 0])
        accum["y"].append(pos[:, 1])
        accum["doping"].append(feat_dict["doping"])
        accum["ElectrostaticPotential"].append(target_dict["ElectrostaticPotential"])
        accum["ElectricField_x"].append(target_dict["ElectricField_x"])
        accum["ElectricField_y"].append(target_dict["ElectricField_y"])
        accum["SpaceCharge"].append(target_dict["SpaceCharge"])
        # Vds as graph-level scalar (single value per sheet).
        accum["vds"].append(torch.tensor([vds_scalar], dtype=torch.float32))

    stacked = {k: torch.cat(v) for k, v in accum.items()}
    normalizer = Normalizer(strategy_map=strategy_map)
    normalizer.fit(stacked)
    return normalizer


def collate_graphs(batch: List[Data]):
    """Alias for PyG's default collate; provided for clarity."""

    from torch_geometric.data import Batch

    return Batch.from_data_list(batch)

