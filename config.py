"""
Project-wide configuration for edge_GNN.

Inputs:
    None at import time; values are static defaults that can be overridden by CLI args or env vars.
Outputs:
    Dataclass instances describing data paths, model hyperparameters, loss weights, and training knobs.
Usage:
    from config import get_default_configs
    paths, data_cfg, model_cfg, loss_cfg, train_cfg = get_default_configs()
"""

from dataclasses import dataclass, asdict
from typing import List, Tuple


@dataclass
class PathConfig:
    """Filesystem locations for data, artifacts, and logs."""

    # Source HDF5 built by the user's mesh graph script.
    data_h5: str = (
        r"D:\paper_GNN_2025\train_data\dat\edge\dfise_results"
        r"\GaN_FinJFET_halfcell_termin_BV_FLR_14n_1.5int\train_data\meshgraph_data.h5"
    )
    # Where to save checkpoints, normalization stats, and visual artifacts.
    artifact_dir: str = "artifacts"
    checkpoint_path: str = "artifacts/edge_gnn.ckpt"
    normalization_path: str = "artifacts/normalization.json"
    metrics_path: str = "artifacts/metrics.pkl"
    # Logging and plots.
    log_dir: str = "logs"
    plot_dir: str = "outputs"


@dataclass
class DataConfig:
    """Data-related hyperparameters and column definitions."""

    target_columns: List[str] = (
        "ElectrostaticPotential",
        "eDensity",
        "hDensity",
        "SpaceCharge",
        "ElectricField_x",
        "ElectricField_y",
        "DopingConcentration",
    ) # type: ignore
    # Outputs we want the model to regress.
    prediction_targets: List[str] = (
        "ElectrostaticPotential",
        "ElectricField_x",
        "ElectricField_y",
        "SpaceCharge",
    ) # type: ignore
    # Node-wise input feature order produced by the dataset.
    input_features: List[str] = ("x", "y", "doping", "vds") # pyright: ignore[reportAssignmentType]
    # Fourier mapping for coordinates to help capture sharp transitions.
    fourier_features: int = 8
    fourier_sigma: float = 1.0
    use_fourier: bool = True
    # Data split proportions (applied at the sheet level).
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    num_workers: int = 0  # PyG DataLoader workers; set >0 if memory allows.
    pin_memory: bool = False


@dataclass
class ModelConfig:
    """Architecture hyperparameters."""

    hidden_dim: int = 160
    num_layers: int = 8
    message_passing_aggr: str = "add"  # options: add | mean | max
    dropout: float = 0.05
    heads: int = 4  # for attention-style aggregators inside decoder.
    fourier_dropout: float = 0.0
    layer_norm: bool = True
    use_edge_attr: bool = True  # enable edge-aware message passing (NNConv)
    edge_dim: int = 6  # expected edge_attr dimension (dx, dy, dist, inv_dist, dirx, diry)


@dataclass
class LossConfig:
    """Composite loss weights tailored for multi-physics regression."""

    # Main SmoothL1 on normalized outputs.
    l1_weight: float = 1.0
    # L1 on de-normalized physical values to directly fit real magnitudes.
    physical_l1_weight: float = 1.0
    # Relative error to emphasize low-magnitude regions.
    relative_l1_weight: float = 0.3
    relative_eps: float = 1e-5
    # Graph total-variation to encourage smoothness while allowing shocks.
    smoothness_weight: float = 0.05
    # Optional auxiliary penalty aligning ∇V with E fields when available.
    gradient_consistency_weight: float = 0.1


@dataclass
class TrainConfig:
    """Training loop parameters."""

    device: str = "cuda"
    epochs: int = 200
    batch_size: int = 1  # Each sheet is a big graph; keep small batches.
    lr: float = 2e-4
    weight_decay: float = 1e-5
    grad_clip: float = 5.0
    early_stop_patience: int = 30
    save_every: int = 10
    amp: bool = True  # mixed precision toggle.
    seed: int = 42


def get_default_configs() -> Tuple[PathConfig, DataConfig, ModelConfig, LossConfig, TrainConfig]:
    """Return default configs as a tuple."""

    return PathConfig(), DataConfig(), ModelConfig(), LossConfig(), TrainConfig()


def configs_to_dict(paths: PathConfig, data: DataConfig, model: ModelConfig, loss: LossConfig, train: TrainConfig):
    """Serialize configs to a plain dict for logging or JSON export."""

    return {
        "paths": asdict(paths),
        "data": asdict(data),
        "model": asdict(model),
        "loss": asdict(loss),
        "train": asdict(train),
    }

