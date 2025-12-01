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
    fourier_features: int = 16  # Increased from 8 for better spatial expressivity
    fourier_sigma: float = 0.5  # Decreased from 1.0 for finer frequency detail
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

    hidden_dim: int = 256  # Increased from 128 for better capacity
    num_layers: int = 10   # Increased from 6 for deeper message passing
    message_passing_aggr: str = "add"  # options: add | mean | max
    dropout: float = 0.05  # Reduced from 0.1 for less regularization (more overfitting allowed)
    heads: int = 8  # Increased from 4 for more diverse decoder heads
    fourier_dropout: float = 0.0
    layer_norm: bool = True
    # Additional capacity parameters
    decoder_hidden: int = 256  # Hidden dim in decoder MLPs
    use_residual: bool = True  # Use residual connections throughout


@dataclass
class LossConfig:
    """Composite loss weights tailored for multi-physics regression."""

    # Main SmoothL1 on normalized outputs - increased weight for better base fitting
    l1_weight: float = 2.0  # Increased from 1.0
    # Relative error to emphasize low-magnitude regions - increased
    relative_l1_weight: float = 0.8  # Increased from 0.3
    # Graph total-variation to encourage smoothness while allowing shocks
    smoothness_weight: float = 0.08  # Increased from 0.05 for better spatial consistency
    # Optional auxiliary penalty aligning ∇V with E fields when available
    gradient_consistency_weight: float = 0.15  # Increased from 0.1 for physics constraint
    # Additional loss components for overfitting
    l2_weight: float = 0.0  # No L2 regularization to allow overfitting
    curvature_weight: float = 0.05  # Penalize high curvature for smoother fields


@dataclass
class TrainConfig:
    """Training loop parameters."""

    device: str = "cuda"
    epochs: int = 400  # Increased from 200 for longer training
    batch_size: int = 1  # Each sheet is a big graph; keep small batches.
    lr: float = 5e-4  # Increased from 2e-4 for faster learning
    weight_decay: float = 0.0  # Set to 0 for no L2 regularization (allow overfitting)
    grad_clip: float = 10.0  # Increased from 5.0 to allow larger gradients
    early_stop_patience: int = 60  # Increased from 30 to train longer before stopping
    save_every: int = 10
    amp: bool = True  # mixed precision toggle.
    seed: int = 42
    # New optimization parameters
    use_warmup: bool = True  # Warmup scheduler
    warmup_epochs: int = 10
    use_scheduler: bool = True  # Learning rate scheduler
    scheduler_type: str = "cosine"  # cosine | exponential | linear
    min_lr: float = 1e-5  # Minimum learning rate for scheduler


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

