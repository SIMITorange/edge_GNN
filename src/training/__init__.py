"""
Training utilities for edge_GNN.

Purpose:
    Implements composite losses, training orchestration, evaluation, and checkpoints.
Exports:
    - CompositeLoss: balanced loss tailored for multi-physics targets.
    - Trainer: high-level training loop with logging and checkpointing.
"""

from .losses import CompositeLoss
from .trainer import Trainer

__all__ = ["CompositeLoss", "Trainer"]

