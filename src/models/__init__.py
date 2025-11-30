"""
Model package for edge_GNN.

Purpose:
    Provides the GNN backbone and multi-head decoders for physical field regression.
Exports:
    - EdgeGNN: message-passing network with residual blocks.
    - build_model: convenience constructor using config dictionaries.
"""

from .gnn_model import EdgeGNN, build_model

__all__ = ["EdgeGNN", "build_model"]

