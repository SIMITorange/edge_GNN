"""
Data package for edge_GNN.

Purpose:
    Houses dataset loading, normalization, and feature mapping utilities.
Imports:
    - dataset: HDF5 -> PyG Data conversion.
    - normalization: scaling and persistence of stats.
    - fourier: Fourier feature mapping for coordinates.
"""

from .dataset import MeshGraphDataset, collate_graphs
from .normalization import NormalizationProfile, Normalizer
from .fourier import FourierFeatureMapper

__all__ = [
    "MeshGraphDataset",
    "collate_graphs",
    "NormalizationProfile",
    "Normalizer",
    "FourierFeatureMapper",
]

