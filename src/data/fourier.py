"""
Fourier feature mapping for 2D coordinates.

Inputs:
    - coords (torch.Tensor | np.ndarray) of shape [N, 2] with raw x/y positions.
Outputs:
    - torch.Tensor of shape [N, 2 * k] where k = num_features; stacked sin/cos projections.
Purpose:
    Improves representation of sharp field transitions by lifting coordinates into a high-frequency space.
"""

from typing import Optional

import torch


class FourierFeatureMapper:
    """Encodes 2D coordinates with random Fourier features."""

    def __init__(self, num_features: int = 8, sigma: float = 1.0, device: Optional[torch.device] = None):
        """
        Args:
            num_features: number of random frequency bases per axis.
            sigma: scaling for frequency magnitude; higher -> higher frequency coverage.
            device: torch device for internal tensors.
        """

        self.num_features = num_features
        self.sigma = sigma
        self.device = device or torch.device("cpu")
        # Sample frequencies once for reproducibility; caller should set torch.manual_seed.
        self.B = torch.randn((2, num_features), device=self.device) * sigma

    def to(self, device: torch.device):
        """Move internal tensors to a device."""

        self.device = device
        self.B = self.B.to(device)
        return self

    def __call__(self, coords: torch.Tensor) -> torch.Tensor:
        """Apply Fourier features.

        Args:
            coords: [N, 2] tensor.
        Returns:
            [N, 4 * num_features] tensor with sin/cos features.
        """

        if coords.dim() != 2 or coords.size(-1) != 2:
            raise ValueError("coords must have shape [N, 2]")

        coords = coords.to(self.device)
        # [N, 2] x [2, k] -> [N, k]
        projected = (2.0 * torch.pi) * coords @ self.B
        return torch.cat([torch.sin(projected), torch.cos(projected)], dim=-1)
