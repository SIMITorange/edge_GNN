"""
Composite loss tailored for multi-physics graph regression.

Inputs:
    - pred_dict: dict of {target_name: tensor [N, 1]} from the model.
    - target_tensor: stacked ground truth [N, num_targets] in matching order.
    - data: PyG Data containing edge_index and pos for smoothness / consistency terms.
Outputs:
    - total loss scalar and a metrics dict with per-component contributions.
Purpose:
    Balance magnitude disparities via normalized SmoothL1 + relative errors,
    encourage spatial smoothness, and enforce potential-field consistency.
"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _relative_l1(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Relative L1 to emphasize low-magnitude regions."""

    return torch.mean(torch.abs(pred - target) / (torch.abs(target) + eps))


def _edge_total_variation(error: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Graph total variation of residuals along edges."""

    src, dst = edge_index
    diff = error[src] - error[dst]
    return diff.abs().mean()


def _laplacian_smoothness(pred: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Encourage smoothness via Laplacian regularization (second-order smoothness)."""

    src, dst = edge_index
    # For each node, compute the difference between its value and neighbors
    node_vals = pred.squeeze(-1)  # [N]
    diff = (node_vals[src] - node_vals[dst]) ** 2
    return diff.mean()


def _gradient_consistency(
    pred_potential: torch.Tensor,
    pred_ex: torch.Tensor,
    pred_ey: torch.Tensor,
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Aligns predicted electric field with negative gradient of predicted potential.

    Uses edge-wise finite differences: -dV/ds should match field projection along each edge.
    """

    src, dst = edge_index
    dpos = pos[dst] - pos[src]  # [E, 2]
    dp = pred_potential[dst] - pred_potential[src]  # [E, 1]
    ds = torch.linalg.norm(dpos, dim=-1, keepdim=True).clamp_min(eps)  # [E, 1]
    grad_proj = -dp / ds  # [E, 1]

    field_proj = (pred_ex[src] * dpos[:, 0:1] + pred_ey[src] * dpos[:, 1:2]) / ds
    return F.smooth_l1_loss(field_proj, grad_proj)


class CompositeLoss(nn.Module):
    """Balanced loss that combines SmoothL1, relative error, smoothness, and consistency terms."""

    def __init__(
        self,
        target_order: Iterable[str],
        l1_weight: float = 1.0,
        relative_l1_weight: float = 0.3,
        smoothness_weight: float = 0.05,
        gradient_consistency_weight: float = 0.1,
        l2_weight: float = 0.0,
        curvature_weight: float = 0.0,
    ):
        super().__init__()
        self.target_order = list(target_order)
        self.weights = {
            "l1": l1_weight,
            "rel": relative_l1_weight,
            "smooth": smoothness_weight,
            "grad_consistency": gradient_consistency_weight,
            "l2": l2_weight,
            "curvature": curvature_weight,
        }

    def forward(self, pred_dict: Dict[str, torch.Tensor], target_tensor: torch.Tensor, data) -> Tuple[torch.Tensor, Dict[str, float]]:
        metrics: Dict[str, float] = {}
        total_loss = 0.0

        # Main + relative terms per target.
        for i, name in enumerate(self.target_order):
            y_true = target_tensor[:, i : i + 1]
            y_pred = pred_dict[name]
            l1 = F.smooth_l1_loss(y_pred, y_true)
            rel = _relative_l1(y_pred, y_true)
            tv = _edge_total_variation(y_pred - y_true, data.edge_index)

            metrics[f"{name}/l1"] = l1.item()
            metrics[f"{name}/rel"] = rel.item()
            metrics[f"{name}/tv"] = tv.item()

            total_loss = total_loss + self.weights["l1"] * l1 + self.weights["rel"] * rel + self.weights["smooth"] * tv

            # Add curvature regularization for extra smoothness
            if self.weights["curvature"] > 0:
                curv = _laplacian_smoothness(y_pred, data.edge_index)
                metrics[f"{name}/curv"] = curv.item()
                total_loss = total_loss + self.weights["curvature"] * curv

        # Physics-inspired consistency between potential and field components.
        if self.weights["grad_consistency"] > 0:
            try:
                grad_loss = _gradient_consistency(
                    pred_potential=pred_dict["ElectrostaticPotential"],
                    pred_ex=pred_dict["ElectricField_x"],
                    pred_ey=pred_dict["ElectricField_y"],
                    pos=data.pos,
                    edge_index=data.edge_index,
                )
                metrics["consistency/grad"] = grad_loss.item()
                total_loss = total_loss + self.weights["grad_consistency"] * grad_loss
            except KeyError:
                metrics["consistency/grad"] = 0.0

        metrics["loss/total"] = float(total_loss)
        return total_loss, metrics

