"""
GNN backbone with residual message passing and multi-head decoders.

Inputs:
    - data.x: node features [N, F_in]
    - data.edge_index: graph connectivity [2, E]
    - data.pos (optional): raw coordinates [N, 2]
Outputs:
    - dict of per-target predictions (unnormalized tensors matching target order)
    - stacked prediction tensor [N, num_targets]
Purpose:
    Serve as a surrogate model for physical field regression on large semiconductor meshes.
"""

from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, LayerNorm, NNConv


class ResidualGraphBlock(nn.Module):
    """GraphConv + residual skip + optional LayerNorm."""

    def __init__(self, hidden_dim: int, aggr: str = "add", dropout: float = 0.1, layer_norm: bool = True):
        super().__init__()
        self.conv = GraphConv(hidden_dim, hidden_dim, aggr=aggr)
        self.lin_skip = nn.Identity()
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.use_ln = layer_norm
        self.ln = LayerNorm(hidden_dim) if layer_norm else nn.Identity()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.conv(x, edge_index)
        h = self.act(h)
        h = self.dropout(h)
        h = h + self.lin_skip(x)
        if self.use_ln:
            h = self.ln(h)
        return h


class EdgeResidualGraphBlock(nn.Module):
    """NNConv with edge attributes + residual skip + optional LayerNorm."""

    def __init__(self, hidden_dim: int, edge_dim: int, aggr: str = "add", dropout: float = 0.1, layer_norm: bool = True):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim * hidden_dim),
        )
        self.conv = NNConv(hidden_dim, hidden_dim, nn=self.edge_mlp, aggr=aggr)
        self.lin_skip = nn.Identity()
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.use_ln = layer_norm
        self.ln = LayerNorm(hidden_dim) if layer_norm else nn.Identity()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        h = self.conv(x, edge_index, edge_attr)
        h = self.act(h)
        h = self.dropout(h)
        h = h + self.lin_skip(x)
        if self.use_ln:
            h = self.ln(h)
        return h


class MultiHeadDecoder(nn.Module):
    """
    Decoder that aggregates multiple expert heads for a single scalar output.

    Each expert is a small MLP; gating weights are predicted per node to mix them.
    """

    def __init__(self, hidden_dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.heads = heads
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, 1),
                )
                for _ in range(heads)
            ]
        )
        self.gate = nn.Linear(hidden_dim, heads)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # [N, heads]
        weights = F.softmax(self.gate(h), dim=-1)
        # [heads, N, 1] -> [N, heads]
        head_outs = torch.stack([expert(h).squeeze(-1) for expert in self.experts], dim=-1)
        out = (weights * head_outs).sum(dim=-1, keepdim=True)
        return out


class EdgeGNN(nn.Module):
    """Full model combining encoder, residual message passing, and per-target decoders."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        target_names: Iterable[str],
        aggr: str = "add",
        dropout: float = 0.1,
        heads: int = 4,
        layer_norm: bool = True,
        use_edge_attr: bool = False,
        edge_dim: int = 0,
    ):
        super().__init__()
        self.target_names = list(target_names)
        self.use_edge_attr = use_edge_attr and edge_dim > 0
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        if self.use_edge_attr:
            self.blocks = nn.ModuleList(
                [
                    EdgeResidualGraphBlock(hidden_dim, edge_dim=edge_dim, aggr=aggr, dropout=dropout, layer_norm=layer_norm)
                    for _ in range(num_layers)
                ]
            )
        else:
            self.blocks = nn.ModuleList(
                [ResidualGraphBlock(hidden_dim, aggr=aggr, dropout=dropout, layer_norm=layer_norm) for _ in range(num_layers)]
            )
        self.decoders = nn.ModuleDict({name: MultiHeadDecoder(hidden_dim, heads=heads, dropout=dropout) for name in self.target_names})

    def forward(self, data) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        x, edge_index = data.x, data.edge_index
        edge_attr = getattr(data, "edge_attr", None)
        if self.use_edge_attr and edge_attr is None:
            raise ValueError("edge_attr is required but missing in data.")
        h = self.encoder(x)
        for block in self.blocks:
            if self.use_edge_attr:
                h = block(h, edge_index, edge_attr)
            else:
                h = block(h, edge_index)

        outputs: List[torch.Tensor] = []
        out_dict: Dict[str, torch.Tensor] = {}
        for name in self.target_names:
            pred = self.decoders[name](h)
            out_dict[name] = pred
            outputs.append(pred)
        stacked = torch.cat(outputs, dim=-1)
        return out_dict, stacked


def build_model(input_dim: int, target_names: Iterable[str], model_cfg, edge_dim: Optional[int] = None) -> EdgeGNN:
    """Factory helper that reads hyperparameters from ModelConfig."""

    return EdgeGNN(
        input_dim=input_dim,
        hidden_dim=model_cfg.hidden_dim,
        num_layers=model_cfg.num_layers,
        target_names=target_names,
        aggr=model_cfg.message_passing_aggr,
        dropout=model_cfg.dropout,
        heads=model_cfg.heads,
        layer_norm=model_cfg.layer_norm,
        use_edge_attr=model_cfg.use_edge_attr,
        edge_dim=edge_dim if edge_dim is not None else getattr(model_cfg, "edge_dim", 0),
    )
