"""
Normalization utilities.

Inputs:
    - Arbitrary per-key tensors containing features or targets.
Outputs:
    - Normalized tensors plus a serializable profile for inverse-transform during inference.
Purpose:
    Reduce scale disparities across physical quantities (potential, electric field, charge, doping, Vds).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional

import torch


@dataclass
class NormalizationProfile:
    """Holds statistics for a single scalar feature/target."""

    method: str = "standard"  # standard | robust | minmax | log_standard | symlog | asinh | none
    mean: float = 0.0
    std: float = 1.0
    median: float = 0.0
    iqr: float = 1.0
    min_val: float = 0.0
    max_val: float = 1.0
    eps: float = 1e-6
    # For symlog and asinh
    symlog_scale: float = 1.0  # scale parameter for symlog
    asinh_scale: float = 1.0   # scale parameter for asinh

    def fit(self, values: torch.Tensor):
        """Compute statistics from a 1D tensor."""

        if self.method == "standard":
            self.mean = values.mean().item()
            self.std = values.std(unbiased=False).clamp_min(self.eps).item()
        elif self.method == "robust":
            self.median = values.median().item()
            q1 = torch.quantile(values, 0.25).item()
            q3 = torch.quantile(values, 0.75).item()
            self.iqr = max(self.eps, q3 - q1)
        elif self.method == "minmax":
            self.min_val = values.min().item()
            self.max_val = values.max().item()
        elif self.method == "log_standard":
            logged = self._safe_log(values)
            self.mean = logged.mean().item()
            self.std = logged.std(unbiased=False).clamp_min(self.eps).item()
        elif self.method == "symlog":
            # Symmetric log: handles both positive and negative with extreme ranges
            # scale = mean absolute value for scale-invariant behavior
            abs_vals = values.abs()
            nonzero_vals = abs_vals[abs_vals > self.eps]
            if len(nonzero_vals) > 0:
                self.symlog_scale = nonzero_vals.median().item()
            else:
                self.symlog_scale = 1.0
        elif self.method == "asinh":
            # Inverse hyperbolic sine: smooth everywhere, handles extreme ranges (8+ orders of magnitude)
            # asinh(x) ≈ sign(x)*log(2|x|) for large |x|, ≈ x for small |x|
            # Mathematically perfect reversibility even for 3e6 to 1e-7 range
            abs_vals = values.abs()
            nonzero_vals = abs_vals[abs_vals > self.eps]
            if len(nonzero_vals) > 0:
                # Scale parameter chosen so that typical values are ~ O(1) after transform
                self.asinh_scale = nonzero_vals.median().item()
            else:
                self.asinh_scale = 1.0
        elif self.method == "none":
            pass
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")

    def _safe_log(self, values: torch.Tensor) -> torch.Tensor:
        """Signed log1p to handle positive/negative magnitudes."""

        return torch.sign(values) * torch.log1p(values.abs() + self.eps)

    def _symlog(self, values: torch.Tensor) -> torch.Tensor:
        """Symmetric log: sign(x) * log(1 + |x|/scale)."""
        return torch.sign(values) * torch.log1p(values.abs() / max(self.symlog_scale, self.eps))

    def _asinh(self, values: torch.Tensor) -> torch.Tensor:
        """Inverse hyperbolic sine: asinh(x/scale). Handles 8+ order magnitude ranges perfectly."""
        return torch.asinh(values / max(self.asinh_scale, self.eps))

    def transform(self, values: torch.Tensor) -> torch.Tensor:
        """Normalize values according to the chosen method."""

        if self.method == "standard":
            return (values - self.mean) / self.std
        if self.method == "robust":
            return (values - self.median) / self.iqr
        if self.method == "minmax":
            span = max(self.eps, self.max_val - self.min_val)
            return (values - self.min_val) / span
        if self.method == "log_standard":
            logged = self._safe_log(values)
            return (logged - self.mean) / self.std
        if self.method == "symlog":
            return self._symlog(values)
        if self.method == "asinh":
            return self._asinh(values)
        if self.method == "none":
            return values
        raise ValueError(f"Unknown normalization method: {self.method}")

    def inverse(self, values: torch.Tensor) -> torch.Tensor:
        """Invert the normalization."""

        if self.method == "standard":
            return values * self.std + self.mean
        if self.method == "robust":
            return values * self.iqr + self.median
        if self.method == "minmax":
            span = max(self.eps, self.max_val - self.min_val)
            return values * span + self.min_val
        if self.method == "log_standard":
            logged = values * self.std + self.mean
            return torch.sign(logged) * (torch.expm1(logged.abs()) + self.eps)
        if self.method == "symlog":
            # Inverse: sign(x) * (exp(|x|) - 1) * scale
            return torch.sign(values) * (torch.expm1(values.abs()) * max(self.symlog_scale, self.eps))
        if self.method == "asinh":
            # Inverse: sinh(x) * scale - mathematically perfect reversibility
            return torch.sinh(values) * max(self.asinh_scale, self.eps)
        if self.method == "none":
            return values
        raise ValueError(f"Unknown normalization method: {self.method}")

    def to_dict(self) -> Dict:
        """Serialize to a Python dict."""

        return asdict(self)

    @classmethod
    def from_dict(cls, state: Dict) -> "NormalizationProfile":
        """Create a profile from serialized state."""

        return cls(**state)


class Normalizer:
    """
    Collection of per-key normalization profiles.

    Inputs:
        - strategy_map: dict mapping key -> normalization method.
    Outputs:
        - transform()/inverse() operations keyed by feature/target name.
        - JSON-serializable state_dict for persistence.
    """

    def __init__(self, strategy_map: Optional[Dict[str, str]] = None):
        self.strategy_map: Dict[str, str] = strategy_map or {}
        self.profiles: Dict[str, NormalizationProfile] = {}

    def fit(self, data: Dict[str, torch.Tensor]):
        """Fit all profiles on provided tensors."""

        for key, tensor in data.items():
            method = self.strategy_map.get(key, "standard")
            profile = NormalizationProfile(method=method)
            profile.fit(tensor.flatten())
            self.profiles[key] = profile

    def transform(self, key: str, values: torch.Tensor) -> torch.Tensor:
        if key not in self.profiles:
            raise KeyError(f"No normalization profile found for key '{key}'")
        return self.profiles[key].transform(values)

    def inverse(self, key: str, values: torch.Tensor) -> torch.Tensor:
        if key not in self.profiles:
            raise KeyError(f"No normalization profile found for key '{key}'")
        return self.profiles[key].inverse(values)

    def state_dict(self) -> Dict[str, Dict]:
        """Return serializable state."""

        return {k: p.to_dict() for k, p in self.profiles.items()}

    def load_state_dict(self, state: Dict[str, Dict]):
        """Load saved normalization profiles."""

        self.profiles = {k: NormalizationProfile.from_dict(v) for k, v in state.items()}
        return self
