"""Query-independent residual utility scorer."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn

from .routing import build_token_features


@dataclass
class ResidualUtilityScorerConfig:
    emb_dim: int = 320
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.05


class ResidualUtilityScorer(nn.Module):
    """Predict residual utility from page-only token features."""

    def __init__(self, config: ResidualUtilityScorerConfig) -> None:
        super().__init__()
        self.config = config
        input_dim = int(config.emb_dim) + 5
        layers: list[nn.Module] = []
        dim = input_dim
        for _ in range(max(int(config.num_layers), 1)):
            layers.extend(
                [
                    nn.Linear(dim, int(config.hidden_dim)),
                    nn.GELU(),
                    nn.Dropout(float(config.dropout)),
                ]
            )
            dim = int(config.hidden_dim)
        self.net = nn.Sequential(*layers)
        self.score_head = nn.Linear(dim, 1)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, page: Mapping[str, Any]) -> torch.Tensor:
        features = build_token_features(page, self.device, emb_dim=self.config.emb_dim)
        return self.score_head(self.net(features)).squeeze(-1)

    def save(self, save_dir: str | Path) -> None:
        path = Path(save_dir)
        path.mkdir(parents=True, exist_ok=True)
        torch.save({"scorer": self.state_dict()}, path / "residual_scorer.pt")
        with (path / "residual_scorer_config.json").open("w", encoding="utf-8") as handle:
            json.dump(asdict(self.config), handle, indent=2)

    def load(self, checkpoint_dir: str | Path) -> None:
        state = _torch_load(Path(checkpoint_dir) / "residual_scorer.pt")
        self.load_state_dict(state["scorer"])


def _torch_load(path: str | Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_residual_scorer(
    checkpoint_dir: str | Path,
    device: torch.device | str = "cpu",
) -> ResidualUtilityScorer:
    path = Path(checkpoint_dir)
    with (path / "residual_scorer_config.json").open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    allowed = ResidualUtilityScorerConfig.__dataclass_fields__
    config = ResidualUtilityScorerConfig(**{key: value for key, value in raw.items() if key in allowed})
    model = ResidualUtilityScorer(config).to(device)
    model.load(path)
    model.eval()
    return model
