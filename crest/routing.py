"""Compact token routing and MaxSim scoring."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


def spatial_xy(n_rows: int, n_cols: int, device: torch.device | str | None = None) -> torch.Tensor:
    """Return normalized row and column coordinates for a page grid."""

    rows = max(int(n_rows), 1)
    cols = max(int(n_cols), 1)
    yy, xx = torch.meshgrid(
        torch.linspace(0.0, 1.0, rows, device=device),
        torch.linspace(0.0, 1.0, cols, device=device),
        indexing="ij",
    )
    return torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=-1)


def _fit_first_dim(values: torch.Tensor, length: int) -> torch.Tensor:
    if int(values.shape[0]) == int(length):
        return values
    if int(values.shape[0]) > int(length):
        return values[:length]
    pad_shape = (int(length) - int(values.shape[0]), *values.shape[1:])
    pad = torch.zeros(pad_shape, dtype=values.dtype, device=values.device)
    return torch.cat([values, pad], dim=0)


def _energy_from_page(page: Mapping[str, Any], length: int, device: torch.device) -> torch.Tensor:
    for key in ("token_energy", "energy"):
        if key in page and page[key] is not None:
            energy = torch.as_tensor(page[key], device=device).float().reshape(-1)
            return _fit_first_dim(energy, length)
    return torch.zeros(length, dtype=torch.float32, device=device)


def _zscore(values: torch.Tensor) -> torch.Tensor:
    if values.numel() <= 1:
        return torch.zeros_like(values)
    std = values.std(unbiased=False)
    if float(std.detach().cpu()) < 1e-8:
        return torch.zeros_like(values)
    return (values - values.mean()) / std.clamp_min(1e-6)


def _rank01(values: torch.Tensor) -> torch.Tensor:
    if values.numel() <= 1:
        return torch.zeros_like(values)
    order = torch.argsort(values, descending=False)
    ranks = torch.empty_like(values, dtype=torch.float32)
    ranks[order] = torch.linspace(0.0, 1.0, int(values.numel()), device=values.device)
    return ranks


def build_token_features(page: Mapping[str, Any], device: torch.device, emb_dim: int | None = None) -> torch.Tensor:
    """Build router features from page embeddings and optional page priors."""

    doc_emb = torch.as_tensor(page["doc_emb"], device=device).float()
    if doc_emb.dim() != 2:
        raise ValueError("page['doc_emb'] must have shape [num_tokens, dim]")
    if emb_dim is not None and int(doc_emb.shape[-1]) != int(emb_dim):
        raise ValueError(f"Expected embedding dim {int(emb_dim)}, got {int(doc_emb.shape[-1])}")

    n_tokens = int(doc_emb.shape[0])
    rows = int(page.get("n_rows", 1))
    cols = int(page.get("n_cols", max(n_tokens, 1)))

    xy_value = page.get("spatial_xy")
    if xy_value is None:
        xy = spatial_xy(rows, cols, device=device)
    else:
        xy = torch.as_tensor(xy_value, device=device).float()
    xy = _fit_first_dim(xy.reshape(-1, 2), n_tokens).clamp(0.0, 1.0)

    energy = _energy_from_page(page, n_tokens, device)
    center_distance = torch.sqrt(((xy - 0.5) ** 2).sum(dim=-1))
    center_prior = 1.0 - center_distance / math.sqrt(0.5)
    prior = torch.stack([_zscore(energy), _rank01(energy), center_prior.clamp(0.0, 1.0)], dim=-1)

    return torch.cat([F.normalize(doc_emb, dim=-1), prior, xy], dim=-1)


def topk_mask(scores: torch.Tensor, retention: float, min_keep: int = 1) -> torch.Tensor:
    """Select the highest-scoring tokens under a fractional retention budget."""

    flat = torch.as_tensor(scores).float().reshape(-1)
    n_tokens = int(flat.numel())
    if n_tokens == 0:
        return torch.zeros_like(flat, dtype=torch.bool)
    keep = int(math.ceil(float(retention) * n_tokens))
    keep = min(n_tokens, max(int(min_keep), keep))
    return topk_mask_by_count(flat, keep)


def topk_mask_by_count(
    scores: torch.Tensor,
    keep: int,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Select exactly `keep` valid tokens whenever possible."""

    flat = torch.as_tensor(scores).float().reshape(-1)
    n_tokens = int(flat.numel())
    keep = max(0, min(int(keep), n_tokens))
    mask = torch.zeros(n_tokens, dtype=torch.bool, device=flat.device)
    if keep == 0 or n_tokens == 0:
        return mask

    score = torch.nan_to_num(flat, nan=-1e9, posinf=1e9, neginf=-1e9)
    if valid_mask is not None:
        valid = torch.as_tensor(valid_mask, device=flat.device).bool().reshape(-1)
        if int(valid.numel()) != n_tokens:
            raise ValueError("valid_mask must have the same length as scores")
        keep = min(keep, int(valid.sum().item()))
        if keep == 0:
            return mask
        score = score.masked_fill(~valid, -1e9)

    indices = torch.topk(score, k=keep, largest=True).indices
    mask[indices] = True
    return mask


@dataclass
class CompactRouterConfig:
    emb_dim: int = 320
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.05
    evidence_bias_max: float = 0.05


class CompactRouter(nn.Module):
    """Query-independent token selector with a bounded scoring bias."""

    def __init__(self, config: CompactRouterConfig) -> None:
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
        self.importance_head = nn.Linear(dim, 1)
        self.bias_head = nn.Linear(dim, 1)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def build_features(self, page: Mapping[str, Any]) -> torch.Tensor:
        return build_token_features(page, self.device, emb_dim=self.config.emb_dim)

    def forward(self, page: Mapping[str, Any], retention: float, min_keep: int = 1) -> dict[str, torch.Tensor]:
        features = self.build_features(page)
        hidden = self.net(features)
        importance = self.importance_head(hidden).squeeze(-1)
        evidence_bias = torch.tanh(self.bias_head(hidden).squeeze(-1)) * float(self.config.evidence_bias_max)
        hard_mask = topk_mask(importance, retention=retention, min_keep=min_keep)
        return {
            "importance_score": importance,
            "evidence_bias": evidence_bias,
            "hard_mask": hard_mask,
        }


def maxsim_score(
    query_emb: torch.Tensor,
    doc_emb: torch.Tensor,
    evidence_bias: torch.Tensor | None = None,
    hard_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Late-interaction MaxSim score for one query-page pair."""

    query = torch.as_tensor(query_emb).float()
    doc = torch.as_tensor(doc_emb, device=query.device).float()
    if query.dim() == 1:
        query = query.unsqueeze(0)
    if doc.dim() != 2:
        raise ValueError("doc_emb must have shape [num_tokens, dim]")
    if doc.numel() == 0:
        return torch.tensor(-1e9, dtype=torch.float32, device=query.device)

    query = F.normalize(query, dim=-1)
    doc = F.normalize(doc, dim=-1)
    sim = query.matmul(doc.T)

    if evidence_bias is not None:
        bias = torch.as_tensor(evidence_bias, device=query.device).float().reshape(1, -1)
        if int(bias.shape[-1]) != int(sim.shape[-1]):
            raise ValueError("evidence_bias must have one value per document token")
        sim = sim + bias

    if hard_mask is not None:
        mask = torch.as_tensor(hard_mask, device=query.device).bool().reshape(-1)
        if int(mask.numel()) != int(sim.shape[-1]):
            raise ValueError("hard_mask must have one value per document token")
        sim = sim.masked_fill(~mask.view(1, -1), -1e4)

    return sim.max(dim=-1).values.mean()
