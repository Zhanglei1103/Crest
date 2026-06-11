"""Residual token selection and residual-aware scoring."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from .routing import maxsim_score, topk_mask_by_count


def zscore_like(values: torch.Tensor) -> torch.Tensor:
    vals = torch.as_tensor(values).float()
    if vals.numel() <= 1:
        return torch.zeros_like(vals)
    std = vals.std(unbiased=False)
    if float(std.detach().cpu()) < 1e-8:
        return torch.zeros_like(vals)
    return (vals - vals.mean()) / std.clamp_min(1e-6)


def query_token_contribution_scores(query_emb: torch.Tensor, page: Mapping[str, Any]) -> torch.Tensor:
    """Estimate how much each page token can contribute to a query."""

    doc = torch.as_tensor(page["doc_emb"], device=query_emb.device).float()
    if doc.numel() == 0:
        return torch.empty(0, dtype=torch.float32, device=query_emb.device)
    query = torch.as_tensor(query_emb, device=query_emb.device).float()
    if query.dim() == 1:
        query = query.unsqueeze(0)
    doc = F.normalize(doc, dim=-1)
    query = F.normalize(query, dim=-1)
    return doc.matmul(query.T).max(dim=1).values.float()


def token_margin_utility_scores(
    query_emb: torch.Tensor,
    page: Mapping[str, Any],
    token_energy: torch.Tensor | None = None,
) -> torch.Tensor:
    """Query-aware utility proxy used for optional residual supervision or analysis."""

    doc = torch.as_tensor(page["doc_emb"], device=query_emb.device).float()
    if doc.numel() == 0:
        return torch.empty(0, dtype=torch.float32, device=query_emb.device)
    query = torch.as_tensor(query_emb, device=query_emb.device).float()
    if query.dim() == 1:
        query = query.unsqueeze(0)

    sim = F.normalize(doc, dim=-1).matmul(F.normalize(query, dim=-1).T)
    if sim.shape[1] >= 2:
        top2 = sim.topk(k=2, dim=1).values
        margin = top2[:, 0] - top2[:, 1]
        best = top2[:, 0]
    else:
        best = sim.max(dim=1).values
        margin = torch.zeros_like(best)

    if token_energy is None:
        energy_value = page.get("token_energy", page.get("energy"))
        if energy_value is None:
            energy = torch.zeros(int(doc.shape[0]), dtype=torch.float32, device=query_emb.device)
        else:
            energy = torch.as_tensor(energy_value, device=query_emb.device).float().reshape(-1)
            if int(energy.numel()) != int(doc.shape[0]):
                energy = torch.zeros(int(doc.shape[0]), dtype=torch.float32, device=query_emb.device)
    else:
        energy = torch.as_tensor(token_energy, device=query_emb.device).float().reshape(-1)

    return best + 0.25 * margin + 0.10 * zscore_like(energy)


def residual_keep_count(
    n_tokens: int,
    dropped_count: int,
    residual_ratio: float,
    min_residual: int = 0,
    residual_token_budget: int | None = None,
) -> int:
    if residual_token_budget is None:
        keep = int(round(float(residual_ratio) * int(n_tokens)))
        keep = max(int(min_residual), keep)
    else:
        keep = max(0, int(residual_token_budget))
    return min(keep, max(int(dropped_count), 0))


def residual_mask_from_scores(
    scores: torch.Tensor,
    hard_mask: torch.Tensor,
    residual_ratio: float = 0.05,
    min_residual: int = 0,
    residual_token_budget: int | None = None,
) -> torch.Tensor:
    """Select residual tokens from explicit utility scores."""

    score = torch.as_tensor(scores).float().reshape(-1)
    hard = torch.as_tensor(hard_mask, device=score.device).bool().reshape(-1)
    if int(score.numel()) != int(hard.numel()):
        raise ValueError("scores and hard_mask must have the same length")
    dropped = ~hard
    keep = residual_keep_count(
        n_tokens=int(score.numel()),
        dropped_count=int(dropped.sum().item()),
        residual_ratio=residual_ratio,
        min_residual=min_residual,
        residual_token_budget=residual_token_budget,
    )
    if keep <= 0:
        return torch.zeros_like(hard)
    score = torch.nan_to_num(score, nan=-1e9, posinf=1e9, neginf=-1e9)
    return topk_mask_by_count(score, keep=keep, valid_mask=dropped) & dropped


def select_residual_mask(
    query_emb: torch.Tensor | None,
    page: Mapping[str, Any],
    importance_score: torch.Tensor,
    hard_mask: torch.Tensor,
    residual_ratio: float = 0.05,
    min_residual: int = 0,
    residual_token_budget: int | None = None,
    strategy: str = "auto",
    random_seed: int = 0,
) -> torch.Tensor:
    """Select a small residual token bank from tokens dropped by the compact router."""

    importance = torch.as_tensor(importance_score).float().reshape(-1)
    hard = torch.as_tensor(hard_mask, device=importance.device).bool().reshape(-1)
    if int(importance.numel()) != int(hard.numel()):
        raise ValueError("importance_score and hard_mask must have the same length")

    dropped = ~hard
    keep = residual_keep_count(
        n_tokens=int(importance.numel()),
        dropped_count=int(dropped.sum().item()),
        residual_ratio=residual_ratio,
        min_residual=min_residual,
        residual_token_budget=residual_token_budget,
    )
    if keep <= 0:
        return torch.zeros_like(hard)

    selected_strategy = strategy
    if selected_strategy == "auto":
        selected_strategy = "utility" if query_emb is not None else "importance"

    if selected_strategy == "utility":
        if query_emb is None:
            raise ValueError("utility residual selection requires query_emb")
        score = 0.65 * token_margin_utility_scores(query_emb.to(importance.device), page)
        score = score.to(importance.device) + 0.35 * importance
    elif selected_strategy == "importance":
        score = importance
    elif selected_strategy == "energy":
        value = page.get("token_energy", page.get("energy"))
        if value is None:
            score = torch.zeros_like(importance)
        else:
            score = torch.as_tensor(value, device=importance.device).float().reshape(-1)
            if int(score.numel()) != int(importance.numel()):
                score = torch.zeros_like(importance)
    elif selected_strategy == "random":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(random_seed) % (2**31 - 1))
        score = torch.rand(int(importance.numel()), generator=generator).to(importance.device)
    else:
        valid = "auto, utility, importance, energy, random"
        raise ValueError(f"Unsupported residual strategy: {strategy}. Expected one of: {valid}")

    if int(score.numel()) != int(importance.numel()):
        raise ValueError("Residual score length does not match token count")
    return residual_mask_from_scores(
        score,
        hard,
        residual_ratio=residual_ratio,
        min_residual=min_residual,
        residual_token_budget=keep,
    )


def score_with_residual(
    query_emb: torch.Tensor,
    page: Mapping[str, Any],
    hard_mask: torch.Tensor,
    residual_mask: torch.Tensor,
    evidence_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Score a page with compact tokens plus selected residual tokens."""

    doc = torch.as_tensor(page["doc_emb"], device=query_emb.device).float()
    merged = torch.as_tensor(hard_mask, device=query_emb.device).bool()
    merged = merged | torch.as_tensor(residual_mask, device=query_emb.device).bool()
    bias = None if evidence_bias is None else torch.as_tensor(evidence_bias, device=query_emb.device).float()
    return maxsim_score(query_emb, doc, evidence_bias=bias, hard_mask=merged)


def mask_to_indices(mask: torch.Tensor) -> list[int]:
    return [int(i) for i in torch.nonzero(mask.bool().detach().cpu(), as_tuple=False).view(-1).tolist()]


def indices_to_mask(indices: Iterable[int], n_tokens: int, device: torch.device | str = "cpu") -> torch.Tensor:
    mask = torch.zeros(int(n_tokens), dtype=torch.bool, device=device)
    valid = [int(i) for i in indices if 0 <= int(i) < int(n_tokens)]
    if valid:
        mask[torch.as_tensor(valid, dtype=torch.long, device=device)] = True
    return mask


def merge_indices(*groups: Iterable[int]) -> list[int]:
    merged: set[int] = set()
    for group in groups:
        for index in group:
            merged.add(int(index))
    return sorted(merged)


def residual_stats(page_infos: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    original = 0
    compact = 0
    residual = 0
    for info in page_infos:
        original += int(info.get("n_original", 0))
        compact += int(info.get("n_compact", 0))
        residual += int(info.get("n_residual", 0))
    return {
        "original_tokens": float(original),
        "compact_tokens": float(compact),
        "residual_tokens": float(residual),
        "compact_retention": compact / max(original, 1),
        "residual_retention": residual / max(original, 1),
        "loaded_retention": (compact + residual) / max(original, 1),
    }
