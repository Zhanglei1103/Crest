"""Crest model wrapper."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn

from .residual import residual_mask_from_scores, score_with_residual, select_residual_mask
from .residual_scorer import ResidualUtilityScorer, ResidualUtilityScorerConfig
from .routing import CompactRouter, CompactRouterConfig, maxsim_score


@dataclass
class CrestConfig:
    emb_dim: int = 320
    router_hidden_dim: int = 256
    router_layers: int = 2
    router_dropout: float = 0.05
    evidence_bias_max: float = 0.05
    retention: float = 0.75
    min_keep: int = 8
    residual_ratio: float = 0.05
    min_residual_tokens: int = 0
    residual_strategy: str = "auto"
    use_residual_scorer: bool = True
    residual_scorer_hidden_dim: int = 256
    residual_scorer_layers: int = 2
    residual_scorer_dropout: float = 0.05


class CrestModel(nn.Module):
    """Compact retrieval model with candidate-bounded residual repair."""

    def __init__(self, config: CrestConfig) -> None:
        super().__init__()
        self.config = config
        self.router = CompactRouter(
            CompactRouterConfig(
                emb_dim=config.emb_dim,
                hidden_dim=config.router_hidden_dim,
                num_layers=config.router_layers,
                dropout=config.router_dropout,
                evidence_bias_max=config.evidence_bias_max,
            )
        )
        self.residual_scorer: ResidualUtilityScorer | None = None
        if config.use_residual_scorer:
            self.residual_scorer = ResidualUtilityScorer(
                ResidualUtilityScorerConfig(
                    emb_dim=config.emb_dim,
                    hidden_dim=config.residual_scorer_hidden_dim,
                    num_layers=config.residual_scorer_layers,
                    dropout=config.residual_scorer_dropout,
                )
            )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def trainable_parameters(self):
        return self.parameters()

    def score_page(
        self,
        query_emb: torch.Tensor,
        page: Mapping[str, Any],
        retention: float | None = None,
    ) -> dict[str, Any]:
        keep_ratio = self.config.retention if retention is None else float(retention)
        query = torch.as_tensor(query_emb, device=self.device).float()
        doc = torch.as_tensor(page["doc_emb"], device=self.device).float()
        route = self.router(page, retention=keep_ratio, min_keep=self.config.min_keep)
        compact_score = maxsim_score(
            query,
            doc,
            evidence_bias=route["evidence_bias"],
            hard_mask=route["hard_mask"],
        )
        n_original = int(doc.shape[0])
        n_compact = int(route["hard_mask"].sum().item())
        return {
            **route,
            "score": compact_score,
            "compact_score": compact_score.detach(),
            "n_original": n_original,
            "n_compact": n_compact,
            "compact_retention": n_compact / max(n_original, 1),
        }

    def score_pages(
        self,
        query_emb: torch.Tensor,
        pages: Sequence[Mapping[str, Any]],
        retention: float | None = None,
    ) -> dict[str, Any]:
        infos = [self.score_page(query_emb, page, retention=retention) for page in pages]
        return {
            "scores": torch.stack([info["score"] for info in infos]) if infos else torch.empty(0),
            "page_infos": infos,
        }

    def full_score(self, query_emb: torch.Tensor, page: Mapping[str, Any]) -> torch.Tensor:
        query = torch.as_tensor(query_emb, device=self.device).float()
        doc = torch.as_tensor(page["doc_emb"], device=self.device).float()
        return maxsim_score(query, doc)

    def select_residual(
        self,
        query_emb: torch.Tensor | None,
        page: Mapping[str, Any],
        route_info: Mapping[str, Any],
        residual_ratio: float | None = None,
        residual_token_budget: int | None = None,
        strategy: str | None = None,
        random_seed: int = 0,
    ) -> torch.Tensor:
        ratio = self.config.residual_ratio if residual_ratio is None else float(residual_ratio)
        selected_strategy = self.config.residual_strategy if strategy is None else strategy
        if self.residual_scorer is not None and selected_strategy in {"auto", "score"}:
            residual_score = self.residual_scorer(page)
            if selected_strategy == "auto":
                residual_score = 0.65 * residual_score + 0.35 * route_info["importance_score"]
            return residual_mask_from_scores(
                residual_score,
                route_info["hard_mask"],
                residual_ratio=ratio,
                min_residual=self.config.min_residual_tokens,
                residual_token_budget=residual_token_budget,
            )

        return select_residual_mask(
            query_emb,
            page,
            route_info["importance_score"],
            route_info["hard_mask"],
            residual_ratio=ratio,
            min_residual=self.config.min_residual_tokens,
            residual_token_budget=residual_token_budget,
            strategy=selected_strategy,
            random_seed=random_seed,
        )

    def score_page_with_residual(
        self,
        query_emb: torch.Tensor,
        page: Mapping[str, Any],
        route_info: Mapping[str, Any] | None = None,
        retention: float | None = None,
        residual_ratio: float | None = None,
        residual_token_budget: int | None = None,
        strategy: str | None = None,
        random_seed: int = 0,
    ) -> dict[str, Any]:
        info = self.score_page(query_emb, page, retention=retention) if route_info is None else dict(route_info)
        residual_mask = self.select_residual(
            query_emb,
            page,
            info,
            residual_ratio=residual_ratio,
            residual_token_budget=residual_token_budget,
            strategy=strategy,
            random_seed=random_seed,
        )
        query = torch.as_tensor(query_emb, device=self.device).float()
        repaired_score = score_with_residual(
            query,
            page,
            info["hard_mask"],
            residual_mask,
            evidence_bias=info.get("evidence_bias"),
        )
        n_residual = int(residual_mask.sum().item())
        n_original = int(info["n_original"])
        return {
            **info,
            "score": repaired_score,
            "residual_score": repaired_score.detach(),
            "residual_mask": residual_mask,
            "n_residual": n_residual,
            "loaded_retention": (int(info["n_compact"]) + n_residual) / max(n_original, 1),
        }

    def save(self, save_dir: str | Path) -> None:
        path = Path(save_dir)
        path.mkdir(parents=True, exist_ok=True)
        with (path / "config.json").open("w", encoding="utf-8") as handle:
            json.dump({"model": asdict(self.config)}, handle, indent=2)
        torch.save({"router": self.router.state_dict()}, path / "router.pt")
        if self.residual_scorer is not None:
            self.residual_scorer.save(path)

    def load(self, checkpoint_dir: str | Path) -> None:
        path = Path(checkpoint_dir)
        router_state = _torch_load(path / "router.pt")
        self.router.load_state_dict(router_state["router"])
        scorer_path = path / "residual_scorer.pt"
        if self.residual_scorer is not None and scorer_path.exists():
            self.residual_scorer.load(path)


def _torch_load(path: str | Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_crest(checkpoint_dir: str | Path, device: torch.device | str = "cpu") -> CrestModel:
    path = Path(checkpoint_dir)
    with (path / "config.json").open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    model_config = raw.get("model", raw)
    allowed = CrestConfig.__dataclass_fields__
    config = CrestConfig(**{key: value for key, value in model_config.items() if key in allowed})
    model = CrestModel(config).to(device)
    model.load(path)
    model.eval()
    return model
