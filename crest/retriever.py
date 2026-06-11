"""End-to-end Crest retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from .model import CrestModel


@dataclass
class RankedPage:
    rank: int
    page_index: int
    page_id: str
    score: float
    compact_score: float
    used_residual: bool
    n_original: int
    n_compact: int
    n_residual: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "page_index": self.page_index,
            "page_id": self.page_id,
            "score": self.score,
            "compact_score": self.compact_score,
            "used_residual": self.used_residual,
            "n_original": self.n_original,
            "n_compact": self.n_compact,
            "n_residual": self.n_residual,
        }


class CrestRetriever:
    """Search pages with compact corpus-wide scoring and bounded residual repair."""

    def __init__(self, model: CrestModel) -> None:
        self.model = model

    def search(
        self,
        query_emb: torch.Tensor,
        pages: Sequence[Mapping[str, Any]],
        top_k: int = 4,
        candidate_k: int = 8,
        retention: float | None = None,
        residual_ratio: float | None = None,
        residual_token_budget: int | None = None,
        margin_gate: float | None = 0.075,
        use_residual: bool = True,
        random_seed: int = 0,
    ) -> dict[str, Any]:
        if not pages:
            return {"ranking": [], "compact_ranking": [], "details": {}}

        query = torch.as_tensor(query_emb, device=self.model.device).float()
        compact = self.model.score_pages(query, pages, retention=retention)
        compact_scores = compact["scores"].detach()
        page_infos = compact["page_infos"]
        final_scores = compact_scores.clone()

        order = torch.argsort(compact_scores, descending=True)
        candidate_count = min(max(int(candidate_k), 0), int(order.numel()))
        candidate_indices = [int(i) for i in order[:candidate_count].detach().cpu().tolist()]

        compact_margin = None
        if int(order.numel()) >= 2:
            vals = compact_scores[order[:2]]
            compact_margin = float((vals[0] - vals[1]).detach().cpu())

        allow_residual = bool(use_residual and candidate_indices)
        if margin_gate is not None and compact_margin is not None:
            allow_residual = allow_residual and compact_margin <= float(margin_gate)

        repaired_infos: dict[int, dict[str, Any]] = {}
        if allow_residual:
            for local_rank, page_index in enumerate(candidate_indices):
                repaired = self.model.score_page_with_residual(
                    query,
                    pages[page_index],
                    route_info=page_infos[page_index],
                    retention=retention,
                    residual_ratio=residual_ratio,
                    residual_token_budget=residual_token_budget,
                    random_seed=random_seed + local_rank,
                )
                repaired_infos[page_index] = repaired
                final_scores[page_index] = repaired["score"].detach()

        final_order = torch.argsort(final_scores, descending=True)
        ranking = [
            self._ranked_page(
                rank=rank + 1,
                page_index=int(page_index),
                page=pages[int(page_index)],
                final_score=final_scores[int(page_index)],
                compact_info=page_infos[int(page_index)],
                repaired_info=repaired_infos.get(int(page_index)),
            ).as_dict()
            for rank, page_index in enumerate(final_order[: max(int(top_k), 0)].detach().cpu().tolist())
        ]
        compact_ranking = [
            self._ranked_page(
                rank=rank + 1,
                page_index=int(page_index),
                page=pages[int(page_index)],
                final_score=compact_scores[int(page_index)],
                compact_info=page_infos[int(page_index)],
                repaired_info=None,
            ).as_dict()
            for rank, page_index in enumerate(order[: max(int(top_k), 0)].detach().cpu().tolist())
        ]

        details = self._cost_details(query, page_infos, repaired_infos, allow_residual, compact_margin)
        details["candidate_indices"] = candidate_indices
        details["reranked_candidates"] = len(repaired_infos)
        return {
            "ranking": ranking,
            "compact_ranking": compact_ranking,
            "details": details,
        }

    def _ranked_page(
        self,
        rank: int,
        page_index: int,
        page: Mapping[str, Any],
        final_score: torch.Tensor,
        compact_info: Mapping[str, Any],
        repaired_info: Mapping[str, Any] | None,
    ) -> RankedPage:
        compact_score = compact_info.get("compact_score", compact_info.get("score"))
        return RankedPage(
            rank=rank,
            page_index=page_index,
            page_id=str(page.get("page_id", page.get("doc_id", page_index))),
            score=float(final_score.detach().cpu()),
            compact_score=float(torch.as_tensor(compact_score).detach().cpu()),
            used_residual=repaired_info is not None,
            n_original=int(compact_info.get("n_original", 0)),
            n_compact=int(compact_info.get("n_compact", 0)),
            n_residual=0 if repaired_info is None else int(repaired_info.get("n_residual", 0)),
        )

    def _cost_details(
        self,
        query: torch.Tensor,
        page_infos: Sequence[Mapping[str, Any]],
        repaired_infos: Mapping[int, Mapping[str, Any]],
        residual_applied: bool,
        compact_margin: float | None,
    ) -> dict[str, Any]:
        q_tokens = int(query.shape[0]) if query.dim() > 1 else 1
        original_tokens = sum(int(info.get("n_original", 0)) for info in page_infos)
        compact_tokens = sum(int(info.get("n_compact", 0)) for info in page_infos)
        residual_tokens = sum(int(info.get("n_residual", 0)) for info in repaired_infos.values())
        rerank_tokens = sum(
            int(info.get("n_compact", 0)) + int(info.get("n_residual", 0))
            for info in repaired_infos.values()
        )
        compact_comparisons = q_tokens * compact_tokens
        rerank_comparisons = q_tokens * rerank_tokens
        return {
            "residual_applied": bool(residual_applied),
            "compact_top1_margin": compact_margin,
            "query_tokens": q_tokens,
            "original_tokens": original_tokens,
            "compact_tokens": compact_tokens,
            "residual_tokens_loaded": residual_tokens,
            "compact_token_retention": compact_tokens / max(original_tokens, 1),
            "loaded_token_retention": (compact_tokens + residual_tokens) / max(original_tokens, 1),
            "compact_maxsim_comparisons": compact_comparisons,
            "residual_rerank_comparisons": rerank_comparisons,
            "total_maxsim_comparisons": compact_comparisons + rerank_comparisons,
        }
