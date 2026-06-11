# Crest

Crest is a compact multi-vector retrieval model for document pages. This repository contains the public core model code only: compact routing, candidate-bounded residual repair, and stability-aware ranking composition.

The release is intentionally minimal. It does not include datasets, trained checkpoints, experiment scripts, logs, or paper table generation code.

## Contents

```
crest/
  routing.py             compact token routing and MaxSim scoring
  residual.py            residual token selection from dropped tokens
  residual_scorer.py     query-independent residual utility scorer
  model.py               Crest model wrapper and checkpoint I/O
  retriever.py           compact retrieval plus candidate residual repair
```

## Install

```bash
pip install -r requirements.txt
pip install -e .
```

## Expected Inputs

Crest operates on precomputed multi-vector embeddings from a frozen visual retriever. A page record should contain:

```python
{
    "page_id": "page-0",
    "doc_emb": page_embeddings,  # Tensor[num_page_tokens, dim]
}
```

Optional page-side features can improve token routing:

```python
{
    "token_energy": token_energy,  # Tensor[num_page_tokens]
    "spatial_xy": spatial_xy,      # Tensor[num_page_tokens, 2]
    "n_rows": 24,
    "n_cols": 24,
}
```

Queries are represented as multi-vector embeddings with shape `Tensor[num_query_tokens, dim]`.

## Minimal Usage

```python
import torch

from crest import CrestConfig, CrestModel, CrestRetriever

model = CrestModel(CrestConfig(emb_dim=320)).eval()
retriever = CrestRetriever(model)

query_emb = torch.randn(32, 320)
pages = [
    {"page_id": "page-0", "doc_emb": torch.randn(256, 320)},
    {"page_id": "page-1", "doc_emb": torch.randn(256, 320)},
]

result = retriever.search(
    query_emb,
    pages,
    top_k=4,
    candidate_k=8,
    retention=0.75,
)
```

For real use, load trained Crest checkpoints with `load_crest(...)`; randomly initialized modules are useful only for interface tests.

## Method Summary

The compact router predicts page-side token importance and keeps a fixed token budget for corpus-wide retrieval. Residual repair selects a small set of dropped tokens and applies them only to compact top candidates. Stability-aware composition controls whether residual-repaired candidate scores should replace compact scores.
