"""
structrag/retrieval/reranker.py
--------------------------------
Cross-encoder reranking layer.

After hybrid BM25+vector retrieval returns top-k candidates cheaply,
a cross-encoder scores each (query, chunk) pair with a more expensive but
more accurate model and reorders the results.

Why this helps
--------------
Embedding models encode query and document independently — they can miss
fine-grained relevance signals. A cross-encoder sees BOTH together in one
forward pass, giving much better precision at the cost of speed.

For this project: retrieve top-20 with hybrid search, rerank to top-5.
This directly targets the remaining failures where the correct chunk IS in
the top-20 but gets ranked below the cutoff by the embedding model alone.

Model choice
------------
'cross-encoder/ms-marco-MiniLM-L-6-v2'  — fast (6 layers), good accuracy,
  runs on CPU in <100ms for 20 candidates. Good enough for dev/eval.
'BAAI/bge-reranker-large'  — better accuracy, slower (~300ms on CPU).
  Recommended for final evaluation.

The model is lazy-loaded on first call and cached.
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Cached model instance
_reranker_model = None
_reranker_model_name: Optional[str] = None


def _get_reranker(model_name: str):
    """Lazy-load and cache the cross-encoder model."""
    global _reranker_model, _reranker_model_name
    if _reranker_model is None or _reranker_model_name != model_name:
        try:
            from sentence_transformers import CrossEncoder
            _reranker_model = CrossEncoder(model_name)
            _reranker_model_name = model_name
            print(f"[Reranker] Loaded model: {model_name}")
        except Exception as e:
            raise ImportError(
                f"Could not load cross-encoder '{model_name}': {e}\n"
                "Install with: pip install sentence-transformers"
            )
    return _reranker_model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rerank(
    query: str,
    candidates: list[dict],
    top_n: int = 5,
    model_name: str = DEFAULT_RERANKER_MODEL,
) -> list[dict]:
    """
    Rerank a list of retrieved chunks using a cross-encoder.

    Parameters
    ----------
    query      : the user's question
    candidates : retrieved chunk dicts (from hybrid_retriever or retriever)
    top_n      : how many to return after reranking
    model_name : cross-encoder model to use

    Returns
    -------
    Top-n chunks reordered by cross-encoder score, highest first.
    Each result gets a new 'rerank_score' field (higher = more relevant).
    """
    if not candidates:
        return []

    reranker = _get_reranker(model_name)

    # Build (query, text) pairs
    pairs = [(query, c["text"]) for c in candidates]

    # Score all pairs
    scores = reranker.predict(pairs)

    # Attach scores and sort
    for chunk, score in zip(candidates, scores):
        chunk["rerank_score"] = float(score)

    ranked = sorted(candidates, key=lambda x: x.get("rerank_score", 0), reverse=True)
    return ranked[:top_n]


def rerank_with_fallback(
    query: str,
    candidates: list[dict],
    top_n: int = 5,
    model_name: str = DEFAULT_RERANKER_MODEL,
) -> list[dict]:
    """
    Rerank with graceful fallback: if the reranker fails (model not available,
    OOM, etc.), returns the top-n candidates in their original order.
    """
    try:
        return rerank(query, candidates, top_n, model_name)
    except Exception as e:
        print(f"[Reranker] Warning: reranking failed ({e}), using original order")
        return candidates[:top_n]
