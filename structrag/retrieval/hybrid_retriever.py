"""
structrag/retrieval/hybrid_retriever.py
----------------------------------------
Fix 3 — Hybrid BM25 + vector retrieval with Reciprocal Rank Fusion.

Symbolic queries like "γc = 1.5" or "εcu2 3.5‰" are exact-match problems
that embeddings handle poorly.  BM25 finds these by term frequency.
We fuse BM25 and cosine similarity scores with Reciprocal Rank Fusion (RRF)
and return the merged, re-ranked result list.

Usage
-----
    from structrag.retrieval.hybrid_retriever import retrieve_hybrid

    results = retrieve_hybrid(
        query="partial safety factor concrete gamma_c",
        n_results=5,
        source_id="ec2_2004_nf",
    )

RRF formula
-----------
    score_rrf(d) = 1/(k + rank_vector(d)) + 1/(k + rank_bm25(d))
    where k=60 (standard constant that dampens rank differences)
"""

from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from structrag.retrieval.retriever import retrieve_hierarchical, DEFAULT_N_RESULTS
from structrag.retrieval.chroma_client import get_collection, COLLECTION_HIERARCHICAL

# ---------------------------------------------------------------------------
# BM25 index — built lazily on first call, cached in memory
# ---------------------------------------------------------------------------
_bm25_index = None
_bm25_chunks: list[dict] = []

CHUNKS_PATH = (
    PROJECT_ROOT / "structrag" / "data" / "chunks" / "ec2_hierarchical_chunks.json"
)


def _build_bm25_index():
    """Load all chunks and build a BM25 index over their text."""
    global _bm25_index, _bm25_chunks
    if _bm25_index is not None:
        return

    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        raise ImportError(
            "rank_bm25 not installed. Run: pip install rank-bm25"
        )

    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"Chunks file not found: {CHUNKS_PATH}")

    with open(CHUNKS_PATH, encoding="utf-8") as f:
        _bm25_chunks = json.load(f)

    # Tokenise: lowercase, split on whitespace + punctuation
    import re

    # French stemmer for better morphological matching
    # armatures/armature, expositions/exposition, contraintes/contrainte etc.
    try:
        from nltk.stem.snowball import FrenchStemmer
        _stemmer = FrenchStemmer()
        def tokenise(text: str) -> list[str]:
            tokens = re.findall(r'[^\s\|\,\;\.\:\!\?\(\)\[\]\{\}]+', text.lower())
            return [_stemmer.stem(t) for t in tokens]
    except ImportError:
        def tokenise(text: str) -> list[str]:
            return re.findall(r'[^\s\|\,\;\.\:\!\?\(\)\[\]\{\}]+', text.lower())

    corpus = [tokenise(c.get("text", "")) for c in _bm25_chunks]
    _bm25_index = BM25Okapi(corpus)
    print(f"[BM25] Index built: {len(_bm25_chunks)} documents (French stemming)")


def _bm25_search(query: str, n: int) -> list[dict]:
    """Return top-n BM25 results as dicts matching retriever output format."""
    _build_bm25_index()

    import re
    # Use same stemmer as the index for consistent matching
    try:
        from nltk.stem.snowball import FrenchStemmer
        _stemmer = FrenchStemmer()
        raw_tokens = re.findall(r'[^\s\|\,\;\.\:\!\?\(\)\[\]\{\}]+', query.lower())
        tokens = [_stemmer.stem(t) for t in raw_tokens]
    except ImportError:
        tokens = re.findall(r'[^\s\|\,\;\.\:\!\?\(\)\[\]\{\}]+', query.lower())
    scores = _bm25_index.get_scores(tokens)

    # Get top-n indices by score
    import heapq
    top_indices = heapq.nlargest(n, range(len(scores)), key=lambda i: scores[i])

    results = []
    for rank, idx in enumerate(top_indices):
        chunk = _bm25_chunks[idx]
        meta = {k: v for k, v in chunk.items() if k != "text"}
        entry = {
            "chunk_id":     chunk.get("chunk_id", ""),
            "text":         chunk.get("text", ""),
            "score":        round(float(scores[idx]), 4),
            "bm25_rank":    rank + 1,
            "metadata":     meta,
            "retrieval":    "bm25",
        }
        # Promote hierarchical metadata fields
        if chunk.get("chunk_type") == "hierarchical":
            entry["ancestor_path"] = chunk.get("ancestor_path", "")
            entry["clause_number"] = chunk.get("clause_number", "")
            entry["level"]         = int(chunk.get("level", -1))
            entry["page_number"]   = int(chunk.get("page_number", 0))
            raw_refs = chunk.get("cross_refs", [])
            entry["cross_refs"] = raw_refs if isinstance(raw_refs, list) else \
                                   [r for r in raw_refs.split(",") if r]
        results.append(entry)

    return results


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

RRF_K = 60   # standard constant


def _rrf_fuse(
    vector_results: list[dict],
    bm25_results:   list[dict],
    n_results:      int,
) -> list[dict]:
    """
    Merge two ranked lists with Reciprocal Rank Fusion.
    Returns the top-n_results items, each annotated with its RRF score
    and which retrievers found it.
    """
    # Build chunk_id -> result map
    all_chunks: dict[str, dict] = {}

    for rank, r in enumerate(vector_results, 1):
        cid = r["chunk_id"]
        if cid not in all_chunks:
            all_chunks[cid] = dict(r)
            all_chunks[cid]["rrf_score"] = 0.0
            all_chunks[cid]["found_by"] = []
        all_chunks[cid]["rrf_score"] += 1.0 / (RRF_K + rank)
        all_chunks[cid]["found_by"].append("vector")
        all_chunks[cid]["vector_rank"] = rank
        all_chunks[cid]["vector_score"] = r.get("score", 0)

    for rank, r in enumerate(bm25_results, 1):
        cid = r["chunk_id"]
        if cid not in all_chunks:
            all_chunks[cid] = dict(r)
            all_chunks[cid]["rrf_score"] = 0.0
            all_chunks[cid]["found_by"] = []
        all_chunks[cid]["rrf_score"] += 1.0 / (RRF_K + rank)
        all_chunks[cid]["found_by"].append("bm25")
        all_chunks[cid]["bm25_rank"] = rank
        all_chunks[cid]["bm25_score"] = r.get("score", 0)

    # Sort by RRF score descending
    ranked = sorted(all_chunks.values(), key=lambda x: x["rrf_score"], reverse=True)

    # Clean up: set unified score = rrf_score, keep top n
    for item in ranked:
        item["score"] = round(item["rrf_score"], 6)
    
    return ranked[:n_results]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def retrieve_hybrid(
    query: str,
    n_results: int = 8,          # k=8 as per Fix 4
    source_id: Optional[str] = None,
    level: Optional[int] = None,
    clause_number: Optional[str] = None,
) -> list[dict]:
    """
    Hybrid BM25 + vector retrieval with RRF re-ranking.

    Automatically expands queries containing exposure class (XC1-XA3) or
    structural class (S1-S6) references to improve BM25 recall on cover
    table lookups (Tableau 4.4N).
    """
    import re as _re

    # Query expansion for table lookups: detect class identifiers
    exposure_classes = _re.findall(r'\b(X[CDSFA][0-9])\b', query, _re.IGNORECASE)
    structural_classes = _re.findall(r'\b(S[1-6])\b', query, _re.IGNORECASE)
    concrete_classes = _re.findall(r'\b(C\d{2}/\d{2})\b', query, _re.IGNORECASE)

    if exposure_classes or structural_classes or concrete_classes:
        extras = " ".join(exposure_classes + structural_classes + concrete_classes)
        bm25_query = f"{query} {extras} enrobage cmin,dur Tableau 4.4N"
    else:
        bm25_query = query

    # Fetch more candidates before re-ranking
    k_fetch = max(n_results * 2, 10)

    # Vector retrieval (uses original query for semantic matching)
    vector_results = retrieve_hierarchical(
        query=query,
        n_results=k_fetch,
        source_id=source_id,
        level=level,
        clause_number=clause_number,
    )

    # BM25 retrieval (uses expanded query for keyword matching)
    try:
        bm25_results = _bm25_search(bm25_query, k_fetch)
    except ImportError:
        return vector_results[:n_results]
    except Exception as e:
        print(f"[BM25] Warning: {e} — falling back to vector only")
        return vector_results[:n_results]

    # Fuse
    fused = _rrf_fuse(vector_results, bm25_results, n_results)
    return fused


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import io
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    test_queries = [
        "partial safety factor concrete gamma_c 1.5",
        "compressive strain epsilon_cu2 parabola rectangle",
        "minimum cover exposure class XC2 XC3",
        "structural classification Table 4.3N",
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        results = retrieve_hybrid(q, n_results=3, source_id="ec2_2004_nf")
        for i, r in enumerate(results, 1):
            found_by = r.get("found_by", ["?"])
            print(f"  [{i}] rrf={r['score']:.4f}  by={found_by}  "
                  f"clause={r.get('clause_number','')}  "
                  f"path={r.get('ancestor_path','')[:60]}")
