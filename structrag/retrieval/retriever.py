"""
retriever.py
------------
Phase 5 — Retrieval logic (query side)

Provides two retrieval functions that hit the same ChromaDB instance:
  - retrieve_hierarchical()  : queries the ec2_hierarchical collection
  - retrieve_flat()          : queries the ec2_flat collection (baseline)

Both accept the same arguments so the RAGAS evaluation can call them
interchangeably without any other code changes.

Metadata filters
----------------
ChromaDB's `where` clause is used to narrow results before semantic ranking.
All filters are optional — if not supplied the query runs across the full
collection.

Common filters for EC2:
    source_id="ec2_2004_nf"      — restrict to one document
    level=2                       — only subclause-level chunks
    clause_number="6.2.2"        — exact clause lookup

Result format
-------------
Each result is a dict:
    chunk_id       str
    text           str    — the chunk text that will go into the LLM prompt
    score          float  — cosine distance (lower = more similar)
    metadata       dict   — all ChromaDB metadata fields

    For hierarchical results only:
    ancestor_path  str
    clause_number  str
    level          int
    page_number    int
    cross_refs     list[str]
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from structrag.retrieval.chroma_client import (
    get_collection,
    COLLECTION_HIERARCHICAL,
    COLLECTION_FLAT,
    COLLECTION_CHILD,
)

# Default number of results to return
DEFAULT_N_RESULTS = 5


# ---------------------------------------------------------------------------
# Internal helper: run a query and normalise results
# ---------------------------------------------------------------------------

def _query(
    collection_name: str,
    query_text: str,
    n_results: int,
    where: Optional[dict],
) -> list[dict]:
    """
    Execute a ChromaDB similarity query and return a normalised result list.
    """
    collection = get_collection(collection_name, create_if_missing=False)

    kwargs: dict = {
        "query_texts": [query_text],
        "n_results":   min(n_results, collection.count()),
        "include":     ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where

    raw = collection.query(**kwargs)

    results: list[dict] = []
    docs      = raw["documents"][0]
    metas     = raw["metadatas"][0]
    distances = raw["distances"][0]
    ids       = raw["ids"][0]

    for doc, meta, dist, chunk_id in zip(docs, metas, distances, ids):
        entry: dict = {
            "chunk_id": chunk_id,
            "text":     doc,
            "score":    round(dist, 4),
            "metadata": meta,
        }
        # Promote commonly used fields to the top level for convenience
        if meta.get("chunk_type") == "hierarchical":
            entry["ancestor_path"]  = meta.get("ancestor_path", "")
            entry["clause_number"]  = meta.get("clause_number", "")
            entry["level"]          = meta.get("level", -1)
            entry["page_number"]    = meta.get("page_number", 0)
            # cross_refs stored as comma-separated string — parse back to list
            raw_refs = meta.get("cross_refs", "")
            entry["cross_refs"] = [r for r in raw_refs.split(",") if r]
        else:
            entry["start_page"] = meta.get("start_page", 0)
            entry["end_page"]   = meta.get("end_page", 0)

    results.append(entry)
    return results


# Rewrite above — zip loop should append inside the loop, not after
def _query_fixed(
    collection_name: str,
    query_text: str,
    n_results: int,
    where: Optional[dict],
) -> list[dict]:
    collection = get_collection(collection_name, create_if_missing=False)

    count = collection.count()
    if count == 0:
        return []

    kwargs: dict = {
        "query_texts": [query_text],
        "n_results":   min(n_results, count),
        "include":     ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where

    raw       = collection.query(**kwargs)
    docs      = raw["documents"][0]
    metas     = raw["metadatas"][0]
    distances = raw["distances"][0]
    ids       = raw["ids"][0]

    results: list[dict] = []
    for doc, meta, dist, chunk_id in zip(docs, metas, distances, ids):
        entry: dict = {
            "chunk_id": chunk_id,
            "text":     doc,
            "score":    round(float(dist), 4),
            "metadata": meta,
        }
        if meta.get("chunk_type") == "hierarchical":
            entry["ancestor_path"] = meta.get("ancestor_path", "")
            entry["clause_number"] = meta.get("clause_number", "")
            entry["level"]         = int(meta.get("level", -1))
            entry["page_number"]   = int(meta.get("page_number", 0))
            raw_refs = meta.get("cross_refs", "")
            entry["cross_refs"] = [r for r in raw_refs.split(",") if r]
        else:
            entry["start_page"] = int(meta.get("start_page", 0))
            entry["end_page"]   = int(meta.get("end_page", 0))
        results.append(entry)

    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def retrieve_hierarchical(
    query: str,
    n_results: int = DEFAULT_N_RESULTS,
    source_id: Optional[str] = None,
    level: Optional[int] = None,
    clause_number: Optional[str] = None,
) -> list[dict]:
    """
    Retrieve from the hierarchical collection with optional metadata filters.

    Parameters
    ----------
    query         : natural-language question or keyword string
    n_results     : number of results to return
    source_id     : filter to one document (e.g. "ec2_2004_nf")
    level         : filter to one heading level (0=section, 1=clause, 2=subclause)
    clause_number : exact clause lookup (e.g. "6.2.2")
    """
    where: Optional[dict] = None
    filters = []
    if source_id:
        filters.append({"source_id": {"$eq": source_id}})
    if level is not None:
        filters.append({"level": {"$eq": level}})
    if clause_number:
        filters.append({"clause_number": {"$eq": clause_number}})

    if len(filters) == 1:
        where = filters[0]
    elif len(filters) > 1:
        where = {"$and": filters}

    return _query_fixed(COLLECTION_HIERARCHICAL, query, n_results, where)


def retrieve_child(
    query: str,
    n_results: int = DEFAULT_N_RESULTS,
    source_id: Optional[str] = None,
    clause_number: Optional[str] = None,
    child_type: Optional[str] = None,
) -> list[dict]:
    """
    Retrieve from the child collection (small, precise chunks).
    Used with resolve_parents() to implement the small-to-big pattern.
    """
    filters = []
    if source_id:
        filters.append({"source_id": {"$eq": source_id}})
    if clause_number:
        filters.append({"clause_number": {"$eq": clause_number}})
    if child_type:
        filters.append({"child_type": {"$eq": child_type}})

    where = None
    if len(filters) == 1:
        where = filters[0]
    elif len(filters) > 1:
        where = {"$and": filters}

    results = _query_fixed(COLLECTION_CHILD, query, n_results, where)

    # Promote parent_chunk_id to top level for easy access
    for r in results:
        r["parent_chunk_id"] = r["metadata"].get("parent_chunk_id", "")
        r["child_type"]      = r["metadata"].get("child_type", "")
        r["paragraph_tag"]   = r["metadata"].get("paragraph_tag", "")
        r["clause_number"]   = r["metadata"].get("clause_number", "")
        r["ancestor_path"]   = r["metadata"].get("ancestor_path", "")
        r["level"]           = int(r["metadata"].get("level", -1))
        r["page_number"]     = int(r["metadata"].get("page_number", 0))

    return results


def retrieve_flat(
    query: str,
    n_results: int = DEFAULT_N_RESULTS,
    source_id: Optional[str] = None,
) -> list[dict]:
    """
    Retrieve from the flat baseline collection.

    Parameters
    ----------
    query     : natural-language question
    n_results : number of results
    source_id : filter by source document
    """
    where: Optional[dict] = None
    if source_id:
        where = {"source_id": {"$eq": source_id}}

    return _query_fixed(COLLECTION_FLAT, query, n_results, where)


# ---------------------------------------------------------------------------
# Quick smoke-test when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_query = "What is the minimum concrete cover for a beam in exposure class XC2?"
    print(f"Query: {test_query}\n")

    print("=== Hierarchical results ===")
    hier = retrieve_hierarchical(test_query, n_results=3, source_id="ec2_2004_nf")
    for i, r in enumerate(hier, 1):
        print(f"  [{i}] score={r['score']}  clause={r.get('clause_number','')}  "
              f"p={r.get('page_number','')}")
        print(f"       path: {r.get('ancestor_path','')[:70]}")
        print(f"       text: {r['text'][:200]}")
        print()

    print("=== Flat results ===")
    flat = retrieve_flat(test_query, n_results=3, source_id="ec2_2004_nf")
    for i, r in enumerate(flat, 1):
        print(f"  [{i}] score={r['score']}  pages={r.get('start_page','')}-{r.get('end_page','')}")
        print(f"       text: {r['text'][:200]}")
        print()
