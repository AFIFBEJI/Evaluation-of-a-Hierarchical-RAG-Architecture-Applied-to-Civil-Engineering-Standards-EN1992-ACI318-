"""
indexer.py
----------
Phase 5 — Indexing and Retrieval (ingestion side)

Reads the chunk JSON files produced by the chunkers and upserts them into
the appropriate ChromaDB collections, with full metadata.

ChromaDB metadata fields stored per chunk
------------------------------------------
Hierarchical chunks:
    source_id, chunk_type, clause_number, level, ancestor_path,
    page_number, token_count, part_index, cross_refs (as comma-separated str)

Flat chunks:
    source_id, chunk_type, start_page, end_page, token_count, part_index

Metadata design decisions
--------------------------
- ChromaDB metadata values must be str, int, float, or bool — no lists.
  cross_refs is stored as a comma-separated string and parsed back on retrieval.
- ancestor_path is stored as metadata AND prepended to the chunk text.
  This dual-storage means you can filter on it AND the embedding captures it.
- clause_number is stored as a string so "6.2.2" sorts / filters correctly.

Batching
--------
ChromaDB performs better with batch upserts.  We use BATCH_SIZE=100 to
avoid memory issues on large corpora while keeping upsert calls reasonable.
"""

from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from structrag.retrieval.chroma_client import (
    get_collection,
    COLLECTION_HIERARCHICAL,
    COLLECTION_FLAT,
    COLLECTION_CHILD,
)

BATCH_SIZE = 100

# ---------------------------------------------------------------------------
# Metadata serialisation helpers
# ---------------------------------------------------------------------------

def _meta_hier(chunk: dict) -> dict[str, Any]:
    """Flatten hierarchical chunk metadata to ChromaDB-compatible types."""
    return {
        "source_id":     chunk.get("source_id", ""),
        "chunk_type":    "hierarchical",
        "clause_number": chunk.get("clause_number") or "",
        "level":         int(chunk.get("level", -1)),
        "ancestor_path": chunk.get("ancestor_path", ""),
        "page_number":   int(chunk.get("page_number", 0)),
        "token_count":   int(chunk.get("token_count", 0)),
        "part_index":    int(chunk.get("part_index", 0)),
        "cross_refs":    ",".join(chunk.get("cross_refs", [])),
    }


def _meta_flat(chunk: dict) -> dict[str, Any]:
    """Flatten flat chunk metadata."""
    return {
        "source_id":   chunk.get("source_id", ""),
        "chunk_type":  "flat",
        "start_page":  int(chunk.get("start_page", 0)),
        "end_page":    int(chunk.get("end_page", 0)),
        "token_count": int(chunk.get("token_count", 0)),
        "part_index":  int(chunk.get("part_index", 0)),
    }


# ---------------------------------------------------------------------------
# Generic batch upsert
# ---------------------------------------------------------------------------

def _upsert_batch(collection, ids, texts, metadatas) -> None:
    collection.upsert(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
    )


def _index_chunks(
    chunks: list[dict],
    collection_name: str,
    meta_fn,
    label: str,
) -> None:
    """Generic indexer: upserts all chunks into the named collection."""
    collection = get_collection(collection_name)
    total = len(chunks)
    print(f"Indexing {total} {label} chunks into '{collection_name}' ...")

    for batch_start in range(0, total, BATCH_SIZE):
        batch = chunks[batch_start: batch_start + BATCH_SIZE]
        ids       = [c["chunk_id"] for c in batch]
        texts     = [c["text"]     for c in batch]
        metadatas = [meta_fn(c)    for c in batch]
        _upsert_batch(collection, ids, texts, metadatas)

        done = min(batch_start + BATCH_SIZE, total)
        print(f"  {done}/{total} upserted ...", end="\r")

    print(f"Done. Collection '{collection_name}' now has "
          f"{collection.count()} documents.")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _meta_child(chunk: dict) -> dict:
    """Flatten child chunk metadata for ChromaDB."""
    return {
        "source_id":       chunk.get("source_id", ""),
        "chunk_type":      "child",
        "parent_chunk_id": chunk.get("parent_chunk_id", ""),
        "clause_number":   chunk.get("clause_number", ""),
        "level":           int(chunk.get("level", -1)),
        "ancestor_path":   chunk.get("ancestor_path", ""),
        "page_number":     int(chunk.get("page_number", 0)),
        "token_count":     int(chunk.get("token_count", 0)),
        "child_type":      chunk.get("child_type", ""),
        "paragraph_tag":   chunk.get("paragraph_tag", ""),
    }


def index_child_chunks(chunks_path: str) -> None:
    """Load child chunks JSON and upsert into ChromaDB ec2_child collection."""
    with open(chunks_path, encoding="utf-8") as f:
        chunks = json.load(f)
    _index_chunks(chunks, COLLECTION_CHILD, _meta_child, "child")
    """Load hierarchical chunks JSON and upsert into ChromaDB."""
    with open(chunks_path, encoding="utf-8") as f:
        chunks = json.load(f)
    _index_chunks(chunks, COLLECTION_HIERARCHICAL, _meta_hier, "hierarchical")


def index_flat(chunks_path: str) -> None:
    """Load flat chunks JSON and upsert into ChromaDB."""
    with open(chunks_path, encoding="utf-8") as f:
        chunks = json.load(f)
    _index_chunks(chunks, COLLECTION_FLAT, _meta_flat, "flat")


# ---------------------------------------------------------------------------
# Entry point — indexes both collections in sequence
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    base = Path(r"C:\Users\manef\OneDrive\Desktop\stageesprit")
    hier_path = base / "structrag" / "data" / "chunks" / "ec2_hierarchical_chunks.json"
    flat_path = base / "structrag" / "data" / "chunks" / "ec2_flat_chunks.json"

    if not hier_path.exists():
        print(f"ERROR: {hier_path} not found. Run hierarchical_chunker.py first.")
        sys.exit(1)

    if not flat_path.exists():
        print(f"ERROR: {flat_path} not found. Run flat_chunker.py first.")
        sys.exit(1)

    print("=== Indexing hierarchical chunks ===")
    index_hierarchical(str(hier_path))

    print("\n=== Indexing flat chunks ===")
    index_flat(str(flat_path))

    print("\n=== Done. Collections in ChromaDB: ===")
    from structrag.retrieval.chroma_client import list_collections
    for name in list_collections():
        print(f"  {name}")
