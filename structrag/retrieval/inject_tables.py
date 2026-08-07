"""
structrag/retrieval/inject_tables.py
-------------------------------------
Injects extracted table nodes into:
  1. ec2_hierarchical_chunks.json  (so BM25 sees them)
  2. ChromaDB ec2_hierarchical collection (so vector search sees them)

Run once after table_extractor.py produces its nodes.
Safe to re-run: uses upsert so duplicates are overwritten.
"""

from __future__ import annotations
import json
import sys
import io
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE       = PROJECT_ROOT
PDF_PATH   = BASE / "FA039724-NF EN 1992-1-1 (octobre 2005) Eurocode 2.pdf"
CHUNKS_PATH = BASE / "structrag" / "data" / "chunks" / "ec2_hierarchical_chunks.json"

from structrag.parsing.table_extractor import extract_tables
from structrag.retrieval.chroma_client import get_collection, COLLECTION_HIERARCHICAL


def node_to_chunk(node) -> dict:
    """Convert a table Node to a chunk dict matching the hierarchical format."""
    cross_refs = node.cross_refs if isinstance(node.cross_refs, list) else []
    return {
        "chunk_id":      node.id,
        "source_id":     node.source_id,
        "chunk_type":    "hierarchical",
        "heading_node_id": node.id,
        "clause_number": node.clause_number or "",
        "level":         node.level,
        "ancestor_path": node.ancestor_path,
        "page_number":   node.page_number,
        "text":          node.text,
        "token_count":   len(node.text.split()),   # rough estimate
        "cross_refs":    cross_refs,
        "part_index":    0,
        "metadata":      node.metadata,
    }


def meta_for_chroma(chunk: dict) -> dict:
    """Metadata dict for ChromaDB (must be str/int/float/bool only)."""
    cross_refs = chunk.get("cross_refs", [])
    return {
        "source_id":     chunk.get("source_id", ""),
        "chunk_type":    "hierarchical",
        "clause_number": chunk.get("clause_number", ""),
        "level":         int(chunk.get("level", 3)),
        "ancestor_path": chunk.get("ancestor_path", ""),
        "page_number":   int(chunk.get("page_number", 0)),
        "token_count":   int(chunk.get("token_count", 0)),
        "part_index":    0,
        "cross_refs":    ",".join(cross_refs) if isinstance(cross_refs, list) else "",
        "table_id":      chunk.get("metadata", {}).get("table_id", ""),
    }


def inject_tables():
    print(f"Extracting tables from PDF...")
    nodes = extract_tables(str(PDF_PATH))
    print(f"Extracted {len(nodes)} table nodes")

    # ── 1. Update chunks JSON ─────────────────────────────────────────────
    print(f"\nLoading existing chunks from {CHUNKS_PATH.name}...")
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        existing_chunks = json.load(f)

    existing_ids = {c["chunk_id"] for c in existing_chunks}
    new_chunks = [node_to_chunk(n) for n in nodes]

    added   = [c for c in new_chunks if c["chunk_id"] not in existing_ids]
    updated = [c for c in new_chunks if c["chunk_id"] in existing_ids]

    # Remove old versions of updated chunks
    existing_chunks = [c for c in existing_chunks if c["chunk_id"] not in {u["chunk_id"] for u in updated}]
    # Append all table chunks
    existing_chunks.extend(new_chunks)

    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(existing_chunks, f, ensure_ascii=False, indent=2)
    print(f"Chunks JSON updated: +{len(added)} new, {len(updated)} updated -> {len(existing_chunks)} total")

    # ── 2. Upsert into ChromaDB ───────────────────────────────────────────
    print(f"\nUpserting into ChromaDB collection '{COLLECTION_HIERARCHICAL}'...")
    collection = get_collection(COLLECTION_HIERARCHICAL)

    ids       = [c["chunk_id"] for c in new_chunks]
    texts     = [c["text"]     for c in new_chunks]
    metadatas = [meta_for_chroma(c) for c in new_chunks]

    collection.upsert(ids=ids, documents=texts, metadatas=metadatas)
    print(f"Upserted {len(ids)} table chunks into ChromaDB")
    print(f"Collection now has {collection.count()} documents total")

    # ── 3. Summary ────────────────────────────────────────────────────────
    print("\nInjected tables:")
    for n in nodes:
        table_id = n.metadata.get("table_id", "?")
        print(f"  Table {table_id:6s}  clause={n.clause_number:10s}  "
              f"chars={len(n.text):5d}  id={n.id}")


if __name__ == "__main__":
    inject_tables()
