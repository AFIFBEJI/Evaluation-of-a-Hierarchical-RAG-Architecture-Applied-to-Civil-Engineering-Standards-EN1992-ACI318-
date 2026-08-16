"""Build parent-child chunks, index children into ChromaDB, save all."""
import sys, os, io, json, traceback
from pathlib import Path

BASE = Path(r"C:\Users\manef\OneDrive\Desktop\stageesprit")
sys.path.insert(0, str(BASE))

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

out = open(BASE / "pc_index_result.txt", "w", encoding="utf-8")
def w(s=""):
    print(str(s)); out.write(str(s) + "\n")

try:
    from structrag.parsing.hierarchy_schema import load_nodes
    from structrag.chunking.parent_child_chunker import (
        chunk_parent_child, _add_table_nodes_to_parent_child, save_chunks
    )
    from structrag.retrieval.indexer import index_child_chunks
    from structrag.retrieval.chroma_client import (
        get_collection, delete_collection, list_collections, COLLECTION_CHILD
    )

    nodes_path = BASE / "structrag/data/parsed/ec2_2004_nf_nodes.json"
    pdf_path   = BASE / "FA039724-NF EN 1992-1-1 (octobre 2005) Eurocode 2.pdf"
    chunks_dir = BASE / "structrag/data/chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Build chunks ───────────────────────────────────────────────────
    w("Loading nodes...")
    nodes = load_nodes(str(nodes_path))
    w(f"Loaded {len(nodes)} nodes")

    w("\nBuilding parent-child chunks from node tree...")
    child_chunks, parent_chunks = chunk_parent_child(nodes, "ec2_2004_nf")
    w(f"  node tree → {len(parent_chunks)} parents, {len(child_chunks)} children")

    w("\nAdding table nodes as parent-child chunks...")
    child_chunks, parent_chunks = _add_table_nodes_to_parent_child(
        str(pdf_path), "ec2_2004_nf", child_chunks, parent_chunks
    )
    w(f"  after tables → {len(parent_chunks)} parents, {len(child_chunks)} children")

    # Stats
    from collections import Counter
    ct = Counter(c["child_type"] for c in child_chunks)
    pt = [p["token_count"] for p in parent_chunks]
    ctt= [c["token_count"] for c in child_chunks]
    w(f"\nParent tokens: min={min(pt)} avg={sum(pt)//len(pt)} max={max(pt)}")
    w(f"Child  tokens: min={min(ctt)} avg={sum(ctt)//len(ctt)} max={max(ctt)}")
    w(f"Child types  : {dict(ct)}")

    # ── 2. Save to disk ───────────────────────────────────────────────────
    child_path  = chunks_dir / "ec2_child_chunks.json"
    parent_path = chunks_dir / "ec2_parent_chunks.json"
    save_chunks(child_chunks,  str(child_path))
    save_chunks(parent_chunks, str(parent_path))
    w(f"\nSaved child  chunks → {child_path.name}")
    w(f"Saved parent chunks → {parent_path.name}")

    # ── 3. Index children into ChromaDB ──────────────────────────────────
    w(f"\nClearing existing '{COLLECTION_CHILD}' collection if present...")
    existing = list_collections()
    if COLLECTION_CHILD in existing:
        delete_collection(COLLECTION_CHILD)

    w(f"Indexing {len(child_chunks)} child chunks into ChromaDB...")
    index_child_chunks(str(child_path))

    col = get_collection(COLLECTION_CHILD, create_if_missing=False)
    w(f"Collection '{COLLECTION_CHILD}' now has {col.count()} documents")

    # ── 4. Spot check ─────────────────────────────────────────────────────
    w("\n--- Spot check: Table 4.4N row children ---")
    table_rows_44N = [c for c in child_chunks
                      if c["child_type"] == "table_row" and "4_4N" in c["chunk_id"]]
    w(f"Table 4.4N row children: {len(table_rows_44N)}")
    for r in table_rows_44N[:3]:
        w(f"  {r['chunk_id']}  tokens={r['token_count']}")
        w(f"  text: {r['text'][:180]}")

    w("\nDone.")

except Exception as e:
    w(f"ERROR: {e}")
    w(traceback.format_exc())

out.close()
