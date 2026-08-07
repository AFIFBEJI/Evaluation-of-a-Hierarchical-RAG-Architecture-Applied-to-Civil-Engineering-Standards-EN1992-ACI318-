"""
run_pipeline.py
---------------
Master runner: Phases 1-5 in sequence.

  Step 1 -- Parse EC2 PDF            -> data/parsed/ec2_2004_nf_nodes.json
  Step 2 -- Hierarchical chunks      -> data/chunks/ec2_hierarchical_chunks.json
  Step 3 -- Flat chunks (baseline)   -> data/chunks/ec2_flat_chunks.json
  Step 4 -- Index into ChromaDB      -> data/chroma_db/
  Step 5 -- Smoke test (retrieval)

Usage:
    python run_pipeline.py
    python run_pipeline.py --from 2   (skip parsing, re-use existing nodes)
    python run_pipeline.py --from 4   (re-index only)
"""

import argparse
import io
import json
import sys
import time
from collections import Counter
from pathlib import Path

# Force UTF-8 output on Windows (avoids cp1252 crash with French/special chars)
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

BASE = Path(r"C:\Users\manef\OneDrive\Desktop\stageesprit")
sys.path.insert(0, str(BASE))

PDF_PATH         = BASE / "FA039724-NF EN 1992-1-1 (octobre 2005) Eurocode 2.pdf"
NODES_PATH       = BASE / "structrag" / "data" / "parsed"   / "ec2_2004_nf_nodes.json"
HIER_CHUNKS_PATH = BASE / "structrag" / "data" / "chunks"   / "ec2_hierarchical_chunks.json"
FLAT_CHUNKS_PATH = BASE / "structrag" / "data" / "chunks"   / "ec2_flat_chunks.json"
LOG_PATH         = BASE / "pipeline_run.log"

lines = []

def log(msg=""):
    print(msg)
    lines.append(str(msg))

def save_log():
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Step 1: Parse
# ---------------------------------------------------------------------------

def step_parse():
    log("\n=== STEP 1: Parsing EC2 PDF ===")
    from structrag.parsing.standards_parser import parse_standard, extract_cross_refs, EC2_CONFIG
    from structrag.parsing.hierarchy_schema import save_nodes

    NODES_PATH.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    nodes = parse_standard(str(PDF_PATH), EC2_CONFIG)
    nodes = extract_cross_refs(nodes)
    save_nodes(nodes, str(NODES_PATH))

    counts = Counter(n.type for n in nodes)
    log(f"Parsed {len(nodes)} nodes in {time.time()-t0:.1f}s")
    for t, c in sorted(counts.items()):
        log(f"  {t:12s}  {c}")
    return nodes


# ---------------------------------------------------------------------------
# Step 2: Hierarchical chunking
# ---------------------------------------------------------------------------

def step_chunk_hierarchical():
    log("\n=== STEP 2: Hierarchical chunking ===")
    from structrag.parsing.hierarchy_schema import load_nodes
    from structrag.chunking.hierarchical_chunker import chunk_hierarchical, save_chunks

    HIER_CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    nodes = load_nodes(str(NODES_PATH))
    chunks = chunk_hierarchical(nodes, source_id="ec2_2004_nf")
    save_chunks(chunks, str(HIER_CHUNKS_PATH))

    tokens = [c["token_count"] for c in chunks]
    log(f"Built {len(chunks)} hierarchical chunks in {time.time()-t0:.1f}s")
    log(f"  Token stats — min:{min(tokens)}  avg:{sum(tokens)//len(tokens)}  max:{max(tokens)}")
    multi = sum(1 for c in chunks if c["part_index"] > 0)
    log(f"  Continuation parts (split long clauses): {multi}")
    return chunks


# ---------------------------------------------------------------------------
# Step 3: Flat chunking
# ---------------------------------------------------------------------------

def step_chunk_flat():
    log("\n=== STEP 3: Flat chunking (baseline) ===")
    from structrag.chunking.flat_chunker import _extract_pages, _sliding_window_chunks, save_chunks

    FLAT_CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    pages = _extract_pages(str(PDF_PATH), skip_before=17)
    chunks = _sliding_window_chunks(pages, source_id="ec2_2004_nf")
    save_chunks(chunks, str(FLAT_CHUNKS_PATH))

    tokens = [c["token_count"] for c in chunks]
    log(f"Built {len(chunks)} flat chunks in {time.time()-t0:.1f}s")
    log(f"  Token stats — min:{min(tokens)}  avg:{sum(tokens)//len(tokens)}  max:{max(tokens)}")
    return chunks


# ---------------------------------------------------------------------------
# Step 4: Index into ChromaDB
# ---------------------------------------------------------------------------

def step_index():
    log("\n=== STEP 4: Indexing into ChromaDB ===")
    from structrag.retrieval.indexer import index_hierarchical, index_flat
    from structrag.retrieval.chroma_client import list_collections, get_collection, delete_collection
    from structrag.retrieval.chroma_client import COLLECTION_HIERARCHICAL, COLLECTION_FLAT

    # Clear existing collections to avoid stale data on re-runs
    existing = list_collections()
    for name in [COLLECTION_HIERARCHICAL, COLLECTION_FLAT]:
        if name in existing:
            log(f"  Clearing existing collection: {name}")
            delete_collection(name)

    t0 = time.time()
    index_hierarchical(str(HIER_CHUNKS_PATH))
    index_flat(str(FLAT_CHUNKS_PATH))
    log(f"Indexing completed in {time.time()-t0:.1f}s")

    log("\nCollections in ChromaDB:")
    for name in list_collections():
        col = get_collection(name, create_if_missing=False)
        log(f"  {name}  ->  {col.count()} documents")


# ---------------------------------------------------------------------------
# Step 5: Smoke test — retrieval only (no LLM needed)
# ---------------------------------------------------------------------------

def step_smoke_test():
    log("\n=== STEP 5: Smoke test (retrieval only) ===")
    from structrag.retrieval.retriever import retrieve_hierarchical, retrieve_flat

    test_queries = [
        "What is the minimum concrete cover for a beam in exposure class XC2?",
        "Shear resistance formula for members without shear reinforcement",
        "Minimum reinforcement ratio for bending",
    ]

    for query in test_queries:
        log(f"\nQuery: {query}")

        hier = retrieve_hierarchical(query, n_results=3, source_id="ec2_2004_nf")
        log(f"  Hierarchical top-3:")
        for r in hier:
            log(f"    score={r['score']}  clause={r.get('clause_number','')}  "
                f"p={r.get('page_number','')}  "
                f"path={r.get('ancestor_path','')[:60]}")

        flat = retrieve_flat(query, n_results=3, source_id="ec2_2004_nf")
        log(f"  Flat top-3:")
        for r in flat:
            log(f"    score={r['score']}  pages={r.get('start_page','')}-{r.get('end_page','')}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StructRAG pipeline runner")
    parser.add_argument("--from", dest="from_step", type=int, default=1,
                        help="Start from step N (1=parse, 2=hier-chunk, 3=flat-chunk, 4=index, 5=test)")
    args = parser.parse_args()

    start = args.from_step

    try:
        if start <= 1:
            step_parse()
        if start <= 2:
            step_chunk_hierarchical()
        if start <= 3:
            step_chunk_flat()
        if start <= 4:
            step_index()
        if start <= 5:
            step_smoke_test()

        log(f"\n{'='*60}")
        log("Pipeline complete.")
        log(f"{'='*60}")
        log("\nNext steps:")
        log("  - Test queries (no API key):")
        log('      python structrag/rag_pipeline/app.py --no-llm --query "minimum cover XC2"')
        log("  - Full RAG app (needs OPENAI_API_KEY):")
        log("      python structrag/rag_pipeline/app.py")
        log("  - Phase 7: build QA benchmark -> structrag/evaluation/")

    except Exception as e:
        import traceback
        log(f"\nERROR in step {start}:")
        log(traceback.format_exc())

    finally:
        save_log()
        log(f"Log saved -> {LOG_PATH}")
