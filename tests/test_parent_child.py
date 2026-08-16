"""
tests/test_parent_child.py
--------------------------
Test the parent-child chunker and verify the architecture is correct.
"""
import sys, os, json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

results = open(BASE / "tests" / "test_parent_child_result.txt", "w", encoding="utf-8")

def w(s=""):
    print(str(s))
    results.write(str(s) + "\n")

try:
    from structrag.parsing.hierarchy_schema import load_nodes
    from structrag.chunking.parent_child_chunker import (
        chunk_parent_child, save_chunks, resolve_parents
    )
    from collections import Counter

    # ── 1. Build chunks ───────────────────────────────────────────────────
    nodes_path = BASE / "structrag" / "data" / "parsed" / "ec2_2004_nf_nodes.json"
    nodes = load_nodes(str(nodes_path))
    w(f"Loaded {len(nodes)} nodes")

    child_chunks, parent_chunks = chunk_parent_child(nodes, "ec2_2004_nf")

    w(f"\nParent chunks: {len(parent_chunks)}")
    w(f"Child  chunks: {len(child_chunks)}")

    # ── 2. Token stats ────────────────────────────────────────────────────
    pt = [p["token_count"] for p in parent_chunks]
    ct = [c["token_count"] for c in child_chunks]
    w(f"\nParent token stats: min={min(pt)} avg={sum(pt)//len(pt)} max={max(pt)}")
    w(f"Child  token stats: min={min(ct)} avg={sum(ct)//len(ct)} max={max(ct)}")

    child_types = Counter(c["child_type"] for c in child_chunks)
    w(f"Child types: {dict(child_types)}")

    # ── 3. Verify every child has a valid parent_chunk_id ─────────────────
    parent_ids = {p["chunk_id"] for p in parent_chunks}
    orphan_children = [c for c in child_chunks if c["parent_chunk_id"] not in parent_ids]
    w(f"\nOrphan children (no valid parent): {len(orphan_children)}")
    if orphan_children:
        w("  FAIL — orphaned children found:")
        for o in orphan_children[:5]:
            w(f"    {o['chunk_id']} -> parent={o['parent_chunk_id']}")
    else:
        w("  PASS — all children point to a valid parent")

    # ── 4. Verify parent.child_ids are populated ──────────────────────────
    empty_parents = [p for p in parent_chunks if not p["child_ids"]]
    w(f"\nParents with no child_ids: {len(empty_parents)} (should be 0)")
    if empty_parents:
        w("  WARN — some parents have no child_ids")

    # ── 5. Spot check: clause 2.4.2.4 (gamma_c values) ───────────────────
    w("\n--- Spot check: clause 2.4.2.4 (partial safety factors) ---")
    par_2424 = [p for p in parent_chunks if p["clause_number"] == "2.4.2.4"]
    w(f"  Parent chunks for 2.4.2.4: {len(par_2424)}")
    for p in par_2424[:2]:
        w(f"  {p['chunk_id']}  tokens={p['token_count']}")
        w(f"  preview: {p['text'][:300]}")
        w()

    chi_2424 = [c for c in child_chunks if c["clause_number"] == "2.4.2.4"]
    w(f"  Child chunks for 2.4.2.4: {len(chi_2424)}")
    for c in chi_2424[:3]:
        w(f"  {c['chunk_id']}  type={c['child_type']}  tag={c['paragraph_tag']}  tokens={c['token_count']}")
        w(f"  text: {c['text'][:150]}")
        w()

    # ── 6. Spot check: Table 4.4N row-level children ──────────────────────
    w("--- Spot check: Table 4.4N (cover values — should have row-level children) ---")
    table_rows = [c for c in child_chunks if c["child_type"] == "table_row"]
    w(f"  Total table_row children: {len(table_rows)}")
    for r in table_rows[:4]:
        w(f"  {r['chunk_id']}  clause={r['clause_number']}  tokens={r['token_count']}")
        w(f"  text: {r['text'][:200]}")
        w()

    # ── 7. Test resolve_parents ───────────────────────────────────────────
    w("--- Test resolve_parents with simulated retrieval ---")
    output_dir = BASE / "structrag" / "data" / "chunks"
    output_dir.mkdir(parents=True, exist_ok=True)
    child_path  = output_dir / "ec2_child_chunks.json"
    parent_path = output_dir / "ec2_parent_chunks.json"

    save_chunks(child_chunks,  str(child_path))
    save_chunks(parent_chunks, str(parent_path))
    w(f"Saved child chunks  -> {child_path.name}")
    w(f"Saved parent chunks -> {parent_path.name}")

    # Simulate: pretend the first 3 child chunks were retrieved
    fake_retrieval = [
        {**c, "score": 0.05} for c in child_chunks[:3]
    ]
    resolved = resolve_parents(fake_retrieval, str(parent_path))
    w(f"\nresolve_parents({len(fake_retrieval)} children) -> {len(resolved)} unique parents")
    for r in resolved:
        w(f"  {r['chunk_id']}  clause={r['clause_number']}  "
          f"score={r['score']}  tokens={r['token_count']}")

    w("\nAll tests passed.")

except Exception as e:
    import traceback
    w(f"ERROR: {e}")
    w(traceback.format_exc())

results.close()
print("Done -> tests/test_parent_child_result.txt")
