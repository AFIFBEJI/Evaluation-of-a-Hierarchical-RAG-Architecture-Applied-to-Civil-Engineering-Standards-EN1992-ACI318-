"""
tests/inspect_chunks.py
-----------------------
Inspect raw chunk text for specific clauses in the indexed collection.
Run this before making any ingestion changes.
"""
import sys, os, json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

# Load .env
for line in (BASE / ".env").read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ[k.strip()] = v.strip()

out = open(BASE / "tests" / "chunk_inspection.txt", "w", encoding="utf-8")

def w(s=""):
    print(s)
    out.write(s + "\n")

# ── 1. Load chunks JSON directly (no ChromaDB needed) ─────────────────────
chunks_path = BASE / "structrag" / "data" / "chunks" / "ec2_hierarchical_chunks.json"
chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
w(f"Total hierarchical chunks loaded: {len(chunks)}")
w()

# ── 2. Target clauses to inspect ──────────────────────────────────────────
targets = {
    "2.4.2.4": "Partial safety factors for materials (gamma_c, gamma_s)",
    "3.1.7":   "Stress-strain relations / compressive strain epsilon_cu2",
    "4.3":     "Durability requirements / structural classification",
    "4.4.1.2": "Minimum cover c_min (Table 4.4N cover values)",
    "4.4.1.1": "Nominal cover definition",
    "2.4.2":   "Design values (characteristic vs design)",
    "3.1.2":   "Concrete strength / fck",
}

# ── 3. Also search for key numeric strings in ALL chunks ──────────────────
numeric_targets = [
    ("γc",      ["γc", "gamma_c", "1,5", "1.5", "1,50"]),
    ("γs",      ["γs", "gamma_s", "1,15", "1.15"]),
    ("εcu2",    ["εcu2", "epsilon_cu2", "3,5", "3‰", "3.5"]),
    ("Table 4.3N", ["4.3N", "Tableau 4.3N", "classe de structure"]),
    ("Table 4.4N", ["4.4N", "Tableau 4.4N"]),
    ("Table 2.1N", ["2.1N", "Tableau 2.1N"]),
    ("XC2 cover",  ["XC2", "25 mm", "25mm"]),
]

w("=" * 70)
w("SECTION 1 — Full text of targeted clauses")
w("=" * 70)

found_clauses = {}
for chunk in chunks:
    cn = chunk.get("clause_number", "")
    if cn in targets:
        found_clauses[cn] = chunk

for clause_id, description in targets.items():
    w()
    w(f"--- Clause {clause_id}: {description} ---")
    if clause_id not in found_clauses:
        # might be split into multiple parts
        parts = [c for c in chunks if c.get("clause_number","").startswith(clause_id)]
        if parts:
            w(f"  Found {len(parts)} chunk part(s):")
            for p in parts:
                w(f"  chunk_id : {p['chunk_id']}")
                w(f"  tokens   : {p['token_count']}")
                w(f"  part_idx : {p['part_index']}")
                w(f"  FULL TEXT:")
                w(p['text'])
                w()
        else:
            w(f"  NOT FOUND in chunks")
    else:
        c = found_clauses[clause_id]
        w(f"  chunk_id : {c['chunk_id']}")
        w(f"  tokens   : {c['token_count']}")
        w(f"  part_idx : {c['part_index']}")
        w(f"  FULL TEXT:")
        w(c['text'])

w()
w("=" * 70)
w("SECTION 2 — Which chunks contain key numeric values")
w("=" * 70)

for label, search_terms in numeric_targets:
    w()
    w(f"--- Searching for: {label} ---")
    hits = []
    for chunk in chunks:
        text = chunk.get("text", "")
        for term in search_terms:
            if term.lower() in text.lower():
                hits.append((chunk.get("clause_number","?"), chunk["chunk_id"], term, text[:300]))
                break
    if hits:
        w(f"  Found in {len(hits)} chunk(s):")
        for cn, cid, matched_term, preview in hits[:5]:
            w(f"    clause={cn}  id={cid}  matched='{matched_term}'")
            w(f"    preview: {preview[:200]}")
            w()
    else:
        w(f"  NOT FOUND in any chunk")

w()
w("=" * 70)
w("SECTION 3 — Image/table blocks: how many, which clauses they belong to")
w("=" * 70)

# Load nodes to check image blocks
nodes_path = BASE / "structrag" / "data" / "parsed" / "ec2_2004_nf_nodes.json"
nodes = json.loads(nodes_path.read_text(encoding="utf-8"))
image_nodes = [n for n in nodes if n["type"] in ("table", "image")]
w(f"Total image/table nodes: {len(image_nodes)}")
w()

# Group by parent clause
from collections import defaultdict
by_parent = defaultdict(list)
for n in image_nodes:
    pid = n.get("parent_id", "none")
    by_parent[pid].append(n)

w("Image nodes grouped by parent clause (top 20 parents by count):")
sorted_parents = sorted(by_parent.items(), key=lambda x: -len(x[1]))
for pid, img_list in sorted_parents[:20]:
    # find parent clause number
    parent_node = next((n for n in nodes if n["id"] == pid), None)
    parent_cn = parent_node["clause_number"] if parent_node else "?"
    parent_title = parent_node["title"][:50] if parent_node else "?"
    pages = sorted({n["page_number"] for n in img_list})
    w(f"  parent={parent_cn} ({parent_title})  count={len(img_list)}  pages={pages}")

w()
w("All image nodes with page numbers (to cross-check against PDF):")
for n in image_nodes:
    parent = next((p for p in nodes if p["id"] == n.get("parent_id","")), None)
    parent_cn = parent["clause_number"] if parent else "?"
    w(f"  p={n['page_number']:3d}  parent_clause={parent_cn:12s}  id={n['id']}")

out.close()
print(f"\nInspection complete -> tests/chunk_inspection.txt")
