"""
hierarchical_chunker.py
-----------------------
Phase 3 — Hierarchical Chunking Pipeline (main contribution)

Takes the flat list of Nodes produced by standards_parser.py and assembles
them into retrievable chunks that respect the document's native hierarchy.

Core idea
---------
A "chunk" here is one complete subclause (or clause, for short clauses that
have no children) together with ALL of its direct child nodes — paragraphs,
notes, and image/table placeholders.  The heading text and ancestor_path are
prepended to the chunk text so the embedding captures the full context, not
just the body.

Example chunk text for node 6.2.2:
    [Eurocode 2 > Section 6 > 6.2 Effort tranchant > 6.2.2]
    Éléments pour lesquels aucune armature d'effort tranchant n'est requise

    (1)P Pour les membres ne nécessitant pas d'armatures de cisaillement ...
    (2)P La résistance au cisaillement de calcul VRd,c est donnée par ...
    NOTE  La valeur de CRd,c à utiliser dans un pays donné peut être ...

Why this is better than flat chunking
--------------------------------------
- The heading + ancestor_path travels with every chunk → the embedding
  knows this is about shear resistance in Section 6 of Eurocode 2.
- All applicability conditions in (1)P, (2)P stay with the formula they
  constrain — not split onto a different page boundary.
- Notes (often containing national annex values) stay attached to their clause.

Token budget
------------
tiktoken is used to count tokens (cl100k_base, same tokeniser as text-embedding-
ada-002 and most OpenAI models).  If a subclause's assembled text exceeds
MAX_TOKENS, it is split at paragraph boundaries — never mid-sentence.
Overlapping is applied: the heading + ancestor_path is repeated at the start
of each continuation chunk so every chunk is self-contained.

Output
------
A list of dicts, each containing:
    chunk_id      str   — unique ID
    source_id     str   — e.g. "ec2_2004_nf"
    chunk_type    str   — "hierarchical"
    heading_node  str   — ID of the section/clause/subclause this chunk covers
    clause_number str   — e.g. "6.2.2"
    level         int   — heading level (0=section, 1=clause, 2=subclause, ...)
    ancestor_path str   — full breadcrumb
    page_number   int   — page of the heading node
    text          str   — assembled chunk text (what gets embedded)
    token_count   int   — approximate token count
    cross_refs    list  — clause numbers referenced in this chunk
    part_index    int   — 0 for full clause, 1/2/3... for continuation chunks
"""

from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from structrag.parsing.hierarchy_schema import Node, load_nodes

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MAX_TOKENS = 512          # target max tokens per chunk
OVERLAP_TOKENS = 64       # heading block repeated on continuation chunks

# Heading levels to chunk at.  We chunk at subclause level (2+) by default.
# Clauses that have NO subclause children are also chunked directly.
CHUNK_LEVELS = {0, 1, 2, 3}   # section, clause, subclause, sub-subclause


# ---------------------------------------------------------------------------
# Token counting  (falls back to word-count estimate if tiktoken unavailable)
# ---------------------------------------------------------------------------
try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        return len(_enc.encode(text))

except ImportError:
    def count_tokens(text: str) -> int:  # type: ignore[misc]
        # Rough approximation: 1 token ≈ 0.75 words
        return int(len(text.split()) / 0.75)


# ---------------------------------------------------------------------------
# Build a parent→children index from the flat node list
# ---------------------------------------------------------------------------

def _build_children_index(nodes: list[Node]) -> dict[str, list[Node]]:
    """Return {parent_id: [child_nodes]} for fast lookup."""
    index: dict[str, list[Node]] = {}
    for node in nodes:
        pid = node.parent_id or "__root__"
        index.setdefault(pid, []).append(node)
    return index


def _build_id_index(nodes: list[Node]) -> dict[str, Node]:
    return {n.id: n for n in nodes}


# ---------------------------------------------------------------------------
# Assemble chunk text from a heading node + its direct children
# ---------------------------------------------------------------------------

def _heading_block(node: Node) -> str:
    """The context prefix prepended to every chunk.
    Format:  [ancestor_path]
             clause_number  title
    """
    num = node.clause_number or ""
    title = node.title or ""
    heading_line = f"{num}  {title}".strip() if (num or title) else ""
    return f"[{node.ancestor_path}]\n{heading_line}".strip()


def _child_text(child: Node) -> str:
    """Convert a child node (paragraph / note / table) to a text snippet."""
    tag = child.metadata.get("paragraph_tag", "")
    if child.type == "paragraph":
        prefix = f"{tag}  " if tag else ""
        return f"{prefix}{child.text}".strip()
    elif child.type == "note":
        return f"NOTE  {child.text}".strip() if child.text else ""
    elif child.type in ("table", "image"):
        return f"[TABLE/FIGURE — page {child.page_number}]"
    return child.text.strip() if child.text else ""


# ---------------------------------------------------------------------------
# Split oversized chunks at paragraph boundaries
# ---------------------------------------------------------------------------

def _split_into_parts(
    heading_block: str,
    child_texts: list[str],
    node: Node,
    source_id: str,
) -> list[dict]:
    """
    If the assembled text is within MAX_TOKENS, return one chunk.
    Otherwise, split at paragraph boundaries and return multiple chunks,
    each starting with the heading_block for context continuity.
    """
    parts: list[dict] = []
    part_index = 0
    current_texts: list[str] = []
    current_tokens = count_tokens(heading_block)

    def _make_chunk(texts: list[str], idx: int) -> dict:
        body = "\n\n".join(t for t in texts if t)
        full_text = heading_block + "\n\n" + body if body else heading_block
        all_refs: list[str] = []
        for n_child_text in texts:
            # re-extract any clause refs visible in text
            import re
            refs = re.findall(r'\b(\d+(?:\.\d+){1,3})\b', n_child_text)
            all_refs.extend(refs)
        return {
            "chunk_id": f"{node.id}_part{idx}" if idx > 0 else node.id,
            "source_id": source_id,
            "chunk_type": "hierarchical",
            "heading_node_id": node.id,
            "clause_number": node.clause_number,
            "level": node.level,
            "ancestor_path": node.ancestor_path,
            "page_number": node.page_number,
            "text": full_text,
            "token_count": count_tokens(full_text),
            "cross_refs": list(dict.fromkeys(all_refs + node.cross_refs)),
            "part_index": idx,
        }

    for ct in child_texts:
        ct_tokens = count_tokens(ct)
        if current_tokens + ct_tokens > MAX_TOKENS and current_texts:
            # flush current part
            parts.append(_make_chunk(current_texts, part_index))
            part_index += 1
            # start new part with heading context (overlap)
            current_texts = [ct]
            current_tokens = count_tokens(heading_block) + ct_tokens
        else:
            current_texts.append(ct)
            current_tokens += ct_tokens

    # flush remaining
    if current_texts or part_index == 0:
        parts.append(_make_chunk(current_texts, part_index))

    return parts


# ---------------------------------------------------------------------------
# Main chunker
# ---------------------------------------------------------------------------

def chunk_hierarchical(nodes: list[Node], source_id: str) -> list[dict]:
    """
    Build hierarchical chunks from a parsed node list.

    Strategy:
    1.  Index nodes by parent_id.
    2.  Walk all heading nodes (section, clause, subclause).
    3.  For each heading, collect its DIRECT children that are paragraphs,
        notes, or table/image placeholders.
    4.  Assemble heading_block + child texts into one chunk (or multiple
        if over the token budget).
    5.  Skip heading nodes whose only content is other heading nodes
        (i.e. pure container sections with no direct text children) —
        those are covered by their children's chunks.
    """
    children_idx = _build_children_index(nodes)

    chunks: list[dict] = []
    heading_types = {"section", "clause", "subclause"}

    for node in nodes:
        if node.type not in heading_types:
            continue

        # Direct children that carry text content
        direct_children = children_idx.get(node.id, [])
        text_children = [
            c for c in direct_children
            if c.type in ("paragraph", "note", "table", "image")
        ]

        # If this heading has no direct text children, skip it —
        # it's a pure structural container (its subclauses have the content)
        if not text_children:
            continue

        heading_block = _heading_block(node)
        child_texts = [_child_text(c) for c in text_children]

        parts = _split_into_parts(heading_block, child_texts, node, source_id)
        chunks.extend(parts)

    return chunks


# ---------------------------------------------------------------------------
# Save / load helpers
# ---------------------------------------------------------------------------

def save_chunks(chunks: list[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(chunks)} hierarchical chunks -> {path}")


def load_chunks(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    base = Path(r"C:\Users\manef\OneDrive\Desktop\stageesprit")
    nodes_path = base / "structrag" / "data" / "parsed" / "ec2_2004_nf_nodes.json"
    output_dir = base / "structrag" / "data" / "chunks"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "ec2_hierarchical_chunks.json"

    print(f"Loading nodes from {nodes_path.name} ...")
    nodes = load_nodes(str(nodes_path))
    print(f"Loaded {len(nodes)} nodes")

    print("Building hierarchical chunks ...")
    chunks = chunk_hierarchical(nodes, source_id="ec2_2004_nf")

    # --- Statistics ---
    from collections import Counter
    token_counts = [c["token_count"] for c in chunks]
    level_counts = Counter(c["level"] for c in chunks)
    part_counts  = Counter(c["part_index"] for c in chunks)

    print(f"\nTotal hierarchical chunks: {len(chunks)}")
    print(f"  Token stats — min:{min(token_counts)}  "
          f"avg:{sum(token_counts)//len(token_counts)}  "
          f"max:{max(token_counts)}")
    print(f"  By heading level: {dict(sorted(level_counts.items()))}")
    print(f"  Multi-part splits (part>0): {sum(v for k,v in part_counts.items() if k>0)}")

    # --- Sample ---
    print("\n--- Sample chunks ---")
    for c in chunks[:3]:
        print(f"  id={c['chunk_id']}")
        print(f"  path={c['ancestor_path'][:70]}")
        print(f"  tokens={c['token_count']}  parts={c['part_index']}")
        print(f"  text preview: {c['text'][:200]}")
        print()

    save_chunks(chunks, str(output_path))
