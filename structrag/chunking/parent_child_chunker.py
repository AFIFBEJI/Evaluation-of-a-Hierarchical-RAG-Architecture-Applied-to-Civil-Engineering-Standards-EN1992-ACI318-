"""
structrag/chunking/parent_child_chunker.py
------------------------------------------
Parent-child (small-to-big) chunking strategy.

Architecture
------------
Standard RAG retrieval has a precision-recall tension:
  - Small chunks  → precise embedding, but missing surrounding context
  - Large chunks  → rich context, but embedding is diluted by irrelevant text

The parent-child pattern resolves this by maintaining TWO representations:
  - CHILD chunks  : small, focused, embeddable (one paragraph or one table row)
                    These are what gets indexed in ChromaDB for retrieval.
  - PARENT chunks : the full clause or section that contains those children
                    These are NOT indexed for retrieval — they are looked up
                    by ID after retrieval to form the generation context.

Query flow
----------
1. Embed user query.
2. Retrieve top-k CHILD chunks by cosine similarity (+ BM25).
3. For each child, look up its PARENT chunk by parent_chunk_id.
4. De-duplicate parents (multiple children may share one parent).
5. Pass the PARENT texts to the LLM — full clause context, not fragments.

Why this fixes failures
-----------------------
Q6 (εcu2 strain): the child chunk "εcu2 (‰) | 3,5" embeds well for the
specific value query; the parent gives the full Table 3.1 with all parameters.

Q8 (characteristic vs design values): a child paragraph from clause 3.1.1
embeds well for "characteristic value fck"; the parent gives the surrounding
explanation including the design value relationship.

Output
------
Two JSON files:
  data/chunks/ec2_child_chunks.json   — small chunks for embedding/retrieval
  data/chunks/ec2_parent_chunks.json  — full-clause chunks for generation

Each CHILD chunk dict:
    chunk_id        str
    source_id       str
    chunk_type      "child"
    parent_chunk_id str   — ID of the parent chunk to look up at generation time
    clause_number   str
    level           int
    ancestor_path   str
    page_number     int
    text            str   — small focused text (one paragraph / table row)
    token_count     int
    child_type      str   — "paragraph" | "note" | "table_row" | "table"
    paragraph_tag   str   — "(1)P", "(2)", etc. if applicable

Each PARENT chunk dict:
    chunk_id        str
    source_id       str
    chunk_type      "parent"
    clause_number   str
    level           int
    ancestor_path   str
    page_number     int
    text            str   — full clause text (heading + all children)
    token_count     int
    cross_refs      list
    child_ids       list  — IDs of child chunks that point to this parent
"""

from __future__ import annotations
import json
import re
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from structrag.parsing.hierarchy_schema import Node, load_nodes

# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------
try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text: str) -> int:
        return len(_enc.encode(text))
except ImportError:
    def count_tokens(text: str) -> int:
        return int(len(text.split()) / 0.75)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PARENT_MAX_TOKENS = 1500   # max tokens in a parent chunk
CHILD_MAX_TOKENS  = 200    # max tokens in a single child chunk

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _heading_block(node: Node) -> str:
    num   = node.clause_number or ""
    title = node.title or ""
    heading = f"{num}  {title}".strip() if (num or title) else ""
    return f"[{node.ancestor_path}]\n{heading}".strip()


def _child_text(child: Node) -> str:
    tag = child.metadata.get("paragraph_tag", "") if child.metadata else ""
    if child.type == "paragraph":
        prefix = f"{tag}  " if tag else ""
        return f"{prefix}{child.text}".strip()
    elif child.type == "note":
        return f"NOTE  {child.text}".strip() if child.text else ""
    elif child.type in ("table", "image"):
        return f"[TABLE/FIGURE — page {child.page_number}]"
    return child.text.strip() if child.text else ""


def _build_children_index(nodes: list[Node]) -> dict[str, list[Node]]:
    idx: dict[str, list[Node]] = {}
    for node in nodes:
        pid = node.parent_id or "__root__"
        idx.setdefault(pid, []).append(node)
    return idx


def _extract_cross_refs(text: str) -> list[str]:
    refs = re.findall(r'\b(\d+(?:\.\d+){1,3})\b', text)
    return list(dict.fromkeys(refs))


# ---------------------------------------------------------------------------
# Split a long parent into multiple parents at paragraph boundaries
# ---------------------------------------------------------------------------

def _split_parent_text(
    heading_block: str,
    child_texts: list[str],
    max_tokens: int,
) -> list[str]:
    """
    If heading + all children exceeds max_tokens, split into multiple parent
    texts, each starting with the heading_block for context continuity.
    Returns a list of text strings (usually one, occasionally more).
    """
    parts: list[str] = []
    current: list[str] = []
    current_tokens = count_tokens(heading_block)

    for ct in child_texts:
        ct_tokens = count_tokens(ct)
        if current_tokens + ct_tokens > max_tokens and current:
            parts.append(heading_block + "\n\n" + "\n\n".join(current))
            current = [ct]
            current_tokens = count_tokens(heading_block) + ct_tokens
        else:
            current.append(ct)
            current_tokens += ct_tokens

    if current or not parts:
        parts.append(heading_block + "\n\n" + "\n\n".join(current))

    return parts


# ---------------------------------------------------------------------------
# Table row splitting — each row of a table becomes its own child chunk
# ---------------------------------------------------------------------------

def _split_table_rows(table_text: str, ancestor_path: str) -> list[str]:
    """
    Split a markdown table into individual row strings, each self-contained
    with the caption and header row repeated.
    Returns a list of row-level strings.
    """
    lines = table_text.strip().split("\n")
    if not lines:
        return [table_text]

    # Find the caption (first line), header row and separator
    caption_lines = []
    header_row    = None
    sep_row       = None
    data_rows     = []

    i = 0
    # Caption lines: everything before the first | row
    while i < len(lines) and not lines[i].strip().startswith("|"):
        caption_lines.append(lines[i])
        i += 1
    # Header row
    if i < len(lines):
        header_row = lines[i]; i += 1
    # Separator row
    if i < len(lines) and set(lines[i].replace("|","").replace("-","").replace(" ","")) == set():
        sep_row = lines[i]; i += 1
    # Data rows
    while i < len(lines):
        if lines[i].strip():
            data_rows.append(lines[i])
        i += 1

    if not data_rows or header_row is None:
        return [table_text]

    caption_str = "\n".join(caption_lines)
    header_str  = f"{header_row}\n{sep_row}" if sep_row else header_row

    row_chunks = []
    for row in data_rows:
        row_text = f"{caption_str}\n{header_str}\n{row}".strip()
        row_chunks.append(row_text)

    return row_chunks if row_chunks else [table_text]


# ---------------------------------------------------------------------------
# Main chunker
# ---------------------------------------------------------------------------

def chunk_parent_child(
    nodes: list[Node],
    source_id: str,
) -> tuple[list[dict], list[dict]]:
    """
    Build parent and child chunk lists from a parsed node list.

    Returns (child_chunks, parent_chunks).
    """
    children_idx = _build_children_index(nodes)
    heading_types = {"section", "clause", "subclause"}

    parent_chunks: list[dict] = []
    child_chunks:  list[dict] = []

    for node in nodes:
        if node.type not in heading_types:
            continue

        direct_children = children_idx.get(node.id, [])
        text_children   = [
            c for c in direct_children
            if c.type in ("paragraph", "note", "table", "image")
        ]

        if not text_children:
            continue   # pure structural container, skip

        heading_block = _heading_block(node)
        child_texts   = [_child_text(c) for c in text_children]

        # ── Build PARENT chunks ──────────────────────────────────────────
        parent_parts = _split_parent_text(heading_block, child_texts, PARENT_MAX_TOKENS)
        parent_ids: list[str] = []

        for part_idx, part_text in enumerate(parent_parts):
            pid = f"{node.id}_parent" if part_idx == 0 else f"{node.id}_parent_p{part_idx}"
            all_refs = _extract_cross_refs(part_text) + node.cross_refs
            parent_chunks.append({
                "chunk_id":      pid,
                "source_id":     source_id,
                "chunk_type":    "parent",
                "clause_number": node.clause_number or "",
                "level":         node.level,
                "ancestor_path": node.ancestor_path,
                "page_number":   node.page_number,
                "text":          part_text,
                "token_count":   count_tokens(part_text),
                "cross_refs":    list(dict.fromkeys(all_refs)),
                "child_ids":     [],   # filled in below
                "part_index":    part_idx,
            })
            parent_ids.append(pid)

        # Map each child paragraph to the correct parent part
        # (simple heuristic: children are distributed round-robin across parts
        #  but since most clauses fit in one parent, this is usually trivial)
        children_per_part = max(1, len(text_children) // len(parent_parts))

        for child_idx, (child_node, child_text) in enumerate(
            zip(text_children, child_texts)
        ):
            # Which parent part does this child belong to?
            part_idx_for_child = min(
                child_idx // children_per_part,
                len(parent_ids) - 1
            )
            parent_id = parent_ids[part_idx_for_child]

            # Update parent's child_ids list
            for p in parent_chunks:
                if p["chunk_id"] == parent_id:
                    p["child_ids"].append(f"{child_node.id}_child")
                    break

            # ── Build CHILD chunks ────────────────────────────────────────
            child_type = child_node.type  # paragraph / note / table / image

            # For tables: split into row-level children
            if child_type in ("table",) and "|\n|" in child_text:
                row_texts = _split_table_rows(child_text, node.ancestor_path)
                for row_idx, row_text in enumerate(row_texts):
                    cid = f"{child_node.id}_child_row{row_idx}"
                    child_chunks.append({
                        "chunk_id":        cid,
                        "source_id":       source_id,
                        "chunk_type":      "child",
                        "parent_chunk_id": parent_id,
                        "clause_number":   node.clause_number or "",
                        "level":           node.level,
                        "ancestor_path":   node.ancestor_path,
                        "page_number":     child_node.page_number,
                        "text":            f"[{node.ancestor_path}]\n{row_text}",
                        "token_count":     count_tokens(row_text),
                        "child_type":      "table_row",
                        "paragraph_tag":   "",
                    })
            else:
                cid = f"{child_node.id}_child"
                tag = child_node.metadata.get("paragraph_tag", "") \
                      if child_node.metadata else ""
                # Prepend ancestor path so embedding captures document location
                full_text = f"[{node.ancestor_path}]\n{child_text}"
                child_chunks.append({
                    "chunk_id":        cid,
                    "source_id":       source_id,
                    "chunk_type":      "child",
                    "parent_chunk_id": parent_id,
                    "clause_number":   node.clause_number or "",
                    "level":           node.level,
                    "ancestor_path":   node.ancestor_path,
                    "page_number":     child_node.page_number,
                    "text":            full_text,
                    "token_count":     count_tokens(full_text),
                    "child_type":      child_type,
                    "paragraph_tag":   tag,
                })

    return child_chunks, parent_chunks


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_chunks(chunks: list[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(chunks)} chunks -> {path}")


def load_chunks(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Parent lookup helper (used at generation time)
# ---------------------------------------------------------------------------

_parent_cache: Optional[dict[str, dict]] = None
_parent_cache_path: Optional[str] = None


def get_parent_lookup(parents_path: str) -> dict[str, dict]:
    """Return a {chunk_id: parent_chunk} dict, cached in memory."""
    global _parent_cache, _parent_cache_path
    if _parent_cache is None or _parent_cache_path != parents_path:
        parents = load_chunks(parents_path)
        _parent_cache = {p["chunk_id"]: p for p in parents}
        _parent_cache_path = parents_path
    return _parent_cache


def resolve_parents(
    child_results: list[dict],
    parents_path: str,
) -> list[dict]:
    """
    Given a list of child chunk retrieval results, look up and return the
    corresponding parent chunks (de-duplicated).

    Parameters
    ----------
    child_results : results from retrieve_child()
    parents_path  : path to ec2_parent_chunks.json

    Returns
    -------
    List of parent chunk dicts, ordered by best child score, de-duplicated.
    """
    parent_lookup = get_parent_lookup(parents_path)
    seen_parents: set[str] = set()
    resolved: list[dict] = []

    for child in child_results:
        pid = child.get("parent_chunk_id", "")
        if pid and pid not in seen_parents:
            parent = parent_lookup.get(pid)
            if parent:
                # Carry the child's retrieval score to the parent for ranking
                parent_copy = dict(parent)
                parent_copy["score"] = child.get("score", 0)
                parent_copy["retrieved_via_child"] = child["chunk_id"]
                resolved.append(parent_copy)
                seen_parents.add(pid)

    return resolved


from structrag.parsing.table_extractor import extract_tables, TABLE_SPECS


def _add_table_nodes_to_parent_child(
    pdf_path: str,
    source_id: str,
    child_chunks: list[dict],
    parent_chunks: list[dict],
) -> tuple[list[dict], list[dict]]:
    """
    After building parent-child chunks from the parsed node tree,
    also inject the pdfplumber-extracted table nodes.
    Each table becomes ONE parent chunk and multiple row-level child chunks.
    """
    table_nodes = extract_tables(pdf_path, source_id)
    parent_ids  = {p["chunk_id"] for p in parent_chunks}

    for tnode in table_nodes:
        table_text = tnode.text
        tid        = tnode.id
        clause     = tnode.clause_number or ""
        path       = tnode.ancestor_path

        # Parent = full table text
        parent_id = f"{tid}_parent"
        parent_chunks.append({
            "chunk_id":      parent_id,
            "source_id":     source_id,
            "chunk_type":    "parent",
            "clause_number": clause,
            "level":         tnode.level,
            "ancestor_path": path,
            "page_number":   tnode.page_number,
            "text":          table_text,
            "token_count":   count_tokens(table_text),
            "cross_refs":    [],
            "child_ids":     [],
            "part_index":    0,
            "metadata":      tnode.metadata,
        })

        # Children = individual table rows
        row_texts = _split_table_rows(table_text, path)
        for row_idx, row_text in enumerate(row_texts):
            cid = f"{tid}_child_row{row_idx}"
            full_text = f"[{path}]\n{row_text}"
            child_chunks.append({
                "chunk_id":        cid,
                "source_id":       source_id,
                "chunk_type":      "child",
                "parent_chunk_id": parent_id,
                "clause_number":   clause,
                "level":           tnode.level,
                "ancestor_path":   path,
                "page_number":     tnode.page_number,
                "text":            full_text,
                "token_count":     count_tokens(full_text),
                "child_type":      "table_row",
                "paragraph_tag":   "",
            })
            # Add child to parent's child_ids
            parent_chunks[-1]["child_ids"].append(cid)

    return child_chunks, parent_chunks

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import io
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    base        = Path(r"C:\Users\manef\OneDrive\Desktop\stageesprit")
    nodes_path  = base / "structrag" / "data" / "parsed"  / "ec2_2004_nf_nodes.json"
    pdf_path    = base / "FA039724-NF EN 1992-1-1 (octobre 2005) Eurocode 2.pdf"
    output_dir  = base / "structrag" / "data" / "chunks"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading nodes...")
    nodes = load_nodes(str(nodes_path))
    print(f"Loaded {len(nodes)} nodes")

    print("Building parent-child chunks from node tree...")
    child_chunks, parent_chunks = chunk_parent_child(nodes, source_id="ec2_2004_nf")

    print("Adding table nodes as parent-child chunks...")
    child_chunks, parent_chunks = _add_table_nodes_to_parent_child(
        str(pdf_path), "ec2_2004_nf", child_chunks, parent_chunks
    )

    from collections import Counter
    child_types   = Counter(c["child_type"] for c in child_chunks)
    parent_tokens = [p["token_count"] for p in parent_chunks]
    child_tokens  = [c["token_count"]  for c in child_chunks]

    print(f"\nParent chunks : {len(parent_chunks)}")
    print(f"  tokens — min:{min(parent_tokens)}  avg:{sum(parent_tokens)//len(parent_tokens)}  max:{max(parent_tokens)}")

    print(f"\nChild  chunks : {len(child_chunks)}")
    print(f"  tokens — min:{min(child_tokens)}  avg:{sum(child_tokens)//len(child_tokens)}  max:{max(child_tokens)}")
    print(f"  by type: {dict(child_types)}")

    print("\n--- Sample: clause 2.4.2.4 (gamma_c / gamma_s) ---")
    for p in parent_chunks:
        if p["clause_number"] == "2.4.2.4" and "_table_" not in p["chunk_id"]:
            print(f"  PARENT: {p['chunk_id']}  tokens={p['token_count']}")
            print(f"  text preview: {p['text'][:300]}")
            break

    print("\n--- Sample: Table 4.4N row children ---")
    table_rows = [c for c in child_chunks
                  if c["child_type"] == "table_row" and "4_4N" in c["chunk_id"]]
    for r in table_rows[:4]:
        print(f"  {r['chunk_id']}  tokens={r['token_count']}")
        print(f"  text: {r['text'][:200]}")

    save_chunks(child_chunks,  str(output_dir / "ec2_child_chunks.json"))
    save_chunks(parent_chunks, str(output_dir / "ec2_parent_chunks.json"))
    print("\nDone.")
