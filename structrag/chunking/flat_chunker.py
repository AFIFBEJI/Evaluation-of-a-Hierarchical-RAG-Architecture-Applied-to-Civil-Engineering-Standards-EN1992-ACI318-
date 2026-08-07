"""
flat_chunker.py
---------------
Phase 6 — Flat-Chunking Baseline

Naive fixed-size sliding-window chunking — the BASELINE that the hierarchical
pipeline is compared against.  Uses the same corpus, same embedding model, and
same token budget.  The only difference is: no hierarchy awareness, no
ancestor_path, no clause boundaries respected.

How it works
------------
1.  Read the raw PDF text page-by-page with PyMuPDF.
2.  Strip page headers/footers, concatenate all content pages.
3.  Tokenise the full text, slide a window of MAX_TOKENS with OVERLAP_TOKENS.
4.  Each window becomes one chunk — it will routinely span clause boundaries.

Output
------
structrag/data/chunks/ec2_flat_chunks.json
Each chunk dict:
    chunk_id     str
    source_id    str
    chunk_type   "flat"
    text         str
    token_count  int
    start_page   int
    end_page     int
    part_index   int
"""

from __future__ import annotations
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Configuration — identical to hierarchical chunker for fair comparison
# ---------------------------------------------------------------------------
MAX_TOKENS     = 512
OVERLAP_TOKENS = 64
SKIP_PAGES_BEFORE = 17  # cover + TOC pages (0-indexed)

# ---------------------------------------------------------------------------
# Token counting — same fallback logic as hierarchical_chunker.py
# ---------------------------------------------------------------------------
try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        return len(_enc.encode(text))

    def _encode(text: str) -> list:
        return _enc.encode(text)

    def _decode(token_ids: list) -> str:
        return _enc.decode(token_ids)

    TIKTOKEN_AVAILABLE = True

except ImportError:
    TIKTOKEN_AVAILABLE = False

    def count_tokens(text: str) -> int:       # type: ignore[misc]
        return int(len(text) / 4)

    def _encode(text: str) -> list:           # type: ignore[misc]
        # Treat whitespace-separated words as tokens
        return text.split()

    def _decode(token_ids: list) -> str:      # type: ignore[misc]
        if token_ids and isinstance(token_ids[0], str):
            return " ".join(token_ids)
        raise NotImplementedError("tiktoken not available; cannot decode token IDs")


# ---------------------------------------------------------------------------
# Extract text from PDF, stripping headers/footers
# ---------------------------------------------------------------------------

# Pattern for the EC2 page header: "Page 14" or "EN 1992-1-1:2004"
_HEADER_RE = re.compile(r'^Page\s+\d+$')
_STANDARD_REF = "EN 1992-1-1:2004"


def _extract_pages(pdf_path: str, skip_before: int) -> list[tuple[int, str]]:
    """
    Return [(page_number_1indexed, cleaned_text), ...] for all content pages.
    """
    import fitz  # PyMuPDF
    doc = fitz.open(pdf_path)
    pages = []
    for i in range(skip_before, len(doc)):
        raw = doc[i].get_text("text")
        lines = raw.splitlines()
        clean_lines = [
            ln for ln in lines
            if not _HEADER_RE.match(ln.strip())
            and ln.strip() != _STANDARD_REF
            and ln.strip()
        ]
        clean_text = re.sub(r'\s+', ' ', " ".join(clean_lines)).strip()
        if clean_text:
            pages.append((i + 1, clean_text))   # 1-indexed
    return pages


# ---------------------------------------------------------------------------
# Sliding-window chunker
# ---------------------------------------------------------------------------

def _sliding_window_chunks(
    pages: list[tuple[int, str]],
    source_id: str,
    max_tokens: int = MAX_TOKENS,
    overlap: int = OVERLAP_TOKENS,
) -> list[dict]:
    """
    Concatenate all pages into one token stream and slide a fixed window.
    The page-boundary table lets us report start_page / end_page per chunk
    even though the chunker itself is completely unaware of page structure.
    """
    # ---- build concatenated text and record char offset of each page ----
    combined_parts: list[str] = []
    page_char_offsets: list[tuple[int, int]] = []   # (char_start, page_num)
    cursor = 0
    for page_num, text in pages:
        page_char_offsets.append((cursor, page_num))
        combined_parts.append(text)
        cursor += len(text) + 1   # +1 for the space separator
    combined_text = " ".join(combined_parts)

    # ---- tokenise -------------------------------------------------------
    all_tokens = _encode(combined_text)

    # Build a sparse map: every N tokens → page number
    # We compute it by encoding prefixes at page boundaries
    page_token_starts: list[tuple[int, int]] = []  # (token_idx, page_num)
    if TIKTOKEN_AVAILABLE:
        for char_start, page_num in page_char_offsets:
            tok_idx = len(_enc.encode(combined_text[:char_start]))
            page_token_starts.append((tok_idx, page_num))
    else:
        for char_start, page_num in page_char_offsets:
            # word-based: count words before this offset
            tok_idx = len(combined_text[:char_start].split())
            page_token_starts.append((tok_idx, page_num))

    def _page_for_token(tok_idx: int) -> int:
        page = page_token_starts[0][1]
        for start, pnum in page_token_starts:
            if tok_idx >= start:
                page = pnum
            else:
                break
        return page

    # ---- slide window ---------------------------------------------------
    chunks: list[dict] = []
    step = max_tokens - overlap
    start = 0
    part_index = 0

    while start < len(all_tokens):
        end = min(start + max_tokens, len(all_tokens))
        window = all_tokens[start:end]
        text = _decode(window)
        start_page = _page_for_token(start)
        end_page   = _page_for_token(end - 1)

        chunks.append({
            "chunk_id":   f"{source_id}_flat_{part_index}",
            "source_id":  source_id,
            "chunk_type": "flat",
            "text":       text,
            "token_count": len(window),
            "start_page": start_page,
            "end_page":   end_page,
            "part_index": part_index,
            # Deliberately no ancestor_path or clause_number — that is the point
        })

        part_index += 1
        start += step
        if end == len(all_tokens):
            break

    return chunks


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_chunks(chunks: list[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(chunks)} flat chunks -> {path}")


def load_chunks(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    base = Path(r"C:\Users\manef\OneDrive\Desktop\stageesprit")
    pdf_path   = base / "FA039724-NF EN 1992-1-1 (octobre 2005) Eurocode 2.pdf"
    output_dir = base / "structrag" / "data" / "chunks"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "ec2_flat_chunks.json"

    print(f"Extracting text from {pdf_path.name} ...")
    pages = _extract_pages(str(pdf_path), SKIP_PAGES_BEFORE)
    print(f"Content pages: {len(pages)}")

    print(f"Building flat chunks (window={MAX_TOKENS}, overlap={OVERLAP_TOKENS}) ...")
    chunks = _sliding_window_chunks(pages, source_id="ec2_2004_nf")

    token_counts = [c["token_count"] for c in chunks]
    print(f"\nTotal flat chunks : {len(chunks)}")
    print(f"Token stats — min:{min(token_counts)}  "
          f"avg:{sum(token_counts)//len(token_counts)}  "
          f"max:{max(token_counts)}")

    print("\n--- Sample flat chunks ---")
    for c in chunks[:3]:
        print(f"  id={c['chunk_id']}  pages={c['start_page']}-{c['end_page']}  "
              f"tokens={c['token_count']}")
        print(f"  text: {c['text'][:200]}")
        print()

    save_chunks(chunks, str(output_path))
