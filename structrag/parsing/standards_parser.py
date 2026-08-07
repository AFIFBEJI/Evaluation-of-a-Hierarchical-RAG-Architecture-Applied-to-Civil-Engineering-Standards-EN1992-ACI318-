"""
standards_parser.py
-------------------
Parses the native hierarchy of structural engineering standards (EC2, ACI 318)
from born-digital PDFs into a list of Node objects defined in hierarchy_schema.py.

Supported document: NF EN 1992-1-1:2004 (and any similarly formatted EC2 edition).
The same parser will handle ACI 318 with minor configuration changes (see ACI_CONFIG).

How it works
------------
1. Open the PDF with PyMuPDF (fitz) — chosen because it gives us per-span font
   size and bold flag, which are the primary heading signals in EC2.

2. Page-by-page, extract every text span with its font metadata.

3. Classify each span using the heading rules discovered during the manual skim
   (see inspect_pdf2.py output):

     sz=12 bold  → Section heading  (level 0)  e.g. "Section 6 ÉTATS-LIMITES ULTIMES"
     sz=11 bold  → Clause           (level 1)  e.g. "6.2  Effort tranchant"
     sz=10 bold  → Sub-clause       (level 2+) e.g. "6.2.2", "5.8.3.1"
     sz=10 norm  → Body text / paragraph
     sz=9  norm  → Notes (NOTE, NOTE 1, ...)
     sz≤8        → Subscript labels, formula indices — treated as inline text

4. The clause number and title are extracted on consecutive spans of the same
   style (the number comes first, the title follows on the same or next line).

5. Each heading opens a new Node.  Body text accumulates into the current
   deepest open node until the next heading is detected.

6. Paragraphs tagged "(1)P", "(2)P" etc. are recorded as child paragraph Nodes
   so their tag is preserved in metadata (useful for principle vs. rule queries).

7. The output is a flat list of Nodes.  Parent-child relationships are encoded
   via parent_id and ancestor_path — the tree structure is implicit, not nested,
   so the list serialises directly to JSON without recursion issues.

Usage
-----
    python standards_parser.py

Output
------
    structrag/data/parsed/ec2_2004_nf_nodes.json
"""

import re
import sys
import os
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Make sure the project root is on the path so we can import hierarchy_schema
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import fitz  # PyMuPDF
from structrag.parsing.hierarchy_schema import Node, save_nodes

# ---------------------------------------------------------------------------
# Document configuration
# ---------------------------------------------------------------------------
# EC2 NF edition — values derived from the manual PDF skim
EC2_CONFIG = {
    "source_id": "ec2_2004_nf",
    "document_title": "Eurocode 2",
    # Pages to skip at the start (cover + TOC pages, 0-indexed)
    # TOC runs from page 0 to ~page 16 (pages 1-17 in human numbering)
    "skip_pages_before": 17,
    # Font size thresholds
    "size_section": 12.0,   # "Section N  TITLE"
    "size_clause": 11.0,    # "N.M  Title"
    "size_subclause": 10.0, # "N.M.K(.L)  Title"
    "size_body": 10.0,
    "size_note": 9.0,
    # Regex that matches clause / subclause numbers
    "clause_number_re": re.compile(r'^\d+(\.\d+)+$'),
    # Regex that matches section header: "Section 6" or "Section 12"
    "section_re": re.compile(r'^Section\s+\d+$', re.IGNORECASE),
    # Regex for paragraph tags like (1)P, (2), (1)P, (2)P etc.
    "paragraph_tag_re": re.compile(r'^\(\d+\)P?$'),
}

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _is_bold(span: dict) -> bool:
    """Return True if the span is bold, using both the flags bitmask and
    the font name.  EC2 uses Helvetica-Bold which sets the font-name signal;
    some PDFs only set the flags bit (bit 4 = 0x10 = 16)."""
    return bool(span["flags"] & 16) or "Bold" in span["font"] or "bold" in span["font"]


def _clean_text(text: str) -> str:
    """Strip leading/trailing whitespace and collapse internal runs of
    whitespace / newlines to a single space."""
    return re.sub(r'\s+', ' ', text).strip()


def _fix_encoding(text: str) -> str:
    """
    The AFNOR PDF was produced by iText 1.4.1 and some characters come through
    as Latin-1 mojibake when PyMuPDF decodes the stream as UTF-8.
    Rather than bringing in a heavy dependency like ftfy, we apply a targeted
    mapping for the characters we observed in the inspection output.
    """
    replacements = {
        '╔': 'É',  'Ú': 'é',  'Ó': 'à',  'Û': 'ê',  'Ô': 'â',
        'Þ': 'è',  'þ': 'è',  '¯': 'ï',  'Ö': 'ô',  '½': '«',
        '╗': '»',  '£': 'œ',  'ÆO': 'œ', 'Æ': "'",  'ÿ': 'ÿ',
        '╩': 'Ê',  '²': '²',  '¹': 'ù',  'Ý': 'î',  'Ü': 'û',
        '¼': 'ü',  'ß': 'à',
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return text


def _make_node_id(source_id: str, node_type: str, identifier: str) -> str:
    """Build a stable, filesystem-safe node ID."""
    safe = re.sub(r'[^a-zA-Z0-9_.\-]', '_', identifier)
    return f"{source_id}_{node_type}_{safe}"


def _build_ancestor_path(stack: list[Node], current_title: str) -> str:
    """Build breadcrumb string from the open-node stack plus the current heading."""
    parts = []
    for node in stack:
        label = f"{node.clause_number} {node.title}".strip() if node.clause_number else node.title
        parts.append(label)
    parts.append(current_title)
    return " > ".join(parts)


# ---------------------------------------------------------------------------
# Core span classifier
# ---------------------------------------------------------------------------

def _classify_span(span: dict, cfg: dict) -> str:
    """
    Return one of:
        'section_number'   — "Section 6"
        'clause_number'    — "6.2"
        'subclause_number' — "6.2.2" or "5.8.3.1"
        'heading_title'    — bold text that is the title after a number
        'paragraph_tag'    — "(1)P", "(2)", etc.
        'note_start'       — starts with "NOTE"
        'body'             — plain body text
        'skip'             — page header/footer, subscript, too short
    """
    text = _clean_text(span["text"])
    size = round(span["size"], 1)
    bold = _is_bold(span)

    if not text or len(text) < 2:
        return 'skip'

    # Page header/footer pattern: short numeric page refs or "EN 1992-1-1:2004"
    if re.match(r'^Page\s+\d+$', text) or text == 'EN 1992-1-1:2004':
        return 'skip'

    # Section marker: "Section 6"
    if size >= cfg["size_section"] and bold and cfg["section_re"].match(text):
        return 'section_number'

    # Section all-caps title (follows section number on same or next span)
    if size >= cfg["size_section"] and bold:
        return 'heading_title'

    # Clause number: X.Y  (level 1)
    if size >= cfg["size_clause"] and bold and cfg["clause_number_re"].match(text):
        # Exactly one dot → clause level
        if text.count('.') == 1:
            return 'clause_number'
        else:
            return 'subclause_number'

    # Clause/subclause title (bold text at clause font size, not a number)
    if size >= cfg["size_clause"] and bold:
        return 'heading_title'

    # Sub-clause number at body font size, bold: X.Y.Z or X.Y.Z.W
    if size == cfg["size_subclause"] and bold and cfg["clause_number_re"].match(text):
        return 'subclause_number'

    # Sub-clause title
    if size == cfg["size_subclause"] and bold:
        return 'heading_title'

    # Paragraph tag
    if cfg["paragraph_tag_re"].match(text):
        return 'paragraph_tag'

    # Note
    if size <= cfg["size_note"] and text.startswith("NOTE"):
        return 'note_start'

    # Subscript / formula indices — too small to be body text
    if size < cfg["size_note"]:
        return 'skip'

    return 'body'


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

def parse_standard(pdf_path: str, cfg: dict) -> list[Node]:
    """
    Parse a standards PDF and return a flat list of Node objects.

    The algorithm maintains a 'stack' of currently open ancestor nodes.
    When a new heading is detected at level N, all stack entries at level >= N
    are popped (closed), then a new node is pushed.

    This mirrors how you'd mentally read the document:
    opening Section 6 closes whatever was open in Section 5,
    opening 6.2 closes 6.1, opening 6.2.2 closes 6.2.1, etc.
    """
    doc = fitz.open(pdf_path)
    source_id = cfg["source_id"]
    nodes: list[Node] = []

    # Stack of currently open ancestor Nodes (section → clause → subclause)
    # Each entry is a Node that has been started but not yet closed.
    ancestor_stack: list[Node] = []

    # Counters for generating unique IDs for un-numbered nodes
    para_counter = 0
    note_counter = 0
    section_counter = 0

    # State machine: track what we last saw to assemble multi-span headings
    pending_type: Optional[str] = None   # 'section' | 'clause' | 'subclause'
    pending_number: Optional[str] = None # e.g. "6.2.2"
    pending_title_parts: list[str] = []  # accumulates title spans

    # Accumulated body text for the current deepest node
    current_body_parts: list[str] = []
    current_para_tag: Optional[str] = None

    def flush_body():
        """Attach accumulated body text to the deepest open node as a
        paragraph Node, then reset the accumulator."""
        nonlocal current_body_parts, current_para_tag, para_counter
        if not current_body_parts:
            return
        body_text = _clean_text(" ".join(current_body_parts))
        if not body_text:
            current_body_parts = []
            current_para_tag = None
            return

        parent = ancestor_stack[-1] if ancestor_stack else None
        para_counter += 1
        para_id = _make_node_id(source_id, "paragraph", str(para_counter))

        node = Node(
            id=para_id,
            source_id=source_id,
            type="paragraph",
            level=(parent.level + 1) if parent else 1,
            title="",
            clause_number=None,
            text=body_text,
            parent_id=parent.id if parent else None,
            ancestor_path=parent.ancestor_path if parent else cfg["document_title"],
            page_number=current_page,
            metadata={"paragraph_tag": current_para_tag} if current_para_tag else {},
        )
        nodes.append(node)
        current_body_parts = []
        current_para_tag = None

    def flush_pending_heading(page_num: int):
        """
        Finalise a heading that was being assembled across multiple spans
        and push it onto the ancestor stack.
        """
        nonlocal pending_type, pending_number, pending_title_parts, section_counter

        if pending_type is None:
            return

        title = _clean_text(" ".join(pending_title_parts))
        num = pending_number

        # Determine level from type
        if pending_type == 'section':
            level = 0
            node_type = "section"
            section_counter += 1
            node_id = _make_node_id(source_id, "section", num or str(section_counter))
        elif pending_type == 'clause':
            level = 1
            node_type = "clause"
            node_id = _make_node_id(source_id, "clause", num)
        else:  # subclause
            # depth = number of dots in the clause number
            level = num.count('.') if num else 2
            node_type = "subclause"
            node_id = _make_node_id(source_id, "subclause", num)

        # Pop anything on the stack at the same or deeper level
        while ancestor_stack and ancestor_stack[-1].level >= level:
            ancestor_stack.pop()

        parent = ancestor_stack[-1] if ancestor_stack else None

        # Build ancestor path label for this new node
        if num and title:
            self_label = f"{num} {title}"
        elif num:
            self_label = num
        else:
            self_label = title

        if parent:
            ancestor_path = parent.ancestor_path + " > " + self_label
        else:
            ancestor_path = cfg["document_title"] + " > " + self_label

        node = Node(
            id=node_id,
            source_id=source_id,
            type=node_type,
            level=level,
            title=title,
            clause_number=num,
            text="",  # direct text filled in later via paragraph nodes
            parent_id=parent.id if parent else None,
            ancestor_path=ancestor_path,
            page_number=page_num,
        )
        nodes.append(node)
        ancestor_stack.append(node)

        # Reset pending state
        pending_type = None
        pending_number = None
        pending_title_parts = []

    current_page = 1

    # -------------------------------------------------------------------
    # Page loop
    # -------------------------------------------------------------------
    for page_num in range(cfg["skip_pages_before"], len(doc)):
        current_page = page_num + 1  # 1-indexed
        page = doc[page_num]
        blocks = page.get_text("dict")["blocks"]

        for block in blocks:
            # Image blocks → mark as table/figure candidate
            if block["type"] == 1:
                # We treat image blocks as potential tables or figures.
                # They are attached to the current deepest ancestor.
                flush_body()
                parent = ancestor_stack[-1] if ancestor_stack else None
                img_id = _make_node_id(source_id, "image", f"p{current_page}_b{block.get('number',0)}")
                img_node = Node(
                    id=img_id,
                    source_id=source_id,
                    type="table",   # refined later when we can inspect content
                    level=(parent.level + 1) if parent else 1,
                    title="[image/table block]",
                    clause_number=None,
                    text="",
                    parent_id=parent.id if parent else None,
                    ancestor_path=parent.ancestor_path if parent else cfg["document_title"],
                    page_number=current_page,
                    metadata={"block_bbox": block["bbox"]},
                )
                nodes.append(img_node)
                continue

            if block["type"] != 0:
                continue

            for line in block["lines"]:
                for span in line["spans"]:
                    raw_text = span["text"]
                    text = _fix_encoding(_clean_text(raw_text))
                    if not text:
                        continue

                    span_copy = dict(span)
                    span_copy["text"] = text
                    label = _classify_span(span_copy, cfg)

                    if label == 'skip':
                        continue

                    # -----------------------------------------------
                    # Section number detected: "Section 6"
                    # -----------------------------------------------
                    if label == 'section_number':
                        flush_body()
                        flush_pending_heading(current_page)
                        pending_type = 'section'
                        # Extract the number from "Section 6"
                        m = re.search(r'\d+', text)
                        pending_number = m.group() if m else text
                        pending_title_parts = []

                    # -----------------------------------------------
                    # Clause number: "6.2"
                    # -----------------------------------------------
                    elif label == 'clause_number':
                        flush_body()
                        flush_pending_heading(current_page)
                        pending_type = 'clause'
                        pending_number = text
                        pending_title_parts = []

                    # -----------------------------------------------
                    # Sub-clause number: "6.2.2" or "5.8.3.1"
                    # -----------------------------------------------
                    elif label == 'subclause_number':
                        flush_body()
                        flush_pending_heading(current_page)
                        pending_type = 'subclause'
                        pending_number = text
                        pending_title_parts = []

                    # -----------------------------------------------
                    # Heading title text (follows a number span)
                    # -----------------------------------------------
                    elif label == 'heading_title':
                        if pending_type is not None:
                            # This is the title belonging to the pending heading
                            pending_title_parts.append(text)
                        else:
                            # Bold text with no preceding number — treat as body
                            current_body_parts.append(text)

                    # -----------------------------------------------
                    # Paragraph tag: "(1)P", "(2)", etc.
                    # -----------------------------------------------
                    elif label == 'paragraph_tag':
                        # A new paragraph tag means a new paragraph node.
                        # Flush the previous paragraph first.
                        flush_body()
                        # Flush any pending heading too (shouldn't happen, but safe)
                        flush_pending_heading(current_page)
                        current_para_tag = text

                    # -----------------------------------------------
                    # Note start: "NOTE", "NOTE 1", "NOTE 2"
                    # -----------------------------------------------
                    elif label == 'note_start':
                        flush_body()
                        flush_pending_heading(current_page)
                        note_counter += 1
                        parent = ancestor_stack[-1] if ancestor_stack else None
                        note_node = Node(
                            id=_make_node_id(source_id, "note", str(note_counter)),
                            source_id=source_id,
                            type="note",
                            level=(parent.level + 1) if parent else 1,
                            title=text,
                            clause_number=None,
                            text=text,
                            parent_id=parent.id if parent else None,
                            ancestor_path=(parent.ancestor_path if parent else cfg["document_title"]) + " > [NOTE]",
                            page_number=current_page,
                        )
                        nodes.append(note_node)
                        current_body_parts = []  # note text accumulates separately

                    # -----------------------------------------------
                    # Body text
                    # -----------------------------------------------
                    elif label == 'body':
                        # If we were building a heading, this span's arrival
                        # means the heading title is done — flush it first.
                        if pending_type is not None:
                            flush_pending_heading(current_page)
                        current_body_parts.append(text)

    # End of document — flush whatever is still pending
    flush_body()
    flush_pending_heading(current_page)

    return nodes


# ---------------------------------------------------------------------------
# Extract cross-references (Phase 4 prep)
# ---------------------------------------------------------------------------

# Pattern matches clause references like "6.2.2", "§6.2.2", "clause 6.2.2"
_XREF_RE = re.compile(
    r'(?:§\s*|clause\s+|article\s+)?(\d+(?:\.\d+){1,3})',
    re.IGNORECASE
)

def extract_cross_refs(nodes: list[Node]) -> list[Node]:
    """
    Post-processing pass: scan each node's text for clause number patterns
    and populate node.cross_refs.  This is a lightweight pass that runs
    immediately after parsing; the actual linking (connecting cross_refs
    to real node IDs in other documents) is done by cross_reference_linker.py.
    """
    for node in nodes:
        if node.type in ("paragraph", "note") and node.text:
            refs = _XREF_RE.findall(node.text)
            node.cross_refs = list(dict.fromkeys(refs))  # deduplicated, ordered
    return nodes


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Resolve paths relative to project root
    pdf_path = PROJECT_ROOT / "FA039724-NF EN 1992-1-1 (octobre 2005) Eurocode 2.pdf"
    output_dir = PROJECT_ROOT / "structrag" / "data" / "parsed"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "ec2_2004_nf_nodes.json"

    if not pdf_path.exists():
        print(f"ERROR: PDF not found at {pdf_path}")
        sys.exit(1)

    print(f"Parsing: {pdf_path.name}")
    print(f"Skipping first {EC2_CONFIG['skip_pages_before']} pages (cover + TOC)")

    nodes = parse_standard(str(pdf_path), EC2_CONFIG)
    nodes = extract_cross_refs(nodes)

    # ---------------------------------------------------------------
    # Summary statistics
    # ---------------------------------------------------------------
    from collections import Counter
    type_counts = Counter(n.type for n in nodes)
    print(f"\nParsed {len(nodes)} total nodes:")
    for node_type, count in sorted(type_counts.items()):
        print(f"  {node_type:12s}  {count}")

    # ---------------------------------------------------------------
    # Validation sample — print 10 nodes from the middle of Section 6
    # ---------------------------------------------------------------
    print("\n--- Sample nodes (first 15 non-paragraph) ---")
    shown = 0
    for node in nodes:
        if node.type not in ("paragraph", "note", "image", "table"):
            print(f"  [{node.type:10s}] lv={node.level}  "
                  f"num={node.clause_number or '':8s}  "
                  f"p={node.page_number:3d}  "
                  f"{node.ancestor_path[:80]}")
            shown += 1
            if shown >= 15:
                break

    # ---------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------
    save_nodes(nodes, str(output_path))
    print(f"\nDone. Output: {output_path}")
