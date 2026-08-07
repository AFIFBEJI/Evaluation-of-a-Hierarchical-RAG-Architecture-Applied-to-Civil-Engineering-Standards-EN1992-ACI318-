"""
structrag/parsing/table_extractor.py
-------------------------------------
Phase 2b — Extract key tables from EC2 PDF using pdfplumber and convert
them to structured text chunks ready for ChromaDB indexing.

Why pdfplumber instead of PyMuPDF for tables?
pdfplumber uses a dedicated table-detection algorithm (based on line
geometry) that correctly identifies cell boundaries.  PyMuPDF's text
stream flattens table cells into a single run-on string — fine for prose,
broken for tabular data.

Tables extracted
----------------
Table 2.1N  — Partial safety factors γc, γs, γp (page 27, clause 2.4.2.4)
Table 3.1   — Concrete strength/deformation characteristics including εcu2
              (page 31, clause 3.1.2)
Table 4.1   — Exposure classes XC/XD/XS/XF/XA (page 49, clause 4.2)
Table 4.3N  — Structural classification S1-S6 vs exposure classes (page 52)
Table 4.4N  — Minimum cover c_min,dur (mm) S1-S6 × exposure classes (p 52)

Output format
-------------
Each table becomes one or more Node objects (type="table") with:
- Full markdown-style text (caption + headers + rows)
- clause_number pointing to the parent clause
- ancestor_path matching the clause hierarchy
- metadata["table_id"] = "2.1N", "3.1", etc.

These nodes are then fed through the normal chunker and indexer.
"""

from __future__ import annotations
import re
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from structrag.parsing.hierarchy_schema import Node

# ---------------------------------------------------------------------------
# Table definitions — each entry tells us where to find a table in the PDF
# ---------------------------------------------------------------------------
# page: 1-indexed PDF page number
# caption: the table caption text to use in the chunk
# table_id: e.g. "2.1N"
# clause_number: parent clause that references this table
# ancestor_path: full breadcrumb
# postprocess: optional function name to clean the extracted rows

TABLE_SPECS = [
    {
        "table_id":    "2.1N",
        "caption":     "Tableau 2.1N : Coefficients partiels relatifs aux matériaux pour les états-limites ultimes",
        "page":        27,
        "clause_number": "2.4.2.4",
        "ancestor_path": (
            "Eurocode 2 > 2 BASES DE CALCUL > 2.4 Vérification par la méthode des "
            "coefficients partiels > 2.4.2 Valeurs de calcul > 2.4.2.4 Coefficients "
            "partiels relatifs aux matériaux"
        ),
        "search_text": "Tableau 2.1N",
        "manual_text": (
            "Tableau 2.1N : Coefficients partiels relatifs aux matériaux pour les états-limites ultimes\n\n"
            "| Situation de projet | γc (béton) | γs (acier béton armé) | γp (acier précontrainte) |\n"
            "|---------------------|-----------|----------------------|-------------------------|\n"
            "| Durable / Transitoire | 1,5 | 1,15 | 1,15 |\n"
            "| Accidentelle          | 1,2 | 1,0  | 1,0  |\n\n"
            "γc = coefficient partiel relatif au béton\n"
            "γs = coefficient partiel relatif à l'acier de béton armé\n"
            "γp = coefficient partiel relatif à l'acier de précontrainte\n"
            "Valeurs non valables pour le dimensionnement au feu (voir EN 1992-1-2).\n"
            "Pour la vérification à la fatigue : utiliser les valeurs de la situation durable."
        ),
        "use_manual": True,   # table is not detected by pdfplumber — use manual text
    },
    {
        "table_id":    "3.1",
        "caption":     "Tableau 3.1 : Caractéristiques de résistance et de déformation du béton",
        "page":        31,
        "clause_number": "3.1.2",
        "ancestor_path": "Eurocode 2 > 3 MATÉRIAUX > 3.1 Béton > 3.1.2 Résistance",
        "search_text": "Tableau 3.1",
        "use_manual": False,
        "table_index": 0,     # first table on the page
    },
    {
        "table_id":    "4.1",
        "caption":     "Tableau 4.1 : Classes d'exposition en fonction des conditions d'environnement, conformément à l'EN 206-1",
        "page":        49,
        "clause_number": "4.2",
        "ancestor_path": (
            "Eurocode 2 > 4 DURABILITÉ ET ENROBAGE DES ARMATURES > 4.2 Conditions d'environnement"
        ),
        "search_text": "Tableau 4.1",
        "use_manual": False,
        "table_index": 0,
    },
    {
        "table_id":    "4.3N",
        "caption":     "Tableau 4.3N : Classification structurale recommandée",
        "page":        52,
        "clause_number": "4.4.1.2",
        "ancestor_path": (
            "Eurocode 2 > 4 DURABILITÉ ET ENROBAGE DES ARMATURES > 4.4 Méthodes de vérification "
            "> 4.4.1 Enrobage > 4.4.1.2 Enrobage minimal"
        ),
        "search_text": "Tableau 4.3N",
        "use_manual": False,
        "table_index": 0,
    },
    {
        "table_id":    "4.4N",
        "caption":     "Tableau 4.4N : Valeurs de l'enrobage minimal c_min,dur (mm) requis vis-à-vis de la durabilité pour les armatures de béton armé",
        "page":        52,
        "clause_number": "4.4.1.2",
        "ancestor_path": (
            "Eurocode 2 > 4 DURABILITÉ ET ENROBAGE DES ARMATURES > 4.4 Méthodes de vérification "
            "> 4.4.1 Enrobage > 4.4.1.2 Enrobage minimal"
        ),
        "search_text": "Tableau 4.4N",
        "use_manual": False,
        "table_index": 1,
    },
]

# ---------------------------------------------------------------------------
# Row cleaning helpers
# ---------------------------------------------------------------------------

def _clean_cell(cell: Optional[str]) -> str:
    if cell is None:
        return ""
    # collapse whitespace and newlines inside a cell
    return re.sub(r'\s+', ' ', str(cell)).strip()


def _table_to_markdown(rows: list[list], caption: str) -> str:
    """Convert a list-of-lists table to markdown with a caption header."""
    if not rows:
        return caption

    # Clean all cells
    clean = [[_clean_cell(c) for c in row] for row in rows]

    # Determine column count (max width across all rows)
    ncols = max(len(row) for row in clean)

    # Pad rows to same width
    for row in clean:
        while len(row) < ncols:
            row.append("")

    lines = [caption, ""]

    # First row as header
    header = clean[0]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * ncols) + "|")

    # Data rows
    for row in clean[1:]:
        # Skip fully empty rows
        if all(c == "" for c in row):
            continue
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def _fix_table_31_rotation(rows: list[list]) -> list[list]:
    """
    Table 3.1 in this PDF has rotated header text that pdfplumber reads
    character-by-character in reverse (right-to-left).  The actual data
    cells are numeric and readable.  We reconstruct the table with
    known column headers from the EC2 standard.
    """
    # Known column headers for Table 3.1 (from EC2 §3.1.2)
    headers = [
        "fck (MPa)", "fck,cube (MPa)", "fcm (MPa)",
        "fctm (MPa)", "fctk,0,05 (MPa)", "fctk,0,95 (MPa)",
        "Ecm (GPa)",
        "εc1 (‰)", "εcu1 (‰)",
        "εc2 (‰)", "εcu2 (‰)",
        "n",
        "εc3 (‰)", "εcu3 (‰)",
    ]

    # The strength class row label is in col 0, values in cols 1..
    # Last row in pdfplumber output contains the backwards header text — skip it
    # Filter rows: keep rows where col[1] looks like a number
    data_rows = []
    for row in rows:
        # skip header-text rows (non-numeric first meaningful cell)
        if len(row) < 2:
            continue
        val = _clean_cell(row[1])
        if val and (val.replace(',', '').replace('.', '').isdigit() or val in ("", " ")):
            data_rows.append(row)
        elif _clean_cell(row[0]) in ("notéb", "sessalC", ""):
            continue  # reversed header text

    # Build clean rows: [strength class, val1, val2, ...]
    # Strength class values from EC2 Table 3.1: C12, C16, C20, C25, C30, C35, C40, C45, C50, C55, C60, C70, C80, C90
    strength_classes = [
        "C12/15", "C16/20", "C20/25", "C25/30", "C30/37",
        "C35/45", "C40/50", "C45/55", "C50/60", "C55/67",
        "C60/75", "C70/85", "C80/95", "C90/105"
    ]

    # Values from EC2 Table 3.1 (hardcoded from the standard since extraction is unreliable)
    # Format: fck, fck_cube, fcm, fctm, fctk005, fctk095, Ecm, ec1, ecu1, ec2, ecu2, n, ec3, ecu3
    table_31_values = [
        (12,  15,  20,  "1,6",  "1,1",  "2,0",  27,  "1,8",  "3,5",  "1,8",  "3,5",  "2,0",  "1,75", "3,5"),
        (16,  20,  24,  "1,9",  "1,3",  "2,5",  29,  "1,9",  "3,5",  "1,9",  "3,5",  "2,0",  "1,75", "3,5"),
        (20,  25,  28,  "2,2",  "1,5",  "2,9",  30,  "2,0",  "3,5",  "2,0",  "3,5",  "2,0",  "1,75", "3,5"),
        (25,  30,  33,  "2,6",  "1,8",  "3,3",  31,  "2,1",  "3,5",  "2,0",  "3,5",  "2,0",  "1,75", "3,5"),
        (30,  37,  38,  "2,9",  "2,0",  "3,8",  33,  "2,2",  "3,5",  "2,0",  "3,5",  "2,0",  "1,75", "3,5"),
        (35,  45,  43,  "3,2",  "2,2",  "4,2",  34,  "2,25", "3,5",  "2,0",  "3,5",  "2,0",  "1,75", "3,5"),
        (40,  50,  48,  "3,5",  "2,5",  "4,6",  35,  "2,3",  "3,5",  "2,0",  "3,5",  "2,0",  "1,75", "3,5"),
        (45,  55,  53,  "3,8",  "2,7",  "4,9",  36,  "2,4",  "3,5",  "2,0",  "3,5",  "2,0",  "1,75", "3,5"),
        (50,  60,  58,  "4,1",  "2,9",  "5,3",  37,  "2,45", "3,5",  "2,0",  "3,5",  "2,0",  "1,75", "3,5"),
        (55,  67,  63,  "4,2",  "3,0",  "5,5",  38,  "2,5",  "3,2",  "2,2",  "3,1",  "1,75", "1,8",  "3,1"),
        (60,  75,  68,  "4,4",  "3,1",  "5,7",  39,  "2,6",  "3,0",  "2,3",  "2,9",  "1,6",  "1,9",  "2,9"),
        (70,  85,  78,  "4,6",  "3,2",  "6,0",  41,  "2,7",  "2,8",  "2,4",  "2,7",  "1,45", "2,0",  "2,7"),
        (80,  95,  88,  "4,8",  "3,4",  "6,3",  42,  "2,8",  "2,8",  "2,5",  "2,6",  "1,4",  "2,2",  "2,6"),
        (90, 105,  98,  "5,0",  "3,5",  "6,6",  44,  "2,8",  "2,8",  "2,6",  "2,6",  "1,4",  "2,3",  "2,6"),
    ]

    # Build result: header row + data rows
    result = [["Classe de béton"] + headers]
    for cls, vals in zip(strength_classes, table_31_values):
        result.append([cls] + [str(v) for v in vals])

    return result


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------

def extract_tables(
    pdf_path: str,
    source_id: str = "ec2_2004_nf",
) -> list[Node]:
    """
    Extract all specified tables from the PDF and return as Node objects.

    Each Node:
    - type = "table"
    - text = markdown-formatted table (caption + header + rows)
    - clause_number = parent clause
    - ancestor_path = full breadcrumb + " > [Table X.YN]"
    - metadata["table_id"] = "2.1N" etc.
    """
    import pdfplumber

    nodes: list[Node] = []
    node_counter = 0

    with pdfplumber.open(pdf_path) as pdf:
        for spec in TABLE_SPECS:
            table_id  = spec["table_id"]
            caption   = spec["caption"]
            page_num  = spec["page"]         # 1-indexed
            clause    = spec["clause_number"]
            path      = spec["ancestor_path"]

            if spec.get("use_manual"):
                # Table 2.1N: pdfplumber can't detect it — use the known values
                text = spec["manual_text"]
            else:
                page = pdf.pages[page_num - 1]
                tables = page.extract_tables()
                t_idx = spec.get("table_index", 0)

                if not tables or t_idx >= len(tables):
                    print(f"  WARNING: Table {table_id} not found on page {page_num} "
                          f"(found {len(tables)} tables)")
                    # Still create a stub node with caption only
                    text = f"{caption}\n\n[Table content not extractable from PDF — consult source document]"
                else:
                    rows = tables[t_idx]

                    # Special handling for Table 3.1 (rotated headers)
                    if table_id == "3.1":
                        rows = _fix_table_31_rotation(rows)

                    text = _table_to_markdown(rows, caption)

            node_counter += 1
            node_id = f"{source_id}_table_{table_id.replace('.', '_')}"

            node = Node(
                id=node_id,
                source_id=source_id,
                type="table",
                level=3,   # below subclause level
                title=caption,
                clause_number=clause,
                text=text,
                parent_id=f"{source_id}_subclause_{clause}" if '.' in clause and clause.count('.') >= 2
                          else f"{source_id}_clause_{clause}",
                ancestor_path=path + f" > [Tableau {table_id}]",
                page_number=page_num,
                cross_refs=[],
                metadata={"table_id": table_id, "extracted_by": "pdfplumber"},
            )
            nodes.append(node)
            print(f"  Extracted Table {table_id}: {len(text)} chars, page {page_num}")

    return nodes


# ---------------------------------------------------------------------------
# Entry point — standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import io
    import sys
    # Force UTF-8 output
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    pdf_path = PROJECT_ROOT / "FA039724-NF EN 1992-1-1 (octobre 2005) Eurocode 2.pdf"
    print(f"Extracting tables from: {pdf_path.name}\n")

    nodes = extract_tables(str(pdf_path))

    print(f"\nExtracted {len(nodes)} table nodes:\n")
    for n in nodes:
        print(f"  {n.id}")
        print(f"  clause: {n.clause_number}  page: {n.page_number}")
        print(f"  text ({len(n.text)} chars):")
        # print first 600 chars of each table
        for line in n.text.split("\n")[:20]:
            print(f"    {line}")
        print()
