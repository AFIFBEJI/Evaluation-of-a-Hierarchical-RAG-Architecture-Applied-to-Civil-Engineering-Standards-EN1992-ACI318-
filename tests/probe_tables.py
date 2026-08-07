"""
tests/probe_tables.py
---------------------
Probe pdfplumber on the pages containing the four critical tables.
Run this BEFORE writing the extractor so we know exactly what pdfplumber sees.
Output -> tests/table_probe.txt
"""
import sys, os, json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

PDF = BASE / "FA039724-NF EN 1992-1-1 (octobre 2005) Eurocode 2.pdf"
OUT = BASE / "tests" / "table_probe.txt"

import pdfplumber

out = open(OUT, "w", encoding="utf-8")

def w(s=""):
    out.write(str(s) + "\n")

# Pages to probe (1-indexed) — from the inspection:
# Table 2.1N  -> clause 2.4.2.4 -> page 27
# Table 3.1   -> clause 3.1.2   -> page 30  (also spans 31?)
# Table 3.2   -> clause 3.1.7   -> page 36
# Table 4.3N  -> clause 4.3/4.4 -> page 50 area
# Table 4.4N  -> clause 4.4.1.2 -> page 51 area

PROBE_PAGES = {
    27:  "Table 2.1N — partial safety factors (gamma_c=1.5, gamma_s=1.15)",
    28:  "Page after Table 2.1N",
    30:  "Table 3.1 — concrete strength/deformation characteristics",
    31:  "Table 3.1 continued?",
    36:  "Table 3.2 — stress-strain parameters (epsilon_cu2)",
    37:  "Table 3.2 continued?",
    48:  "Table 4.1 — exposure classes (XC, XD, XF, XA)",
    49:  "Table 4.1 continued / Table 4.2?",
    50:  "Table 4.3N — structural classification / Table 4.4N — cover values",
    51:  "Table 4.4N cover values continued",
    52:  "Table 4.4N continued?",
}

w("=" * 70)
w("pdfplumber table probe — EC2 NF EN 1992-1-1:2004")
w("=" * 70)

with pdfplumber.open(str(PDF)) as pdf:
    for page_num, description in PROBE_PAGES.items():
        page = pdf.pages[page_num - 1]   # 0-indexed

        w()
        w(f"{'='*60}")
        w(f"PAGE {page_num} — {description}")
        w(f"{'='*60}")

        # --- raw text ---
        raw_text = page.extract_text() or ""
        w(f"\n[RAW TEXT ({len(raw_text)} chars)]")
        w(raw_text[:2000])

        # --- tables ---
        tables = page.extract_tables()
        w(f"\n[TABLES FOUND: {len(tables)}]")
        for t_idx, table in enumerate(tables):
            w(f"\n  Table {t_idx + 1} ({len(table)} rows):")
            for row in table:
                # clean None cells
                clean_row = [str(cell).strip() if cell else "" for cell in row]
                w(f"    {' | '.join(clean_row)}")

        # --- words with bounding boxes (for detecting table-like structures
        #     that pdfplumber doesn't pick up as formal tables) ---
        words = page.extract_words()
        w(f"\n[WORD COUNT: {len(words)}]")
        # show first 30 words with their x positions to detect columns
        for word in words[:30]:
            w(f"  x={word['x0']:6.1f}  y={word['top']:6.1f}  text={word['text']}")

out.close()
print(f"Done -> {OUT}")
