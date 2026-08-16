"""
structrag/evaluation/hallucination_guard.py
--------------------------------------------
Hallucination guardrail: detects when the model cites clause numbers that
don't exist in the indexed corpus.

Why this matters
----------------
In a safety-critical domain like structural engineering, a model that
confidently cites "per EC2 clause 7.2.4.1" when that clause doesn't exist is
more dangerous than one that says "I don't know". This guardrail catches
fabricated citations before they reach the user.

How it works
------------
1.  Build a set of known clause numbers from the parsed nodes JSON.
2.  After generation, extract all clause numbers cited in the answer
    (e.g. "clause 6.2.2", "EC2 4.4.1.2", "§3.1.7").
3.  Check each cited clause against the known set.
4.  Return a GuardrailResult with: is_clean flag, list of valid citations,
    list of hallucinated citations, and an annotated answer.

Usage
-----
    from structrag.evaluation.hallucination_guard import check_hallucination
    result = check_hallucination(answer_text, source_id="ec2_2004_nf")
    if not result.is_clean:
        print(f"WARNING: fabricated clauses: {result.hallucinated}")
"""

from __future__ import annotations
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

NODES_PATH = PROJECT_ROOT / "structrag" / "data" / "parsed" / "ec2_2004_nf_nodes.json"

# ---------------------------------------------------------------------------
# Known clause index — built once, cached in memory
# ---------------------------------------------------------------------------
_known_clauses: Optional[dict[str, set[str]]] = None   # source_id -> set of clause numbers


def _load_known_clauses(source_id: str = "ec2_2004_nf") -> set[str]:
    global _known_clauses
    if _known_clauses is None:
        _known_clauses = {}

    if source_id not in _known_clauses:
        nodes_file = (
            PROJECT_ROOT / "structrag" / "data" / "parsed" / f"{source_id}_nodes.json"
        )
        if not nodes_file.exists():
            nodes_file = NODES_PATH   # fallback to default

        if nodes_file.exists():
            with open(nodes_file, encoding="utf-8") as f:
                nodes = json.load(f)
            clauses = {
                n["clause_number"]
                for n in nodes
                if n.get("clause_number")
            }
            _known_clauses[source_id] = clauses
        else:
            _known_clauses[source_id] = set()

    return _known_clauses[source_id]


# ---------------------------------------------------------------------------
# Clause number extractor
# ---------------------------------------------------------------------------

# Patterns that indicate a clause citation in the model answer
_CITATION_PATTERNS = [
    # "clause 6.2.2", "Clause 4.4.1.2", "clause 6.2"
    re.compile(r'[Cc]lause\s+(\d+(?:\.\d+){1,4})'),
    # "EC2 6.2.2", "EC2 §6.2.2", "Eurocode 2 6.2.2"
    re.compile(r'(?:EC2|Eurocode\s*2)\s+(?:§\s*)?(\d+(?:\.\d+){1,4})'),
    # "§6.2.2", "§ 6.2.2"
    re.compile(r'§\s*(\d+(?:\.\d+){1,4})'),
    # "per 6.2.2", "see 6.2.2", "(6.2.2)", "[6.2.2]"
    re.compile(r'(?:per|see|cf\.?|ref\.?|in|from)[\s\(]+(\d+(?:\.\d+){2,4})'),
    # Table references: "Table 4.4N", "Tableau 3.1", "Table 2.1N"
    re.compile(r'(?:Table|Tableau)\s+(\d+\.\d+N?)'),
    # Bare clause numbers in bold: "**4.4.1.2**", "*4.4.1.2*"
    re.compile(r'\*{1,2}(\d+(?:\.\d+){2,4})\*{1,2}'),
]


def extract_cited_clauses(answer: str) -> list[str]:
    """Extract all clause numbers cited in an answer text."""
    found = set()
    for pattern in _CITATION_PATTERNS:
        for match in pattern.finditer(answer):
            found.add(match.group(1).strip())
    return sorted(found)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class GuardrailResult:
    is_clean:      bool          # True if no hallucinated clauses found
    cited:         list[str]     # all clause numbers found in the answer
    valid:         list[str]     # cited clauses that exist in the corpus
    hallucinated:  list[str]     # cited clauses NOT in the corpus
    annotated_answer: str        # answer with [HALLUCINATED: X.Y.Z] tags added
    source_id:     str = "ec2_2004_nf"

    def summary(self) -> str:
        if self.is_clean:
            if not self.cited:
                return "No clause citations found — no hallucinations possible."
            return f"All {len(self.cited)} cited clause(s) verified in corpus."
        return (
            f"WARNING: {len(self.hallucinated)} hallucinated clause(s): "
            f"{', '.join(self.hallucinated)}"
        )


# ---------------------------------------------------------------------------
# Main guard function
# ---------------------------------------------------------------------------

def check_hallucination(
    answer: str,
    source_id: str = "ec2_2004_nf",
) -> GuardrailResult:
    """
    Check whether all clause numbers cited in `answer` exist in the corpus.

    Parameters
    ----------
    answer    : the model-generated answer text
    source_id : which document's clause index to check against

    Returns
    -------
    GuardrailResult
    """
    known = _load_known_clauses(source_id)
    cited = extract_cited_clauses(answer)

    valid:        list[str] = []
    hallucinated: list[str] = []

    for clause in cited:
        # Exact match or prefix match
        # e.g. "6.2" is valid if "6.2" is in known
        # "Table 4.4N" — strip the N suffix for lookup
        lookup = clause.rstrip("N").rstrip("n")
        if clause in known or lookup in known:
            valid.append(clause)
        else:
            # Check if it's a direct sub-clause of a known clause
            # Only allow ONE level up (e.g. 6.2.2.1 is valid if 6.2.2 exists)
            # Do NOT allow jumping 3 levels (e.g. 7.9.4.12 should NOT be valid just because 7.9 exists)
            parts = clause.split(".")
            found = False
            # Only check immediate parent (one level up)
            if len(parts) > 1:
                parent = ".".join(parts[:-1])
                if parent in known:
                    valid.append(clause)
                    found = True
            if not found:
                hallucinated.append(clause)

    # Build annotated answer
    annotated = answer
    for h in hallucinated:
        annotated = re.sub(
            re.escape(h),
            f"{h}[⚠ NOT IN CORPUS]",
            annotated,
        )

    return GuardrailResult(
        is_clean=len(hallucinated) == 0,
        cited=cited,
        valid=valid,
        hallucinated=hallucinated,
        annotated_answer=annotated,
        source_id=source_id,
    )


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_cases = [
        # Good answer — real clauses
        ("The minimum cover is defined in clause 4.4.1.2 and depends on EC2 §4.4.1.1.", True),
        # Hallucinated clause
        ("Per EC2 clause 7.9.4.12, the cover must be at least 50mm.", False),
        # Mixed
        ("Per 4.4.1.2, the cover is cmin. See also clause 99.99.99 for details.", False),
        # No citations
        ("Concrete cover should be sufficient for durability.", True),
    ]

    for answer, expected_clean in test_cases:
        result = check_hallucination(answer)
        status = "PASS" if result.is_clean == expected_clean else "FAIL"
        print(f"[{status}] is_clean={result.is_clean}  cited={result.cited}")
        print(f"       valid={result.valid}  hallucinated={result.hallucinated}")
        print(f"       {result.summary()}")
        print()
