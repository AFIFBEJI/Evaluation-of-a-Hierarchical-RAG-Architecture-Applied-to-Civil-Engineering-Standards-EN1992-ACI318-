"""
structrag/retrieval/query_rewriter.py  — G3: Query Rewriter
-------------------------------------------------------------
Expands the user's question into a richer retrieval query by:
  1. Extracting EC2 symbols already in the query (via symbol_dictionary.py)
  2. Adding the EC2 clause numbers those symbols belong to
  3. Optionally calling a lightweight LLM (GPT-OSS-20B on Groq) to
     produce a technical search form of the question

The rewritten query is used ONLY for retrieval — the original question
is passed to the generation model.

Why this works:
  - "Comment calcule-t-on la résistance de calcul de l'acier ?" 
    → rewritten to → "fyd acier armature clause 3.2.7 résistance calcul"
  - "Calculer fcm pour fck=35 MPa"
    → rewritten to → "fcm fck résistance moyenne compression béton clause 3.1.2 fcm = fck + 8"
  
  The retriever then finds the exact chunk instead of a vague match.

Two modes:
  FAST mode (default): pure regex/symbol extraction, ~0ms, no API call
  LLM mode:  adds a Groq API call (~1-2s) for more complex rewrites

Integration: called from app.py and full_ragas_eval.py before retrieval.
"""

from __future__ import annotations
import re, os
from typing import Optional

from structrag.retrieval.symbol_dictionary import SYMBOLS, extract_symbols


# ---------------------------------------------------------------------------
# EC2 technical vocabulary for query expansion (no API call needed)
# ---------------------------------------------------------------------------

# Maps French/English question words to their EC2 technical equivalents
_CONCEPT_EXPANSIONS: dict[str, list[str]] = {
    # Materials — concrete tensile strength fractions
    r"fctk.*0[,.]?05|fctk.*inf[eé]rieure|fractile.*5.*traction": ["fctk,0,05", "0,70", "fctm", "3.1.2"],
    r"fctk.*0[,.]?95|fctk.*sup[eé]rieure|fractile.*95.*traction": ["fctk,0,95", "1,30", "fctm", "3.1.2"],
    r"module.*b[eé]ton.*instant.*t|Ecm.*\(t\)":  ["Ecm(t)", "fcm(t)", "s", "3.1.3"],

    # Materials — steel
    r"r[eé]sistance.*calcul.*acier":     ["fyd", "fyk", "γs", "3.2.7"],
    r"d[eé]formation.*ultime.*acier":    ["εud", "εuk", "0,9", "3.2.7"],
    r"acier.*calcul":                    ["fyd", "fyk", "γs"],
    r"limite.*[eé]lasticit[eé]":         ["fyk", "fyd", "3.2"],

    # Materials — concrete design values
    r"r[eé]sistance.*calcul.*compression.*b[eé]ton": ["fcd", "αcc", "γc", "3.1.6"],
    r"r[eé]sistance.*calcul.*b[eé]ton":  ["fcd", "fctd", "3.1.6"],
    r"r[eé]sistance.*moyenne.*compression": ["fcm", "fck", "8 MPa", "3.1.2"],
    r"r[eé]sistance.*moyenne.*traction":  ["fctm", "0,30", "fck", "3.1.2"],
    r"module.*[eé]lasticit[eé].*b[eé]ton": ["Ecm", "fcm", "22", "3.1.3"],
    r"module.*s[eé]cant":                ["Ecm", "3.1.3"],
    r"d[eé]formation.*ultime.*compression": ["εcu2", "3,5", "3.1.7"],

    # Partial factors
    r"coefficient.*partiel.*b[eé]ton":   ["γc", "1,5", "2.4.2.4"],
    r"coefficient.*partiel.*acier":      ["γs", "1,15", "2.4.2.4"],
    r"facteur.*s[eé]curit[eé].*b[eé]ton": ["γc", "1,5", "2.4.2.4"],

    # Cover
    r"enrobage nominal":                 ["cnom", "cmin", "Δcdev", "4.4.1.1"],
    r"enrobage minimal":                 ["cmin", "cmin,dur", "cmin,b", "4.4.1.2"],
    r"tol[eé]rance.*ex[eé]cution":       ["Δcdev", "10 mm", "4.4.1.3"],

    # ULS hypotheses / why tension neglected
    r"traction.*b[eé]ton.*n[eé]glig[eé]|pourquoi.*traction.*n[eé]glig|r[eé]sistance.*traction.*n[eé]glig": [
        "béton traction négligée", "hypothèse ELU", "εcu2", "section béton armé", "6.1"
    ],
    r"hypoth[eè]se.*[eé]tat limite ultime|hypoth[eè]ses.*ELU|ELU.*hypoth[eè]ses": [
        "Bernoulli", "traction béton négligée", "εcu2 3,5", "εud 0,9×εuk", "6.1"
    ],

    # Exposure class identification
    r"zone de marnage|marnage|zone.*marnage": ["XS3", "marnage", "eau de mer", "4.2"],
    r"immersion permanente.*mer|submerg.*eau.*mer": ["XS2", "immersion permanente", "4.2"],
    r"forte saturation.*d[eé]vergla[cç]age|saturation.*[eé]lev[eé].*d[eé]vergla[cç]": ["XF4", "déverglaçage", "4.2"],
    r"chlorures.*eau.*mer|corrosion.*eau.*mer": ["XS", "XS1", "XS2", "XS3", "4.2"],

    # Durability table dependency
    r"table.*durabilit[eé].*d[eé]pend|durabilit[eé].*param[eè]tre|valeur.*table.*durabilit[eé]": [
        "classe exposition", "classe structurale", "Tableau 4.4N", "cmin,dur", "4.4.1.2"
    ],
    r"profondeur.*zone comprim[eé]e":    ["x", "λx", "fcd", "6.1"],
    r"moment r[eé]sistant":              ["MRd", "As", "fyd", "z", "6.1"],
    r"[eé]tat limite ultime.*flexion":   ["ELU", "flexion", "x/d", "εcu2", "6.1"],

    # Shear
    r"r[eé]sistance.*cisaillement":      ["VRd,c", "CRd,c", "0,12", "6.2.2"],
    r"effort tranchant":                 ["VRd", "CRd,c", "k", "ρl", "6.2"],

    # Anchorage
    r"longueur.*ancrage":                ["lb,rqd", "fbd", "φ", "σsd", "8.4.3"],
    r"adh[eé]rence":                     ["fbd", "η1", "η2", "fctd", "8.4.2"],

    # Crack
    r"largeur.*fissure":                 ["wk", "sr,max", "εsm", "εcm", "7.3.4"],
    r"armature.*fissuration":            ["As,min", "fct,eff", "kc", "7.3.2"],

    # Detailing
    r"armature.*minimale.*poutre":       ["As,min", "fctm", "fyk", "9.2.1.1"],
    r"armature.*transversale.*minimale": ["ρw,min", "fck", "fyk", "9.2.2"],
    r"espacement.*armature":             ["sl,max", "0,75d", "9.2.2"],
}


def _fast_rewrite(query: str) -> str:
    """
    Expand query with EC2 symbols and clause numbers using only regex.
    No API call — runs in <1ms.

    Strategy:
    1. Find symbols already in the query → add their clause numbers
    2. Find concept patterns → add related symbols and clauses
    3. Deduplicate and join
    """
    additions = []

    # Step 1: symbols already in query → add their clause
    symbol_ids = extract_symbols(query)
    for sid in symbol_ids:
        entry = SYMBOLS[sid]
        # Add the formula itself and the clause number
        additions.append(entry["formula"])
        additions.append(f"clause {entry['clause']}")

    # Step 2: concept patterns → add related symbols
    q_lower = query.lower()
    for pattern, expansions in _CONCEPT_EXPANSIONS.items():
        if re.search(pattern, q_lower, re.IGNORECASE):
            additions.extend(expansions)

    if not additions:
        return query  # no expansion — return original

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for a in additions:
        if a not in seen:
            seen.add(a)
            unique.append(a)

    # Combine original query with technical expansions
    expansion_str = " | ".join(unique[:10])  # cap at 10 terms
    return f"{query} [{expansion_str}]"


def _llm_rewrite(query: str, api_key: str) -> str:
    """
    Use GPT-OSS-20B on Groq to rewrite the query as a technical EC2 search
    string. Falls back to fast_rewrite on any failure.

    Model note: gpt-oss-20b is a reasoning model — it needs max_tokens=256
    and a prompt that forces visible text output before the result.
    """
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        prompt = (
            "Task: Rewrite the question below as a short technical search query "
            "for Eurocode 2 (EN 1992-1-1).\n"
            "Rules:\n"
            "- Include EC2 symbol names exactly as written in the standard "
            "(e.g. fctk,0,05  fyd  εcu2  cmin,dur  VRd,c)\n"
            "- Include the relevant clause number if you know it\n"
            "- Use comma notation for decimals: 0,05 not 0.05\n"
            "- Output ONLY the search query, nothing else, max 25 words\n\n"
            f"Question: {query}\n"
            "Search query:"
        )
        resp = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,   # reasoning model needs headroom
            temperature=0.0,
        )
        rewritten = (resp.choices[0].message.content or "").strip()
        # Clean up: take only the last line if model added preamble
        lines = [l.strip() for l in rewritten.splitlines() if l.strip()]
        rewritten = lines[-1] if lines else ""

        if rewritten and len(rewritten) > 5:
            # Merge with fast expansion to catch any symbols the LLM missed
            fast = _fast_rewrite(query)
            if fast != query:
                return f"{rewritten} | {fast}"
            return rewritten
        # LLM returned empty — fall back
        return _fast_rewrite(query)

    except Exception:
        return _fast_rewrite(query)


def rewrite_query(
    query: str,
    use_llm: bool = True,
    api_key: Optional[str] = None,
) -> str:
    """
    Main entry point. Always runs the fast rewriter first.
    If use_llm=True AND fast rewrite produced no expansion, tries LLM.

    The fast rewriter covers ~80% of cases through symbol detection + concept
    patterns. The LLM handles the remaining cases where no symbol or pattern
    matched (novel phrasings, conceptual questions).

    Parameters
    ----------
    query    : original user question
    use_llm  : if True, fall back to LLM when fast rewrite has no expansion
    api_key  : Groq API key (read from GROQ_API_KEY env var if not given)

    Returns
    -------
    Expanded query for retrieval. Original question is unchanged — pass
    the original to the generation model.
    """
    # Always run fast rewrite first (zero latency)
    fast = _fast_rewrite(query)

    # If fast found symbols/expansions, use that — it's accurate and instant
    if fast != query:
        return fast

    # Fast found nothing — try LLM for conceptual/phrasing translations
    if use_llm:
        key = api_key or os.environ.get("GROQ_API_KEY", "")
        if key:
            return _llm_rewrite(query, key)

    return query
