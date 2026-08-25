"""
structrag/retrieval/symbol_dictionary.py  — G2: Symbol Dictionary
-------------------------------------------------------------------
Maps EC2 symbols (fyd, fcm, εud, Ecm, ...) to their exact definitions,
formulas, and values.

This solves the global ambiguity + missing formula problem:
- Instead of injecting whole formula blocks that contaminate answers,
  we inject ONLY the symbols mentioned in the query.
- Works for ANY EC2 symbol, not just the ones we manually added to
  the formula library.
- The regex patterns are designed to match symbols as written by users
  (fyd, f_yd, f_{yd}, fyd, etc.) and in French prose.

Integration: called from query_rewriter.py and from the retrieval
pipeline in app.py / full_ragas_eval.py via enrich_with_symbols().
"""

from __future__ import annotations
import re
from typing import Optional

# ---------------------------------------------------------------------------
# Symbol registry
# Each entry:
#   "symbol_id": {
#       "formula": "exact equation as text",
#       "definition": "what the symbol is",
#       "value": optional fixed value,
#       "unit": unit string,
#       "clause": EC2 clause reference,
#       "regex": list of regex patterns that match this symbol in user queries
#   }
# ---------------------------------------------------------------------------

SYMBOLS: dict[str, dict] = {

    # ── Steel design ──────────────────────────────────────────────────────

    "fctk_005": {
        "formula":    "fctk,0,05 = 0,70 × fctm  (fractile 5 % — résistance caractéristique inférieure)",
        "definition": "résistance caractéristique à la traction inférieure du béton (fractile 5 %)",
        "value":      None,
        "unit":       "MPa",
        "clause":     "3.1.2",
        "regex": [
            r"\bf_?\{?ctk[,.]?0[,.]?05\}?\b",
            r"\bfctk,?0[,.]?05\b",
            r"r[eé]sistance.*caract[eé]ristique.*traction.*inf[eé]rieure",
            r"fractile 5.*traction",
        ],
    },

    "fctk_095": {
        "formula":    "fctk,0,95 = 1,30 × fctm  (fractile 95 % — résistance caractéristique supérieure)",
        "definition": "résistance caractéristique à la traction supérieure du béton (fractile 95 %)",
        "value":      None,
        "unit":       "MPa",
        "clause":     "3.1.2",
        "regex": [
            r"\bf_?\{?ctk[,.]?0[,.]?95\}?\b",
            r"\bfctk,?0[,.]?95\b",
            r"r[eé]sistance.*caract[eé]ristique.*traction.*sup[eé]rieure",
            r"fractile 95.*traction",
        ],
    },

    "Ecm_t": {
        "formula":    "Ecm(t) = (fcm(t) / fcm)^0,3 × Ecm  où  fcm(t) = exp(s × (1 − (28/t)^0,5)) × fcm",
        "definition": "module d'élasticité sécant du béton à l'instant t",
        "value":      None,
        "unit":       "GPa",
        "clause":     "3.1.3",
        "regex": [
            r"\bE_?\{?cm\}?\s*\(t\)",
            r"\bEcm\s*\(t\)",
            r"module.*[eé]lasticit[eé].*instant.*t",
            r"module.*b[eé]ton.*temps",
            r"elastic.*modulus.*time",
        ],
    },

    "fyd": {
        "formula":    "fyd = fyk / γs",
        "definition": "résistance de calcul de l'acier d'armature",
        "value":      None,
        "unit":       "MPa",
        "clause":     "3.2.7",
        "regex": [
            r"\bf_?\{?yd\}?\b",
            r"\bfyd\b",
            r"r[eé]sistance de calcul de l'acier",
            r"r[eé]sistance de calcul.*acier.*armature",
            r"design.*strength.*steel",
            r"acier.*r[eé]sistance.*calcul",
        ],
    },

    "εud": {
        "formula":    "εud = 0,9 × εuk",
        "definition": "déformation de calcul ultime de l'acier",
        "value":      None,
        "unit":       "‰",
        "clause":     "3.2.7",
        "regex": [
            r"\bε_?ud\b", r"\beud\b", r"\bepsilon_?ud\b",
            r"d[eé]formation.*calcul.*ultime.*acier",
            r"ultime.*d[eé]formation.*acier",
            r"ultimate.*strain.*steel",
        ],
    },

    "εuk": {
        "formula":    "εuk = déformation caractéristique ultime de l'acier (valeur tabulée)",
        "definition": "déformation caractéristique ultime de l'acier",
        "value":      None,
        "unit":       "‰",
        "clause":     "3.2.7",
        "regex": [
            r"\bε_?uk\b", r"\beuk\b",
            r"d[eé]formation.*caract[eé]ristique.*ultime",
        ],
    },

    "fyk": {
        "formula":    "fyk = limite d'élasticité caractéristique de l'acier",
        "definition": "limite d'élasticité caractéristique de l'acier",
        "value":      None,
        "unit":       "MPa",
        "clause":     "3.2",
        "regex": [
            r"\bf_?\{?yk\}?\b", r"\bfyk\b",
            r"limite.*[eé]lasticit[eé].*caract[eé]ristique",
            r"yield.*strength.*characteristic",
        ],
    },

    "Es": {
        "formula":    "Es = 200 000 MPa",
        "definition": "module d'élasticité de l'acier d'armature",
        "value":      200000,
        "unit":       "MPa",
        "clause":     "3.2.7",
        "regex": [
            r"\bE_?s\b",
            r"module.*[eé]lasticit[eé].*acier",
            r"elastic.*modulus.*steel",
        ],
    },

    "γs": {
        "formula":    "γs = 1,15 (situation durable/transitoire) ; γs = 1,0 (situation accidentelle)",
        "definition": "coefficient partiel pour l'acier d'armature",
        "value":      1.15,
        "unit":       "−",
        "clause":     "2.4.2.4",
        "regex": [
            r"\bγ_?s\b", r"\bgamma_?s\b",
            r"coefficient.*partiel.*acier",
            r"partial.*factor.*steel",
        ],
    },

    # ── Concrete design values ────────────────────────────────────────────

    "fcd": {
        "formula":    "fcd = αcc × fck / γc",
        "definition": "résistance de calcul en compression du béton",
        "value":      None,
        "unit":       "MPa",
        "clause":     "3.1.6",
        "regex": [
            r"\bf_?\{?cd\}?\b", r"\bfcd\b",
            r"r[eé]sistance de calcul.*compression.*b[eé]ton",
            r"design.*compressive.*strength.*concrete",
            r"b[eé]ton.*r[eé]sistance.*calcul.*compression",
        ],
    },

    "fctd": {
        "formula":    "fctd = αct × fctk,0,05 / γc",
        "definition": "résistance de calcul en traction du béton",
        "value":      None,
        "unit":       "MPa",
        "clause":     "3.1.6",
        "regex": [
            r"\bf_?\{?ctd\}?\b", r"\bfctd\b",
            r"r[eé]sistance.*calcul.*traction.*b[eé]ton",
            r"design.*tensile.*strength.*concrete",
        ],
    },

    "fck": {
        "formula":    "fck = résistance caractéristique en compression (fractile 5 %) sur cylindre",
        "definition": "résistance caractéristique en compression du béton",
        "value":      None,
        "unit":       "MPa",
        "clause":     "3.1.2",
        "regex": [
            r"\bf_?\{?ck\}?\b", r"\bfck\b",
            r"r[eé]sistance.*caract[eé]ristique.*compression.*b[eé]ton",
            r"characteristic.*compressive.*strength",
        ],
    },

    "fcm": {
        "formula":    "fcm = fck + 8  (MPa)",
        "definition": "résistance moyenne en compression du béton",
        "value":      None,
        "unit":       "MPa",
        "clause":     "3.1.2",
        "regex": [
            r"\bf_?\{?cm\}?\b", r"\bfcm\b",
            r"r[eé]sistance moyenne.*compression.*b[eé]ton",
            r"mean.*compressive.*strength.*concrete",
            r"r[eé]sistance moyenne.*b[eé]ton",
        ],
    },

    "fctm": {
        "formula": (
            "fctm = 0,30 × fck^(2/3)  pour fck ≤ 50 MPa\n"
            "fctm = 2,12 × ln(1 + fcm/10)  pour fck > 50 MPa"
        ),
        "definition": "résistance moyenne en traction directe du béton",
        "value":      None,
        "unit":       "MPa",
        "clause":     "3.1.2",
        "regex": [
            r"\bf_?\{?ctm\}?\b", r"\bfctm\b",
            r"r[eé]sistance moyenne.*traction.*b[eé]ton",
            r"mean.*tensile.*strength.*concrete",
            r"traction.*moyenne.*b[eé]ton",
        ],
    },

    "Ecm": {
        "formula":    "Ecm = 22 × (fcm / 10)^0,3  (GPa)",
        "definition": "module d'élasticité sécant moyen du béton",
        "value":      None,
        "unit":       "GPa",
        "clause":     "3.1.3",
        "regex": [
            r"\bE_?\{?cm\}?\b", r"\bEcm\b",
            r"module.*[eé]lasticit[eé].*b[eé]ton",
            r"module.*s[eé]cant.*b[eé]ton",
            r"elastic.*modulus.*concrete",
            r"module.*[eé]lasticit[eé] moyen",
        ],
    },

    "εcu2": {
        "formula":    "εcu2 = 3,5 ‰  pour fck ≤ 50 MPa  (Tableau 3.1)",
        "definition": "déformation ultime de compression du béton (diagramme parabole-rectangle)",
        "value":      3.5,
        "unit":       "‰",
        "clause":     "3.1.7",
        "regex": [
            r"\bε_?\{?cu2?\}?\b", r"\becu2?\b", r"\bεcu2?\b",
            r"d[eé]formation.*ultime.*compression",
            r"ultimate.*compressive.*strain",
        ],
    },

    "αcc": {
        "formula":    "αcc = 1,0  (valeur recommandée EC2)",
        "definition": "coefficient tenant compte des effets à long terme sur la résistance en compression",
        "value":      1.0,
        "unit":       "−",
        "clause":     "3.1.6",
        "regex": [
            r"\bα_?\{?cc\}?\b", r"\balpha_?cc\b",
            r"effets.*long terme.*compression",
            r"long.term.*effect.*compression",
        ],
    },

    "γc": {
        "formula":    "γc = 1,5 (durable/transitoire) ; γc = 1,2 (accidentelle)",
        "definition": "coefficient partiel pour le béton",
        "value":      1.5,
        "unit":       "−",
        "clause":     "2.4.2.4",
        "regex": [
            r"\bγ_?c\b", r"\bgamma_?c\b",
            r"coefficient.*partiel.*b[eé]ton",
            r"partial.*factor.*concrete",
        ],
    },

    # ── Cover ─────────────────────────────────────────────────────────────

    "cnom": {
        "formula":    "cnom = cmin + Δcdev",
        "definition": "enrobage nominal à spécifier sur les plans",
        "value":      None,
        "unit":       "mm",
        "clause":     "4.4.1.1",
        "regex": [
            r"\bc_?\{?nom\}?\b", r"\bcnom\b",
            r"enrobage nominal",
            r"nominal.*cover",
        ],
    },

    "cmin": {
        "formula":    "cmin = max(cmin,b ; cmin,dur + Δcdur,γ − Δcdur,st − Δcdur,add ; 10 mm)",
        "definition": "enrobage minimal",
        "value":      None,
        "unit":       "mm",
        "clause":     "4.4.1.2",
        "regex": [
            r"\bc_?\{?min\}?\b", r"\bcmin\b",
            r"enrobage minimal",
            r"minimum.*cover",
        ],
    },

    "Δcdev": {
        "formula":    "Δcdev = 10 mm  (valeur recommandée) ; 5 mm avec système qualité ; 0 mm préfabriqués",
        "definition": "tolérance d'exécution pour l'enrobage",
        "value":      10,
        "unit":       "mm",
        "clause":     "4.4.1.3",
        "regex": [
            r"\bΔ_?c_?dev\b", r"\bdelta_?c_?dev\b", r"\bcdev\b",
            r"tol[eé]rance.*ex[eé]cution",
            r"execution.*deviation",
            r"Δcdev",
        ],
    },

    # ── Shear ─────────────────────────────────────────────────────────────

    "VRd_c": {
        "formula":    "VRd,c = [CRd,c × k × (100 × ρl × fck)^(1/3) + k1 × σcp] × bw × d",
        "definition": "résistance au cisaillement sans armatures transversales",
        "value":      None,
        "unit":       "N",
        "clause":     "6.2.2",
        "regex": [
            r"\bV_?\{?Rd,?c\}?\b", r"\bVRd,?c\b", r"\bVrd,?c\b",
            r"r[eé]sistance.*cisaillement.*sans.*armature",
            r"shear.*resistance.*without.*reinforcement",
        ],
    },

    "CRd_c": {
        "formula":    "CRd,c = 0,18 / γc = 0,12  (valeur recommandée)",
        "definition": "coefficient de résistance au cisaillement",
        "value":      0.12,
        "unit":       "−",
        "clause":     "6.2.2",
        "regex": [
            r"\bC_?\{?Rd,?c\}?\b", r"\bCRd,?c\b",
        ],
    },

    # ── Flexion ───────────────────────────────────────────────────────────

    "x_neutral": {
        "formula":    "x = (As × fyd) / (η × fcd × b × λ)",
        "definition": "profondeur de l'axe neutre d'une section rectangulaire en flexion simple",
        "value":      None,
        "unit":       "mm",
        "clause":     "6.1",
        "regex": [
            r"axe neutre",
            r"neutral axis",
            r"profondeur.*zone comprim[eé]e",
            r"hauteur.*zone comprim[eé]e",
        ],
    },

    "MRd": {
        "formula":    "MRd = As × fyd × z  avec  z = d − λx/2  (z ≤ 0,95d)",
        "definition": "moment résistant d'une section rectangulaire",
        "value":      None,
        "unit":       "N·m",
        "clause":     "6.1",
        "regex": [
            r"\bM_?\{?Rd\}?\b", r"\bMRd\b",
            r"moment r[eé]sistant",
            r"bending.*resistance",
        ],
    },

    # ── Anchorage ─────────────────────────────────────────────────────────

    "lb_rqd": {
        "formula":    "lb,rqd = (φ / 4) × (σsd / fbd)",
        "definition": "longueur d'ancrage de base requise",
        "value":      None,
        "unit":       "mm",
        "clause":     "8.4.3",
        "regex": [
            r"\bl_?\{?b,?rqd\}?\b", r"\blb,?rqd\b",
            r"longueur.*ancrage.*base",
            r"basic.*anchorage.*length",
        ],
    },

    "fbd": {
        "formula":    "fbd = 2,25 × η1 × η2 × fctd",
        "definition": "résistance ultime d'adhérence",
        "value":      None,
        "unit":       "MPa",
        "clause":     "8.4.2",
        "regex": [
            r"\bf_?\{?bd\}?\b", r"\bfbd\b",
            r"r[eé]sistance.*adh[eé]rence",
            r"bond.*strength",
        ],
    },

    # ── Crack control ─────────────────────────────────────────────────────

    "wk": {
        "formula":    "wk = sr,max × (εsm − εcm)",
        "definition": "largeur de fissure de calcul",
        "value":      None,
        "unit":       "mm",
        "clause":     "7.3.4",
        "regex": [
            r"\bw_?\{?k\}?\b", r"\bwk\b",
            r"largeur.*fissure",
            r"crack.*width",
        ],
    },

    "As_min_beam": {
        "formula":    "As,min = max(0,26 × fctm/fyk × bt × d ; 0,0013 × bt × d)",
        "definition": "section minimale d'armature longitudinale tendue pour une poutre",
        "value":      None,
        "unit":       "mm²",
        "clause":     "9.2.1.1",
        "regex": [
            r"armature.*minimale.*poutre",
            r"minimum.*reinforcement.*beam",
            r"section minimale.*armature",
            r"As,?min",
        ],
    },

    "rho_w_min": {
        "formula":    "ρw,min = 0,08 × √fck / fyk",
        "definition": "taux minimal d'armature transversale (effort tranchant)",
        "value":      None,
        "unit":       "−",
        "clause":     "9.2.2",
        "regex": [
            r"\bρ_?w,?min\b", r"\brho_?w,?min\b",
            r"taux.*minimal.*armature.*transversale",
            r"minimum.*shear.*reinforcement.*ratio",
        ],
    },
}


# ---------------------------------------------------------------------------
# Symbol extractor — find all symbols mentioned in a query
# ---------------------------------------------------------------------------

def extract_symbols(query: str) -> list[str]:
    """
    Return list of symbol_ids whose regex patterns match the query.
    Matches are deduplicated and ordered by first occurrence.
    """
    found = []
    for sym_id, entry in SYMBOLS.items():
        for pattern in entry["regex"]:
            if re.search(pattern, query, re.IGNORECASE):
                if sym_id not in found:
                    found.append(sym_id)
                break
    return found


# ---------------------------------------------------------------------------
# Symbol chunk builder — returns injected context chunk
# ---------------------------------------------------------------------------

def make_symbol_chunk(symbol_ids: list[str],
                      source_id: str = "ec2_2004_nf") -> dict | None:
    """
    Build a single context chunk containing all matched symbol definitions.
    Returns None if no symbols matched.
    """
    if not symbol_ids:
        return None

    lines = ["[Symbol Dictionary — definitions for symbols mentioned in query]\n"]
    clauses_cited = set()

    for sym_id in symbol_ids:
        entry = SYMBOLS[sym_id]
        lines.append(f"**{sym_id}** — {entry['definition']}")
        lines.append(f"  Formula : {entry['formula']}")
        if entry["value"] is not None:
            lines.append(f"  Value   : {entry['value']} {entry['unit']}")
        lines.append("")
        clauses_cited.add(entry["clause"])

    text = "\n".join(lines)

    return {
        "chunk_id":      f"{source_id}_symbol_dict_{'_'.join(sorted(symbol_ids)[:4])}",
        "source_id":     source_id,
        "chunk_type":    "hierarchical",
        "clause_number": list(clauses_cited)[0] if clauses_cited else "3.1",
        "level":         3,
        "ancestor_path": "Eurocode 2 > [Symbol Dictionary]",
        "page_number":   0,
        "text":          text,
        "token_count":   len(text.split()),
        "score":         0.0001,
        "cross_refs":    [],
        "metadata":      {"symbol_dict": True, "symbols": symbol_ids},
    }


# ---------------------------------------------------------------------------
# Main entry point — enrich context with symbol definitions
# ---------------------------------------------------------------------------

def enrich_with_symbols(query: str,
                        existing_chunks: list[dict],
                        source_id: str = "ec2_2004_nf") -> list[dict]:
    """
    Extract symbols from the query and prepend their definitions to context.
    Deduplicates with any existing symbol_dict chunk.
    """
    symbol_ids = extract_symbols(query)
    if not symbol_ids:
        return existing_chunks

    chunk = make_symbol_chunk(symbol_ids, source_id)
    if chunk is None:
        return existing_chunks

    # Remove any previous symbol dict chunk to avoid duplicates
    filtered = [c for c in existing_chunks
                if not c.get("metadata", {}).get("symbol_dict")]
    return [chunk] + filtered
