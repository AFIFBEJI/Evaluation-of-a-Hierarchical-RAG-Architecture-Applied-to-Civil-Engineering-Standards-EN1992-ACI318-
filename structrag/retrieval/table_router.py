"""
structrag/retrieval/table_router.py
-------------------------------------
Structured table lookup for direct numeric queries.

Instead of relying on embedding similarity to find a number in a markdown
table, this module intercepts table-lookup-shaped queries and answers them
directly from pre-built dicts.

Tables covered
--------------
Table 2.1N  — Partial safety factors (γc, γs, γp)
Table 3.1   — Concrete strength/deformation (fck, Ecm, εcu2, etc.)
Table 4.3N  — Structural classification (concrete class by exposure class)
Table 4.4N  — Minimum cover cmin,dur (mm) by structural class × exposure class

Usage
-----
    from structrag.retrieval.table_router import route_table_query
    result = route_table_query("cover for XC2 S4")
    if result:
        # result is a dict-chunk, inject directly into context
        chunks = [result] + other_chunks
"""

from __future__ import annotations
import re
from typing import Optional

# ---------------------------------------------------------------------------
# Table 2.1N — Partial safety factors (clause 2.4.2.4)
# ---------------------------------------------------------------------------
TABLE_2_1N = {
    "persistent":   {"gamma_c": 1.5,  "gamma_s": 1.15, "gamma_p": 1.15},
    "transient":    {"gamma_c": 1.5,  "gamma_s": 1.15, "gamma_p": 1.15},
    "accidental":   {"gamma_c": 1.2,  "gamma_s": 1.0,  "gamma_p": 1.0},
}

TABLE_2_1N_TEXT = (
    "Tableau 2.1N : Coefficients partiels relatifs aux matériaux (clause 2.4.2.4)\n\n"
    "| Situation de projet | γc (béton) | γs (acier BA) | γp (acier PC) |\n"
    "|---------------------|-----------|--------------|---------------|\n"
    "| Durable / Transitoire | **1,5** | **1,15** | **1,15** |\n"
    "| Accidentelle          | **1,2** | **1,0**  | **1,0**  |\n\n"
    "γc = coefficient partiel pour le béton\n"
    "γs = coefficient partiel pour l'acier de béton armé\n"
    "γp = coefficient partiel pour l'acier de précontrainte\n"
    "Ces valeurs s'appliquent aux vérifications aux états-limites ultimes.\n"
    "Pour le feu : voir EN 1992-1-2."
)

# ---------------------------------------------------------------------------
# Table 3.1 — Concrete properties (clause 3.1.2)
# ---------------------------------------------------------------------------
# Key = concrete class string (e.g. "C30/37")
# Values: fck(MPa), fck_cube(MPa), fcm(MPa), fctm(MPa), Ecm(GPa), ecu2(permille)
TABLE_3_1 = {
    "C12/15": {"fck": 12,  "fck_cube": 15,  "fcm": 20,  "fctm": 1.6, "Ecm": 27, "ecu2": 3.5},
    "C16/20": {"fck": 16,  "fck_cube": 20,  "fcm": 24,  "fctm": 1.9, "Ecm": 29, "ecu2": 3.5},
    "C20/25": {"fck": 20,  "fck_cube": 25,  "fcm": 28,  "fctm": 2.2, "Ecm": 30, "ecu2": 3.5},
    "C25/30": {"fck": 25,  "fck_cube": 30,  "fcm": 33,  "fctm": 2.6, "Ecm": 31, "ecu2": 3.5},
    "C30/37": {"fck": 30,  "fck_cube": 37,  "fcm": 38,  "fctm": 2.9, "Ecm": 33, "ecu2": 3.5},
    "C35/45": {"fck": 35,  "fck_cube": 45,  "fcm": 43,  "fctm": 3.2, "Ecm": 34, "ecu2": 3.5},
    "C40/50": {"fck": 40,  "fck_cube": 50,  "fcm": 48,  "fctm": 3.5, "Ecm": 35, "ecu2": 3.5},
    "C45/55": {"fck": 45,  "fck_cube": 55,  "fcm": 53,  "fctm": 3.8, "Ecm": 36, "ecu2": 3.5},
    "C50/60": {"fck": 50,  "fck_cube": 60,  "fcm": 58,  "fctm": 4.1, "Ecm": 37, "ecu2": 3.5},
    "C55/67": {"fck": 55,  "fck_cube": 67,  "fcm": 63,  "fctm": 4.2, "Ecm": 38, "ecu2": 3.1},
    "C60/75": {"fck": 60,  "fck_cube": 75,  "fcm": 68,  "fctm": 4.4, "Ecm": 39, "ecu2": 2.9},
    "C70/85": {"fck": 70,  "fck_cube": 85,  "fcm": 78,  "fctm": 4.6, "Ecm": 41, "ecu2": 2.7},
    "C80/95": {"fck": 80,  "fck_cube": 95,  "fcm": 88,  "fctm": 4.8, "Ecm": 42, "ecu2": 2.6},
    "C90/105":{"fck": 90,  "fck_cube":105,  "fcm": 98,  "fctm": 5.0, "Ecm": 44, "ecu2": 2.6},
}

# ---------------------------------------------------------------------------
# Table 4.3N — Minimum concrete class by exposure class (clause 4.4.1.2)
# ---------------------------------------------------------------------------
TABLE_4_3N = {
    "X0":   "C12/15",
    "XC1":  "C20/25",
    "XC2":  "C25/30",  # minimum; for structural class adjustments see Table 4.3N
    "XC3":  "C30/37",
    "XC4":  "C30/37",
    "XD1":  "C30/37",
    "XD2":  "C35/45",
    "XD3":  "C35/45",  # with possible minoration; Table 4.3N recommends >= C45/55 baseline
    "XS1":  "C30/37",
    "XS2":  "C35/45",
    "XS3":  "C35/45",
    "XF1":  "C30/37",
    "XF2":  "C25/30",
    "XF3":  "C30/37",
    "XF4":  "C30/37",
    "XA1":  "C30/37",
    "XA2":  "C35/45",
    "XA3":  "C35/45",
}

TABLE_4_3N_TEXT = (
    "Tableau 4.3N : Classes de résistance indicatives pour les classes d'exposition "
    "(clause 4.4.1.2)\n\n"
    "| Classe d'exposition | Classe de résistance minimale | Notes |\n"
    "|---------------------|-------------------------------|-------|\n"
    "| X0       | C12/15  | Pas de risque de corrosion |\n"
    "| XC1      | C20/25  | Carbonatation, sec ou humide en permanence |\n"
    "| XC2/XC3  | C30/37  | Carbonatation, humide/modéré → **C35/45** recommandé |\n"
    "| XC4      | C30/37  | Carbonatation cyclique → **C35/45** recommandé |\n"
    "| XD1      | C30/37  | Chlorures, humidité modérée |\n"
    "| XD2/XS2  | C35/45  | Chlorures, humide rarement sec |\n"
    "| XD3/XS3  | **C45/55** | Chlorures, cyclique — classe la plus sévère |\n"
    "| XF1–XF4  | C30/37–C35/45 | Gel-dégel |\n\n"
    "Note: Ces valeurs sont des minima. Le Tableau 4.3N complet inclut des "
    "majorations/minorations selon la durée d'utilisation, l'élément (dalle) "
    "et le contrôle qualité. Valeurs per EC2 clause 4.4.1.2 et Annexe E."
)

# ---------------------------------------------------------------------------
# Table 4.4N — Minimum cover cmin,dur (mm) by structural class × exposure class
# ---------------------------------------------------------------------------
# [structural_class][exposure_class] = cmin,dur in mm
TABLE_4_4N = {
    "S1": {"X0": 10, "XC1": 10, "XC2": 10, "XC3": 10, "XC4": 15, "XD1": 20, "XD2": 25, "XD3": 30,
           "XS1": 20, "XS2": 25, "XS3": 30},
    "S2": {"X0": 10, "XC1": 10, "XC2": 15, "XC3": 15, "XC4": 20, "XD1": 25, "XD2": 30, "XD3": 35,
           "XS1": 25, "XS2": 30, "XS3": 35},
    "S3": {"X0": 10, "XC1": 10, "XC2": 20, "XC3": 20, "XC4": 25, "XD1": 30, "XD2": 35, "XD3": 40,
           "XS1": 30, "XS2": 35, "XS3": 40},
    "S4": {"X0": 10, "XC1": 15, "XC2": 25, "XC3": 25, "XC4": 30, "XD1": 35, "XD2": 40, "XD3": 45,
           "XS1": 35, "XS2": 40, "XS3": 45},
    "S5": {"X0": 15, "XC1": 20, "XC2": 30, "XC3": 30, "XC4": 35, "XD1": 40, "XD2": 45, "XD3": 50,
           "XS1": 40, "XS2": 45, "XS3": 50},
    "S6": {"X0": 20, "XC1": 25, "XC2": 35, "XC3": 35, "XC4": 40, "XD1": 45, "XD2": 50, "XD3": 55,
           "XS1": 45, "XS2": 50, "XS3": 55},
}

TABLE_4_4N_TEXT_TEMPLATE = (
    "Tableau 4.4N : Enrobage minimal c_min,dur (mm) — clause 4.4.1.2\n\n"
    "| Classe structurale | X0 | XC1 | XC2/XC3 | XC4 | XD1/XS1 | XD2/XS2 | XD3/XS3 |\n"
    "|--------------------|----|----|---------|-----|---------|---------|----------|\n"
    "| S1 | 10 | 10 | 10  | 15 | 20 | 25 | 30 |\n"
    "| S2 | 10 | 10 | 15  | 20 | 25 | 30 | 35 |\n"
    "| S3 | 10 | 10 | 20  | 25 | 30 | 35 | 40 |\n"
    "| **S4** | 10 | **15** | **25** | **30** | **35** | **40** | **45** |\n"
    "| S5 | 15 | 20 | 30  | 35 | 40 | 45 | 50 |\n"
    "| S6 | 20 | 25 | 35  | 40 | 45 | 50 | 55 |\n\n"
    "Classe structurale recommandée pour 50 ans = S4.\n"
    "XC2 et XC3 ont les mêmes valeurs de cmin,dur."
)

# ---------------------------------------------------------------------------
# Table 4.1 — All exposure classes (clause 4.2)
# ---------------------------------------------------------------------------
TABLE_4_1_TEXT = (
    "Tableau 4.1 : Classes d'exposition environnementale — clause 4.2\n\n"
    "**Classe X0 — Aucun risque de corrosion ou d'attaque**\n"
    "| Classe | Description |\n"
    "|--------|-------------|\n"
    "| X0 | Béton sans armatures ni pièces métalliques noyées — toute exposition sauf gel/dégel, abrasion, attaque chimique |\n\n"
    "**Classes XC — Corrosion induite par carbonatation**\n"
    "| Classe | Description de l'environnement |\n"
    "|--------|--------------------------------|\n"
    "| XC1 | Sec ou humide en permanence |\n"
    "| XC2 | Humide, rarement sec |\n"
    "| XC3 | Humidité modérée |\n"
    "| XC4 | Alternativement humide et sec |\n\n"
    "**Classes XD — Corrosion induite par les chlorures d'origine non marine**\n"
    "| Classe | Description de l'environnement |\n"
    "|--------|--------------------------------|\n"
    "| XD1 | Humidité modérée |\n"
    "| XD2 | Humide, rarement sec |\n"
    "| XD3 | Alternativement humide et sec |\n\n"
    "**Classes XS — Corrosion induite par les chlorures présents dans l'eau de mer**\n"
    "| Classe | Description de l'environnement |\n"
    "|--------|--------------------------------|\n"
    "| XS1 | Exposé à l'air véhiculant du sel marin, mais pas en contact direct avec l'eau de mer |\n"
    "| XS2 | Submergé en permanence |\n"
    "| XS3 | Zones de marnage, zones soumises à des projections ou à des embruns |\n\n"
    "**Classes XF — Attaque gel/dégel avec ou sans agent de déverglaçage**\n"
    "| Classe | Description de l'environnement |\n"
    "|--------|--------------------------------|\n"
    "| XF1 | Saturation modérée en eau, sans agent de déverglaçage |\n"
    "| XF2 | Saturation modérée en eau, avec agent de déverglaçage |\n"
    "| XF3 | Saturation élevée en eau, sans agent de déverglaçage |\n"
    "| XF4 | Saturation élevée en eau, avec agent de déverglaçage |\n\n"
    "**Classes XA — Attaque chimique**\n"
    "| Classe | Description de l'environnement |\n"
    "|--------|--------------------------------|\n"
    "| XA1 | Environnement à faible agressivité chimique (sols et eaux souterraines) |\n"
    "| XA2 | Environnement à agressivité chimique modérée |\n"
    "| XA3 | Environnement à forte agressivité chimique |\n\n"
    "Total : 18 classes d'exposition réparties en 6 catégories (X0, XC, XD, XS, XF, XA).\n"
    "Source : Eurocode 2, Tableau 4.1, clause 4.2."
)
TABLE_7_1N = {
    ("X0", "XC1"):            {"quasi_permanent": 0.3, "frequent": 0.2},
    ("XC2", "XC3", "XC4"):   {"quasi_permanent": 0.3, "frequent": 0.2},
    ("XD1", "XD2", "XS1", "XS2", "XS3"): {"quasi_permanent": 0.3, "frequent": 0.2},
}

TABLE_7_1N_TEXT = (
    "Tableau 7.1N : Largeur de fissure maximale recommandée wmax (mm) — clause 7.3.1\n\n"
    "| Classe d'exposition | Béton armé — combinaison quasi-permanente | "
    "Béton armé — combinaison fréquente |\n"
    "|---------------------|------------------------------------------|-----------------------------------|\n"
    "| X0, XC1             | 0,3 mm | 0,2 mm |\n"
    "| XC2, XC3, XC4       | 0,3 mm | 0,2 mm |\n"
    "| XD1, XD2, XS1–XS3   | 0,3 mm | Décompression recommandée |\n\n"
    "Note: Pour les éléments précontraints avec adhérence, vérifier aussi les critères de "
    "décompression. Ces valeurs sont des recommandations (NDP). Per EC2 clause 7.3.1."
)

# ---------------------------------------------------------------------------
# Table 8.2 — Minimum mandrel diameter for bar bending (clause 8.3)
# ---------------------------------------------------------------------------
TABLE_8_3_TEXT = (
    "Tableau 8.2 — Diamètre minimal du mandrin (diamètre intérieur) de pliage — clause 8.3\n\n"
    "| Diamètre de barre φ | Diamètre minimal du mandrin |\n"
    "|---------------------|-----------------------------|\n"
    "| φ ≤ 16 mm           | **4φ** |\n"
    "| φ > 16 mm           | **7φ** |\n\n"
    "Per EC2 clause 8.3. Ces valeurs s'appliquent aux barres HA courbes (crochets, cadres, "
    "épingles). Le diamètre intérieur du mandrin doit être suffisant pour éviter la rupture "
    "ou la fissuration de la barre ou du béton."
)

# ---------------------------------------------------------------------------
# Table 8.7.3 — Minimum lap length (clause 8.7.3)
# ---------------------------------------------------------------------------
TABLE_8_7_3_TEXT = (
    "Clause 8.7.3 — Longueur minimale de recouvrement\n\n"
    "| Condition | Longueur minimale l0,min |\n"
    "|-----------|-------------------------|\n"
    "| Armatures tendues | max(0,3 × α6 × lb,rqd ; 15φ ; 200 mm) |\n"
    "| Armatures comprimées | max(0,3 × lb,rqd ; 15φ ; 200 mm) |\n\n"
    "où α6 dépend du % de barres recouvrées au même endroit:\n"
    "  α6 = 1,0 pour ≤ 25% ; 1,4 pour ≥ 50% ; 1,5 pour 100% (interpolation)\n"
    "Per EC2 clause 8.7.3."
)

# ---------------------------------------------------------------------------
# Table 9.1N — Minimum reinforcement ratio for walls and columns
# ---------------------------------------------------------------------------
TABLE_9_COLUMNS_TEXT = (
    "Clause 9.5 — Armatures des poteaux\n\n"
    "Section minimale armatures longitudinales (Eq. 9.12N):\n"
    "  As,min = max(0,10 × NEd / fyd ; 0,002 × Ac)\n\n"
    "Section maximale:\n"
    "  As,max = 0,04 × Ac  (hors zones de recouvrement)\n\n"
    "Nombre minimal de barres: 4 barres (section rectangulaire), 6 barres (section circulaire)\n\n"
    "Armatures transversales (clause 9.5.3):\n"
    "  Diamètre minimal : max(6 mm ; φlong/4)\n"
    "  Espacement maximal : min(20 × φlong min ; b ou h min ; 400 mm)\n"
    "Per EC2 clause 9.5."
)

# ---------------------------------------------------------------------------
# Query pattern detection
# ---------------------------------------------------------------------------

_EXPOSURE_RE = re.compile(
    r'\b(X[0]|X[CDSFA][1-4])\b',
    re.IGNORECASE
)
_STRUCTURAL_RE = re.compile(r'\b(S[1-6])\b', re.IGNORECASE)
_CONCRETE_RE   = re.compile(r'\b(C\d{2}/\d{2})\b', re.IGNORECASE)
_SAFETY_FACTOR_RE = re.compile(
    r'\b(gamma[_\s]?[csp]|γ[csp]|coefficient partiel|partial safety|facteur partiel)\b',
    re.IGNORECASE
)
_STRAIN_RE = re.compile(
    r'\b(ecu2|εcu2|epsilon|deformation ultime|compressive strain|parabole.rectangle)\b',
    re.IGNORECASE
)
_STRENGTH_RE = re.compile(
    r'\b(fck|fcm|fctm|Ecm|resistance.*compression|compressive strength)\b',
    re.IGNORECASE
)


def _make_chunk(text: str, clause: str, title: str, source_id: str = "ec2_2004_nf") -> dict:
    """Build a fake chunk dict from table text for injection into context."""
    return {
        "chunk_id":      f"{source_id}_table_router_{clause.replace('.', '_')}",
        "source_id":     source_id,
        "chunk_type":    "hierarchical",
        "clause_number": clause,
        "level":         3,
        "ancestor_path": f"Eurocode 2 > [Table Router] > Clause {clause}",
        "page_number":   0,
        "text":          text,
        "token_count":   len(text.split()),
        "score":         0.001,   # low score so it doesn't override real results
        "cross_refs":    [],
        "metadata":      {"table_router": True},
    }


def route_table_query(
    query: str,
    source_id: str = "ec2_2004_nf",
) -> Optional[dict]:
    """
    Detect if the query is a table lookup and return a pre-built context chunk.
    Returns None if no table match is found.

    Handles:
    - Cover lookup: exposure class + structural class → cmin,dur from Table 4.4N
    - Partial safety factor → Table 2.1N
    - Concrete class properties → Table 3.1
    - Minimum concrete class for exposure → Table 4.3N
    """
    # ── Table 4.1: all exposure classes list ─────────────────────────────
    if re.search(
        r'\b(classes?.d.exposition|exposure.class|toutes?.les?.classes?|'
        r'liste?.des?.classes?|quelles?.sont?.les?.classes?|'
        r'environnementale|XC.*XD|XD.*XS|XF.*XA)\b',
        query, re.IGNORECASE
    ):
        text = f"[Table Router — Tableau 4.1, clause 4.2]\n\n{TABLE_4_1_TEXT}"
        return _make_chunk(text, "4.2", "Table 4.1 — all exposure classes", source_id)

    # ── Table 4.4N: cover lookup ──────────────────────────────────────────
    exp_matches  = _EXPOSURE_RE.findall(query)
    str_matches  = _STRUCTURAL_RE.findall(query)

    if exp_matches and str_matches:
        # Specific cell lookup: exposure × structural class
        exp_cls = exp_matches[0].upper()
        str_cls = str_matches[0].upper()

        # Normalise: XC2/XC3 share same row
        lookup_exp = "XC2" if exp_cls == "XC3" else exp_cls

        cover_val = TABLE_4_4N.get(str_cls, {}).get(lookup_exp)

        if cover_val is not None:
            text = (
                f"[Table Router — Tableau 4.4N, clause 4.4.1.2]\n\n"
                f"Pour la classe d'exposition **{exp_cls}** et la classe structurale **{str_cls}** :\n\n"
                f"**c_min,dur = {cover_val} mm**\n\n"
                f"{TABLE_4_4N_TEXT_TEMPLATE}"
            )
            return _make_chunk(text, "4.4.1.2", f"Table 4.4N — cover {exp_cls} {str_cls}", source_id)

    # ── Table 4.3N: concrete class for exposure ───────────────────────────
    if exp_matches and not str_matches:
        # Query about minimum concrete class for an exposure class
        if re.search(r'classe.*résistance|concrete.*class|béton.*classe|strength.*class|C\d{2}',
                     query, re.IGNORECASE):
            exp_cls = exp_matches[0].upper()
            min_class = TABLE_4_3N.get(exp_cls)
            if min_class:
                text = (
                    f"[Table Router — Tableau 4.3N, clause 4.4.1.2]\n\n"
                    f"Pour la classe d'exposition **{exp_cls}**, la classe de résistance "
                    f"minimale recommandée est **≥ {min_class}**.\n\n"
                    f"{TABLE_4_3N_TEXT}"
                )
                return _make_chunk(text, "4.4.1.2", f"Table 4.3N — class {exp_cls}", source_id)

    # ── Table 2.1N: partial safety factors ───────────────────────────────
    if _SAFETY_FACTOR_RE.search(query):
        text = (
            f"[Table Router — Tableau 2.1N, clause 2.4.2.4]\n\n"
            f"{TABLE_2_1N_TEXT}"
        )
        return _make_chunk(text, "2.4.2.4", "Table 2.1N — safety factors", source_id)

    # ── Table 3.1: concrete class properties ─────────────────────────────
    concrete_matches = _CONCRETE_RE.findall(query)
    if concrete_matches and (_STRAIN_RE.search(query) or _STRENGTH_RE.search(query)):
        cls = concrete_matches[0].upper()
        props = TABLE_3_1.get(cls)
        if props:
            text = (
                f"[Table Router — Tableau 3.1, clause 3.1.2]\n\n"
                f"Propriétés du béton **{cls}** (Tableau 3.1) :\n\n"
                f"| Propriété | Valeur |\n"
                f"|-----------|--------|\n"
                f"| fck (résistance caract. en compression) | {props['fck']} MPa |\n"
                f"| fcm (résistance moyenne) | {props['fcm']} MPa |\n"
                f"| fctm (résistance en traction) | {props['fctm']} MPa |\n"
                f"| Ecm (module d'élasticité sécant) | {props['Ecm']} GPa |\n"
                f"| εcu2 (déformation ultime, parabole-rectangle) | **{props['ecu2']} ‰** |\n\n"
                f"Toutes les valeurs sont per EC2 clause 3.1.2, Tableau 3.1."
            )
            return _make_chunk(text, "3.1.2", f"Table 3.1 — {cls}", source_id)

    # ── Table 3.1: strain query without specific class ────────────────────
    if _STRAIN_RE.search(query):
        text = (
            f"[Table Router — Tableau 3.1, clause 3.1.2]\n\n"
            f"Déformation ultime de compression εcu2 (diagramme parabole-rectangle) :\n\n"
            f"| Classe de béton | εcu2 (‰) |\n"
            f"|-----------------|----------|\n"
            f"| C12/15 à C50/60 | **3,5** |\n"
            f"| C55/67          | 3,1 |\n"
            f"| C60/75          | 2,9 |\n"
            f"| C70/85          | 2,7 |\n"
            f"| C80/95          | 2,6 |\n"
            f"| C90/105         | 2,6 |\n\n"
            f"Per EC2 clause 3.1.2 et 3.1.7, Tableau 3.1. "
            f"La valeur εcu2 = 3,5 ‰ s'applique aux bétons courants jusqu'à C50/60."
        )
        return _make_chunk(text, "3.1.2", "Table 3.1 — ecu2 strain", source_id)

    # ── Table 7.1N: crack width ───────────────────────────────────────────
    if re.search(r'\b(fissure|crack|wmax|w_max|largeur.*fissure|crack.*width)\b',
                 query, re.IGNORECASE):
        text = f"[Table Router — Tableau 7.1N, clause 7.3.1]\n\n{TABLE_7_1N_TEXT}"
        return _make_chunk(text, "7.3.1", "Table 7.1N — crack width", source_id)

    # ── Table 8.3: mandrel diameter ───────────────────────────────────────
    if re.search(r'\b(mandrin|mandrel|pliage|bending|cintrage)\b', query, re.IGNORECASE):
        text = f"[Table Router — Tableau 8.2, clause 8.3]\n\n{TABLE_8_3_TEXT}"
        return _make_chunk(text, "8.3", "Table 8.2 — mandrel diameter", source_id)

    # ── Clause 8.7.3: lap length ──────────────────────────────────────────
    if re.search(r'\b(recouvrement|lap.*length|longueur.*recouvrement)\b',
                 query, re.IGNORECASE):
        text = f"[Table Router — Clause 8.7.3]\n\n{TABLE_8_7_3_TEXT}"
        return _make_chunk(text, "8.7.3", "Clause 8.7.3 — lap length", source_id)

    # ── Clause 9.5: column reinforcement ─────────────────────────────────
    if re.search(r'\b(poteau|column|poteaux|armature.*poteau|section.*poteau)\b',
                 query, re.IGNORECASE):
        text = f"[Table Router — Clause 9.5]\n\n{TABLE_9_COLUMNS_TEXT}"
        return _make_chunk(text, "9.5", "Clause 9.5 — columns", source_id)

    return None


# ---------------------------------------------------------------------------
# Integration helper — inject table chunk at top of context if available
# ---------------------------------------------------------------------------

def enrich_with_table_router(
    query: str,
    existing_chunks: list[dict],
    source_id: str = "ec2_2004_nf",
) -> list[dict]:
    """
    Check if the query can be answered by a table lookup.
    If so, prepend the table chunk to the existing context.
    The table chunk provides the exact numeric value; existing chunks
    provide surrounding context.
    """
    table_chunk = route_table_query(query, source_id)
    if table_chunk is None:
        return existing_chunks

    # Prepend table chunk, remove any existing table-router duplicates
    filtered = [c for c in existing_chunks if not c.get("metadata", {}).get("table_router")]
    return [table_chunk] + filtered
