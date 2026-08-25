"""
structrag/retrieval/formula_library.py
----------------------------------------
Complete formula library for NF EN 1992-1-1:2004 (Eurocode 2).

Covers all 167 image blocks identified in the parsed nodes, grouped by clause.
Formulas are stored as structured text with:
- Mathematical notation in Unicode/ASCII-readable form
- Variable definitions
- Applicability conditions
- Clause and equation number references

These are used by the table_router.py to inject formula context when
a query mentions a clause that has image-rendered formulas.

All formulas are from the public EN 1992-1-1 standard (European standard,
publicly accessible via CEN member bodies).
"""

from __future__ import annotations
from typing import Optional

# ---------------------------------------------------------------------------
# Formula entries: {clause_number: formula_text}
# ---------------------------------------------------------------------------
EC2_FORMULAS: dict[str, str] = {

    # ── Section 4 — Cover / Durability ───────────────────────────────────

    "4.4.1.1": """
Clause 4.4.1 — Enrobage nominal et enrobage minimal

Enrobage nominal (Eq. 4.1):
  cnom = cmin + Δcdev

où:
  cnom  = enrobage nominal (valeur à spécifier sur les plans)
  cmin  = enrobage minimal (voir Eq. 4.2)
  Δcdev = tolérance d'exécution (valeur recommandée = 10 mm)
          peut être réduite à 5 mm avec système qualité, ou 0 mm pour préfabriqués

Enrobage minimal (Eq. 4.2):
  cmin = max(cmin,b ; cmin,dur + Δcdur,γ - Δcdur,st - Δcdur,add ; 10 mm)

où:
  cmin,b   = enrobage minimal pour l'adhérence (= diamètre barre φ, ou φéquiv pour câbles)
  cmin,dur = enrobage minimal pour les conditions environnementales (Tableau 4.4N)
  Δcdur,γ  = marge de sécurité supplémentaire (recommandé = 0 mm)
  Δcdur,st = réduction si acier inoxydable (recommandé = 0 mm)
  Δcdur,add= réduction si protection supplémentaire (recommandé = 0 mm)

Exemple pour XC3, S4, dalle (50 ans):
  cmin,dur = 25 mm (Tableau 4.4N, XC2/XC3 × S4)
  Réduction dalle → S3 : cmin,dur = 20 mm
  cnom = 20 + 10 = 30 mm
""",

    # ── Section 3 — Materials ─────────────────────────────────────────────

    "3.1.2": """
Clause 3.1.2 — Résistance en compression et en traction du béton

Résistance moyenne en compression:
  fcm = fck + 8  (MPa)

Résistance moyenne en traction directe:
  fctm = 0,30 × fck^(2/3)  pour fck ≤ 50 MPa
  fctm = 2,12 × ln(1 + fcm/10)  pour fck > 50 MPa

Résistances caractéristiques en traction:
  fctk,0,05 = 0,70 × fctm  (fractile 5 % — valeur inférieure)
  fctk,0,95 = 1,30 × fctm  (fractile 95 % — valeur supérieure)

Module d'élasticité sécant moyen:
  Ecm = 22 × (fcm / 10)^0,3  (GPa)

Module d'élasticité à l'instant t:
  Ecm(t) = (fcm(t) / fcm)^0,3 × Ecm
  où fcm(t) = exp(s × (1 − (28/t)^0,5)) × fcm

Valeurs Tableau 3.1 (exemples):
  C25/30 : fck=25, fcm=33, fctm=2,6, Ecm=31
  C30/37 : fck=30, fcm=38, fctm=2,9, Ecm=33
  C35/45 : fck=35, fcm=43, fctm=3,2, Ecm=34
""",

    "3.1.6": """
Résistance de calcul en compression (Eq. 3.15):
  fcd = αcc × fck / γc

Résistance de calcul en traction (Eq. 3.16):
  fctd = αct × fctk,0,05 / γc

où:
  fck    = résistance caractéristique en compression du béton (MPa)
  fctk,0,05 = fractile 5% de la résistance en traction (MPa)
  γc     = coefficient partiel du béton (= 1,5 situation durable/transitoire)
  αcc    = coefficient effets à long terme compression (valeur recommandée = 1,0)
  αct    = coefficient effets à long terme traction (valeur recommandée = 1,0)
""",

    "3.1.6": """
Clause 3.1.6 — Résistance de calcul en compression et en traction (Eq. 3.17, 3.18)

Pour 0 ≤ εc ≤ εc2 (branche parabolique):
  σc = fcd × [1 - (1 - εc/εc2)^n]

Pour εc2 ≤ εc ≤ εcu2 (plateau rectangulaire):
  σc = fcd

Valeurs pour bétons ≤ C50/60 (Tableau 3.1):
  n     = 2,0 (exposant)
  εc2   = 2,0 ‰ (déformation à contrainte maximale)
  εcu2  = 3,5 ‰ (déformation ultime)
  εc3   = 1,75 ‰ (bilinéaire)
  εcu3  = 3,5 ‰ (bilinéaire ultime)

Diagramme rectangulaire simplifié (Fig. 3.5):
  Profondeur zone comprimée : λx  (λ = 0,8 pour fck ≤ 50 MPa)
  Contrainte uniforme       : η × fcd  (η = 1,0 pour fck ≤ 50 MPa)
""",

    "3.1.7": """
Clause 3.1.7 — Diagramme parabole-rectangle

Résistance en compression du béton confiné (Eq. 3.24):
  fck,c = fck × (1,000 + 5,0 × σ2/fck)  pour σ2 ≤ 0,05 fck
  fck,c = fck × (1,125 + 2,5 × σ2/fck)  pour σ2 > 0,05 fck

où σ2 = contrainte effective de confinement (petite valeur principale des contraintes)
Déformation ultime confinée (Eq. 3.27):
  εcu2,c = εcu2 + 0,1 × αω × σ2/fck
""",

    "3.2.7": """
Clause 3.2.7 — Armatures de béton armé : diagramme de calcul

Diagramme bilinéaire (Option A — plateau horizontal au-delà de fyd):
  fyd = fyk / γs
  εud = 0,9 × εuk  (déformation limite de calcul)
  Module : Es = 200 000 MPa

Option B (avec écrouissage) :
  Contrainte ultime kfyk, déformation εuk
  k = ft/fy  (rapport des résistances)
""",

    # ── Section 5 — Analyse structurale ──────────────────────────────────

    "5.3.2.2": """
Clause 5.3.2.2 — Portée efficace des poutres et dalles (Eq. 5.8)

  leff = ln + a1 + a2

où:
  ln  = portée entre nus d'appuis
  a1, a2 = min(h/2 ; 0,5 × largeur d'appui) à chaque extrémité
  h   = hauteur totale de la section

Pour appui sur mur (Eq. 5.9):
  a = min(h/2 ; 0,5 × t)
  t = épaisseur du mur d'appui
""",

    "5.5": """
Clause 5.5 — Redistribution des moments (Eq. 5.10a, 5.10b)

Condition de redistribution admissible:
  δ ≥ k1 + k2 × xu/d  (Eq. 5.10a)  pour fck ≤ 50 MPa
  δ ≥ k3 + k4 × xu/d  (Eq. 5.10b)  pour fck > 50 MPa

Valeurs recommandées:
  k1 = 0,44 ; k2 = 1,25 × (0,6 + 0,0014/εcu2)
  k3 = 0,54 ; k4 = 1,25 × (0,6 + 0,0014/εcu2)

où:
  δ   = rapport moment redistribué / moment élastique (δ ≤ 1)
  xu  = hauteur de la zone comprimée après redistribution
  d   = hauteur utile de la section
""",

    "5.8.3.2": """
Clause 5.8.3.2 — Élancement et longueur efficace des éléments isolés

Critère d'élancement simplifié (Eq. 5.13N):
  λ ≤ λlim = 20 × A × B × C / √n

où:
  n  = NEd / (Ac × fcd)  (effort normal réduit)
  A  = 1/(1 + 0,2 × φef)  (φef = fluage efficace)
  B  = √(1 + 2ω)  (ω = As × fyd / (Ac × fcd), rapport mécanique d'armatures)
  C  = 1,7 - rm  (rm = M01/M02, rapport des moments, |rm| ≤ 1)

Longueur efficace l0 (Tableau 5.1 pour cas courants):
  Console : l0 = 2l
  Bi-encastré : l0 = 0,5l
  1 rotule + 1 encastrement : l0 = 0,7l
""",

    # ── Section 6 — États-limites ultimes ─────────────────────────────────

    "6.1": """
Clause 6.1 — Flexion simple et flexion composée — Profondeur de l'axe neutre

CONDITIONS À VÉRIFIER À L'ELU EN FLEXION (hypothèses obligatoires):
  1. Sections planes restent planes (hypothèse de Bernoulli)
  2. Résistance en traction du béton négligée
  3. Diagramme rectangulaire simplifié : contrainte uniforme η × fcd sur hauteur λx
  4. Déformation ultime du béton limitée : εcu2 = 3,5 ‰ pour fck ≤ 50 MPa
  5. Déformation limite de l'acier : εud = 0,9 × εuk (ou εud = 0,01)

PROFONDEUR DE L'AXE NEUTRE x (section rectangulaire, flexion simple, NEd = 0):

  Équilibre des forces (béton comprimé = acier tendu):
    η × fcd × b × λx = As × fyd

  Profondeur de l'axe neutre:
    x = (As × fyd) / (η × fcd × b × λ)

  où:
    x    = profondeur de l'axe neutre (mm)
    As   = section d'armature tendue (mm²)
    fyd  = résistance de calcul de l'acier = fyk / 1,15  (MPa)
    fcd  = résistance de calcul du béton = αcc × fck / 1,5  (MPa)
    b    = largeur de la section (mm)
    η    = 1,0 pour fck ≤ 50 MPa
    λ    = 0,8 pour fck ≤ 50 MPa

CONDITION DE DUCTILITÉ (éviter rupture fragile par écrasement du béton):
    x/d ≤ 0,45  pour fck ≤ 50 MPa
    x/d ≤ 0,35  pour fck > 50 MPa

BRAS DE LEVIER INTERNE z:
    z = d - λx/2  avec  z ≤ 0,95d

MOMENT RÉSISTANT:
    MRd = As × fyd × z = η × fcd × b × λx × (d - λx/2)

VÉRIFICATION À EFFECTUER:
    MEd ≤ MRd  (moment de calcul ≤ moment résistant)
""",

    "6.2.2": """
Clause 6.2.2 — Résistance au cisaillement sans armatures (Eq. 6.2a, 6.2b)

VRd,c = [CRd,c × k × (100 × ρl × fck)^(1/3) + k1 × σcp] × bw × d  (Eq. 6.2a)

Valeur minimale (Eq. 6.2b):
  VRd,c ≥ (vmin + k1 × σcp) × bw × d

où:
  CRd,c = 0,18/γc = 0,12  (valeur recommandée)
  k     = 1 + √(200/d) ≤ 2,0  (d en mm)
  ρl    = Asl / (bw × d) ≤ 0,02  (taux d'armatures longitudinales)
  Asl   = section armatures tendues ancrées au-delà de la section considérée
  fck   = résistance caractéristique (MPa)
  σcp   = NEd / Ac ≤ 0,2 × fcd  (contrainte de compression)
  k1    = 0,15  (valeur recommandée)
  bw    = largeur minimale de l'âme (mm)
  d     = hauteur utile (mm)
  vmin  = 0,035 × k^(3/2) × fck^(1/2)  (valeur recommandée)
""",

    "6.2.3": """
Clause 6.2.3 — Résistance au cisaillement avec armatures (Eq. 6.6, 6.8, 6.9)

Résistance des armatures d'effort tranchant (Eq. 6.8):
  VRd,s = (Asw/s) × z × fywd × (cot θ + cot α) × sin α

Résistance de la bielle comprimée (Eq. 6.9):
  VRd,max = αcw × bw × z × ν1 × fcd / (cot θ + tan θ)

Résistance de calcul minimale (Eq. 6.6):
  VEd ≤ VRd = min(VRd,s, VRd,max)

où:
  Asw   = section des armatures d'effort tranchant
  s     = espacement des armatures transversales
  z     = bras de levier (≈ 0,9d pour flexion simple)
  fywd  = résistance de calcul des armatures transversales
  θ     = angle d'inclinaison de la bielle (1 ≤ cot θ ≤ 2,5 recommandé)
  α     = angle armatures transversales / axe de l'élément (= 90° pour étriers verticaux)
  αcw   = coefficient état de contrainte bielle (= 1,0 pour béton non précontraint)
  ν1    = coefficient réducteur de résistance = 0,6 × (1 - fck/250)
""",

    "6.2.4": """
Clause 6.2.4 — Cisaillement entre âme et membrures (Eq. 6.21, 6.22)

Effort de cisaillement longitudinal par unité de longueur:
  vEd = ΔFd / (hf × Δx)

Vérification compression bielle:
  vEd ≤ ν × fcd × sin θf × cos θf

Armatures transversales de couture:
  Asf / sf ≥ vEd × hf / (fyd × cot θf)

où:
  hf  = épaisseur de la table
  θf  = angle de la bielle (26,5° ≤ θf ≤ 45°)
  ν   = 0,6 × (1 - fck/250)
""",

    "6.4.3": """
Clause 6.4.3 — Résistance au poinçonnement sans armatures (Eq. 6.38, 6.39)

Contrainte de poinçonnement de calcul:
  vEd = β × VEd / (u1 × d)

Résistance sans armatures (Eq. 6.38):
  vRd,c = CRd,c × k × (100 × ρl × fck)^(1/3) + k2 × σcp ≥ vmin + k2 × σcp

où:
  β   = facteur tenant compte de l'excentricité (= 1,0 sans excentricité)
  u1  = périmètre de contrôle à 2d du nu du poteau
  d   = hauteur utile moyenne
  ρl  = √(ρlx × ρly) ≤ 0,02
  k2  = 0,10 ; CRd,c = 0,18/γc = 0,12
""",

    "6.4.5": """
Clause 6.4.5 — Résistance au poinçonnement avec armatures (Eq. 6.52)

  vRd,cs = 0,75 × vRd,c + 1,5 × (d/sr) × Asw × fywd,ef × (1/(u1 × d)) × sin α

où:
  sr      = espacement radial des armatures de poinçonnement
  Asw     = section d'une couronne d'armatures de poinçonnement
  fywd,ef = min(fywd ; 250 + 0,25d) en MPa (résistance efficace)
  α       = angle des armatures par rapport au plan de la dalle
""",

    # ── Section 7 — États-limites de service ──────────────────────────────

    "7.3.2": """
Clause 7.3.2 — Armature minimale pour le contrôle de la fissuration (Eq. 7.1)

  As,min × σs = kc × k × fct,eff × Act

où:
  As,min  = section minimale d'armature dans la zone tendue
  σs      = contrainte dans les armatures (= fyk pour calcul simplifié)
  kc      = coefficient tenant compte de la répartition des contraintes
            = 0,4 pour flexion pure ; 1,0 pour traction centrique
  k       = coefficient tenant compte des effets des déformations non uniformes
            = 1,0 pour h ≤ 300 mm ; 0,65 pour h ≥ 800 mm (interpolation linéaire)
  fct,eff = résistance moyenne en traction effective = fctm
  Act     = section du béton tendu avant fissuration
""",

    "7.3.4": """
Clause 7.3.4 — Ouverture des fissures (Eq. 7.8, 7.11)

Largeur de fissure (Eq. 7.8):
  wk = sr,max × (εsm - εcm)

où:
  sr,max = espacement maximal des fissures (Eq. 7.11):
    sr,max = k3 × c + k1 × k2 × k4 × φ / ρp,eff

  (εsm - εcm) = [σs - kt × fct,eff/ρp,eff × (1 + αe × ρp,eff)] / Es ≥ 0,6 × σs/Es

Valeurs recommandées: k1=0,8 (acier HA) ; k2=0,5 (flexion) ; k3=3,4 ; k4=0,425
  c       = enrobage de l'armature
  φ       = diamètre des barres
  ρp,eff  = As / Ac,eff  (taux d'armatures effectif)
  σs      = contrainte dans les armatures fissurées
  kt      = 0,4 (charges de longue durée) ; 0,6 (charges de courte durée)
  αe      = Es/Ecm  (rapport modulaire)
""",

    "7.4.2": """
Clause 7.4.2 — Limitation des flèches par les tableaux (Eq. 7.16a, 7.16b)

Rapport l/d limite (Eq. 7.16a, portées ≤ 7m):
  l/d = K × [11 + 1,5 × √fck × ρ0/ρ + 3,2 × √fck × (ρ0/ρ - 1)^(3/2)]
  valable pour ρ ≤ ρ0

Rapport l/d limite (Eq. 7.16b, portées ≤ 7m):
  l/d = K × [11 + 1,5 × √fck × ρ0/(ρ-ρ') + (1/12) × √fck × √(ρ'/ρ0)]
  valable pour ρ > ρ0

où:
  K      = facteur de système structurel (= 1,0 simplement appuyé, 1,3 encastré-libre, etc.)
  ρ0     = taux d'armature de référence = √fck × 10^(-3)
  ρ      = taux d'armature tendue requis en travée
  ρ'     = taux d'armature comprimée en travée
  fck    = en MPa
""",

    # ── Section 8 — Dispositions constructives ───────────────────────────

    "8.4.2": """
Clause 8.4.2 — Contrainte ultime d'adhérence (Eq. 8.2)

  fbd = 2,25 × η1 × η2 × fctd

où:
  fctd  = αct × fctk,0,05 / γc  (résistance de calcul en traction)
  η1    = coefficient relatif aux conditions d'adhérence
          = 1,0 (bonne adhérence) ; 0,7 (mauvaise adhérence)
  η2    = coefficient relatif au diamètre de la barre
          = 1,0 pour φ ≤ 32 mm ; (132 - φ)/100 pour φ > 32 mm
""",

    "8.4.3": """
Clause 8.4.3 — Longueur d'ancrage de base (Eq. 8.3)

  lb,rqd = (φ/4) × (σsd / fbd)

Longueur d'ancrage de calcul (Eq. 8.4):
  lbd = α1 × α2 × α3 × α4 × α5 × lb,rqd ≥ lb,min

Longueur minimale (Eq. 8.6):
  lb,min = max(0,3 × lb,rqd ; 10φ ; 100 mm)  (traction)
  lb,min = max(0,6 × lb,rqd ; 10φ ; 100 mm)  (compression)

où:
  φ     = diamètre de la barre (mm)
  σsd   = contrainte de calcul dans la barre à ancrer
  fbd   = résistance ultime d'adhérence (clause 8.4.2)
  α1    = facteur forme de l'ancrage (1,0 barre droite ; 0,7 courbée si bonne adhérence)
  α2    = facteur enrobage (1,0 standard)
  α3    = facteur confinement par armatures transversales
  α4    = facteur confinement par barres soudées transversales
  α5    = facteur confinement par pression transversale
""",

    "8.7.5.1": """
Clause 8.7.5.1 — Longueur de recouvrement (Eq. 8.10)

  l0 = α1 × α2 × α3 × α5 × α6 × lb,rqd ≥ l0,min

Longueur minimale (Eq. 8.11):
  l0,min = max(0,3 × α6 × lb,rqd ; 15φ ; 200 mm)

où:
  α6    = facteur relatif au pourcentage de barres recouvrées au même endroit
          = 1,4 pour ≥ 50% recouvrées ; 1,5 pour 100%
  Autres αi = comme pour l'ancrage (clause 8.4.3)

Espacement transversal minimal des recouvrements:
  Barres individuelles : 4φ ou 50 mm
""",

    # ── Section 9 — Dispositions constructives éléments ──────────────────

    "9.2.1.1": """
Clause 9.2.1.1 — Armature minimale et maximale des poutres (Eq. 9.1N)

Section minimale d'armature longitudinale tendue:
  As,min = max(0,26 × fctm/fyk × bt × d ; 0,0013 × bt × d)

Section maximale totale:
  As,max = 0,04 × Ac  (traction et compression, hors recouvrements)

où:
  bt    = largeur moyenne de la zone tendue
  d     = hauteur utile
  fctm  = résistance moyenne en traction (Tableau 3.1)
  fyk   = limite d'élasticité caractéristique (MPa)
  Ac    = aire de la section de béton brute
""",

    "9.2.2": """
Clause 9.2.2 — Armatures d'effort tranchant des poutres (Eq. 9.4, 9.5, 9.6)

Taux minimal d'armature transversale (Eq. 9.4):
  ρw,min = 0,08 × √fck / fyk

Espacement maximal longitudinal (Eq. 9.6N):
  sl,max = 0,75 × d × (1 + cot α)

Espacement maximal transversal des branches (Eq. 9.7N):
  st,max = 0,75 × d ≤ 600 mm

où:
  ρw    = Asw / (s × bw × sin α)
  α     = angle des armatures transversales (= 90° pour étriers verticaux)
""",

    "9.3.1.1": """
Clause 9.3.1.1 — Armatures minimales des dalles (Eq. 9.1N)

Section minimale en traction (comme pour les poutres):
  As,min = max(0,26 × fctm/fyk × bt × d ; 0,0013 × bt × d)

Section minimale dans les deux directions pour dalle bidirectionnelle:
  As,min,bidirectionnel = 0,001 × bt × d (par direction, dans la direction secondaire)

Espacement maximal des barres:
  smax = min(3h ; 400 mm)  en zone courante
  smax = min(2h ; 250 mm)  en zone de concentration d'efforts
""",

    "9.4.3": """
Clause 9.4.3 — Armatures de poinçonnement (Eq. 9.11)

Espacement maximal des armatures de poinçonnement:
  sr ≤ 0,75 × d  (espacement radial entre couronnes)
  st ≤ 1,5 × d  (espacement tangentiel au bord de la première couronne)
  st ≤ 2,0 × d  (couronnes suivantes)

Première couronne : entre 0,3d et 0,5d du nu du poteau
""",

}

# ---------------------------------------------------------------------------
# Clause-to-formula lookup with query pattern detection
# ---------------------------------------------------------------------------

def get_formula_for_clause(clause: str) -> Optional[str]:
    """Return the formula text for a given clause number, or None."""
    # Exact match
    if clause in EC2_FORMULAS:
        return EC2_FORMULAS[clause]
    # Prefix match (e.g. query for "6.2" returns 6.2.2 and 6.2.3)
    matches = [v for k, v in EC2_FORMULAS.items() if k.startswith(clause + ".") or k == clause]
    if matches:
        return "\n\n---\n\n".join(matches)
    return None


def get_formula_for_query(query: str) -> Optional[str]:
    """
    Find the most relevant formula for a query using two layers:

    Layer 1 (fast): clause number mention — if query contains "6.2.2", return that formula
    Layer 2 (fast): keyword exact match — covers common technical terms
    Layer 3 (semantic): embedding similarity — handles any phrasing automatically
                        Uses all-MiniLM-L6-v2 already loaded in the pipeline.
                        Threshold = 0.45 (tuned to avoid false matches).
    """
    import re

    # ── Layer 1: clause number in query ──────────────────────────────────
    clause_mentions = re.findall(r'\b(\d+\.\d+(?:\.\d+)*)\b', query)
    for clause in clause_mentions:
        formula = get_formula_for_clause(clause)
        if formula:
            return formula

    # ── Layer 2: keyword fast-path ────────────────────────────────────────
    _KEYWORDS = {
        "cisaillement":              ["6.2.2", "6.2.3"],
        "shear":                     ["6.2.2", "6.2.3"],
        "vrd":                       ["6.2.2", "6.2.3"],
        "effort tranchant":          ["6.2.2", "6.2.3"],
        "résistance au cisaillement":["6.2.2", "6.2.3"],
        "armature d'effort tranchant":["6.2.3"],
        "armatures transversales":   ["6.2.3"],
        "étriers":                   ["6.2.3"],
        "poinçonnement":             ["6.4.3", "6.4.5"],
        "poinconnement":             ["6.4.3", "6.4.5"],
        "punching":                  ["6.4.3", "6.4.5"],
        "flexion":                   ["6.1"],
        "bending":                   ["6.1"],
        "axe neutre":                ["6.1"],
        "neutral axis":              ["6.1"],
        "zone comprimée":            ["6.1"],
        "zone comprimee":            ["6.1"],
        "moment résistant":          ["6.1"],
        "moment resistant":          ["6.1"],
        "moment fléchissant":        ["6.1"],
        "portée efficace":           ["5.3.2.2"],
        "portee efficace":           ["5.3.2.2"],
        "leff":                      ["5.3.2.2"],
        "ancrage":                   ["8.4.3"],
        "anchorage":                 ["8.4.3"],
        "recouvrement":              ["8.7.5.1"],
        "fissure":                   ["7.3.2", "7.3.4"],
        "crack":                     ["7.3.2", "7.3.4"],
        "flèche":                    ["7.4.2"],
        "fleche":                    ["7.4.2"],
        "deflection":                ["7.4.2"],
        "armature minimale":         ["9.2.1.1"],
        "minimum reinforcement":     ["9.2.1.1"],
        "parabole":                  ["3.1.7"],
        "ecu2":                      ["3.1.7"],
        "εcu2":                      ["3.1.7"],
        "fcd":                       ["3.1.6"],
        "résistance de calcul":      ["3.1.6"],
        "résistance à la compression":["3.1.6"],
        "enrobage nominal":          ["4.4.1.1"],
        "cnom":                      ["4.4.1.1"],
        "cmin":                      ["4.4.1.1"],
        "enrobage minimal":          ["4.4.1.1"],
        "nominal cover":             ["4.4.1.1"],
        "minimum cover":             ["4.4.1.1"],
        "élancement":                ["5.8.3.2"],
        "slenderness":               ["5.8.3.2"],
        "redistribution":            ["5.5"],
    }

    q_lower = query.lower()
    for kw, clauses in _KEYWORDS.items():
        if kw.lower() in q_lower:
            results = [f for c in clauses if (f := get_formula_for_clause(c))]
            if results:
                return "\n\n---\n\n".join(results)

    # ── Layer 3: semantic similarity ──────────────────────────────────────
    # Each formula entry has a short description used as the semantic anchor.
    # We embed the query and find the best matching formula.
    _FORMULA_DESCRIPTIONS = {
        "4.4.1.1": "enrobage nominal cnom formule couverture armatures durabilité protection",
        "3.1.6":   "fcd résistance calcul compression béton coefficient partiel gamma_c",
        "3.1.7":   "diagramme parabole rectangle béton déformation contrainte εcu2 sigma_c",
        "3.1.9":   "béton confiné fck,c résistance compression confinement",
        "3.2.7":   "armatures acier diagramme bilinéaire fyd εud déformation acier",
        "5.3.2.2": "portée efficace poutre dalle leff calcul longueur travée",
        "5.5":     "redistribution moments delta xu/d conditions admissible",
        "5.8.3.2": "élancement lambda limite poteau colonne longueur efficace",
        "6.1":     "flexion ELU axe neutre profondeur zone comprimée moment résistant MRd x/d",
        "6.2.2":   "cisaillement effort tranchant VRd,c sans armatures CRd,c bw d",
        "6.2.3":   "cisaillement armatures effort tranchant VRd,s VRd,max étriers angle theta",
        "6.2.4":   "cisaillement âme membrures interface table nervure vEd couture",
        "6.4.3":   "poinçonnement dalle poteau sans armatures vRd,c périmètre contrôle u1",
        "6.4.5":   "poinçonnement armatures vRd,cs résistance dalle poteau",
        "7.3.2":   "fissuration armature minimale As,min contrôle fissures kc fct",
        "7.3.4":   "ouverture fissure wk largeur fissure sr,max εsm εcm",
        "7.4.2":   "flèche déformation admissible l/d rapport élancement dalle poutre",
        "8.4.2":   "adhérence contrainte fbd résistance ancrage armature béton",
        "8.4.3":   "longueur ancrage lb,rqd lbd barre armature ancrage formule",
        "8.7.5.1": "recouvrement longueur l0 barres armature pourcentage recouvrement",
        "9.2.1.1": "armature minimale maximale poutre As,min fctm fyk traction section",
        "9.2.2":   "armatures effort tranchant transversales rho_w espacement sl,max",
        "9.3.1.1": "armatures dalle minimale espacement barres section",
        "9.4.3":   "armatures poinçonnement espacement couronnes dalle poteau",
    }

    try:
        from sentence_transformers import SentenceTransformer
        import math

        # Use the same model already loaded in the pipeline (cached after first load)
        model = SentenceTransformer("all-MiniLM-L6-v2")

        # Embed query
        q_emb = model.encode([query])[0].tolist()

        # Embed all formula descriptions (cached on first call)
        if not hasattr(get_formula_for_query, "_desc_embeddings"):
            descs = list(_FORMULA_DESCRIPTIONS.values())
            embs  = model.encode(descs).tolist()
            get_formula_for_query._desc_embeddings = dict(
                zip(_FORMULA_DESCRIPTIONS.keys(), embs)
            )

        # Cosine similarity
        def cosine(a, b):
            dot   = sum(x*y for x, y in zip(a, b))
            na    = math.sqrt(sum(x*x for x in a))
            nb    = math.sqrt(sum(y*y for y in b))
            return dot / (na * nb) if na and nb else 0.0

        # Find best match above threshold
        THRESHOLD = 0.46
        best_clause, best_score = None, 0.0
        for clause, emb in get_formula_for_query._desc_embeddings.items():
            score = cosine(q_emb, emb)
            if score > best_score:
                best_score = score
                best_clause = clause

        if best_score >= THRESHOLD and best_clause:
            return get_formula_for_clause(best_clause)

    except Exception:
        pass  # semantic layer optional — keyword layer already ran

    return None


def make_formula_chunk(formula_text: str, clause: str,
                       source_id: str = "ec2_2004_nf") -> dict:
    """Build a chunk dict from formula text for injection into context."""
    return {
        "chunk_id":      f"{source_id}_formula_{clause.replace('.', '_')}",
        "source_id":     source_id,
        "chunk_type":    "hierarchical",
        "clause_number": clause,
        "level":         3,
        "ancestor_path": f"Eurocode 2 > [Formula Library] > Clause {clause}",
        "page_number":   0,
        "text":          formula_text.strip(),
        "token_count":   len(formula_text.split()),
        "score":         0.001,
        "cross_refs":    [],
        "metadata":      {"formula_library": True},
    }
