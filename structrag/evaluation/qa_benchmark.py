"""
structrag/evaluation/qa_benchmark.py
--------------------------------------
QA benchmark: 40 questions in French + 10 in English = 50 total.

The EC2 document is in French (NF EN 1992-1-1:2004, AFNOR).
French questions test the pipeline in the document's native language.
English questions test cross-language retrieval capability.

Categories: cover, materials, design_basis, analysis, uls, sls,
            detailing, cross_clause, out_of_scope
"""

QA_BENCHMARK = [

    # ── FRENCH QUESTIONS (40) ─────────────────────────────────────────────

    # Cover / Durabilité (Section 4)
    {
        "question": "Quelle est la formule de l'enrobage nominal ?",
        "ground_truth": "L'enrobage nominal est cnom = cmin + Δcdev, où cmin est l'enrobage minimal et Δcdev est la tolérance d'exécution (clause 4.4.1.1).",
        "ground_truth_clauses": ["4.4.1.1"],
        "category": "cover", "difficulty": "easy", "in_scope": True, "lang": "fr",
    },
    {
        "question": "Quelle est la valeur recommandée de la tolérance d'exécution Δcdev ?",
        "ground_truth": "La valeur recommandée est 10 mm. Elle peut être réduite à 5 mm avec un système qualité ou à 0 mm pour les éléments préfabriqués (clause 4.4.1.3).",
        "ground_truth_clauses": ["4.4.1.3"],
        "category": "cover", "difficulty": "easy", "in_scope": True, "lang": "fr",
    },
    {
        "question": "Quel est l'enrobage minimal pour la classe d'exposition XC1, classe structurale S4 ?",
        "ground_truth": "Pour XC1 et la classe structurale S4, l'enrobage minimal cmin,dur est de 15 mm (Tableau 4.4N, clause 4.4.1.2).",
        "ground_truth_clauses": ["4.4.1.2"],
        "category": "cover", "difficulty": "medium", "in_scope": True, "lang": "fr",
    },
    {
        "question": "Quel est l'enrobage minimal pour la classe d'exposition XC2, classe structurale S4 ?",
        "ground_truth": "Pour XC2/XC3 et la classe structurale S4, l'enrobage minimal cmin,dur est de 25 mm (Tableau 4.4N, clause 4.4.1.2).",
        "ground_truth_clauses": ["4.4.1.2"],
        "category": "cover", "difficulty": "medium", "in_scope": True, "lang": "fr",
    },
    {
        "question": "Quel est l'enrobage minimal pour la classe d'exposition XD3, classe structurale S4 ?",
        "ground_truth": "Pour XD3/XS3 et la classe structurale S4, l'enrobage minimal cmin,dur est de 45 mm (Tableau 4.4N, clause 4.4.1.2).",
        "ground_truth_clauses": ["4.4.1.2"],
        "category": "cover", "difficulty": "medium", "in_scope": True, "lang": "fr",
    },
    {
        "question": "Quelles sont les classes d'exposition liées à la carbonatation ?",
        "ground_truth": "XC1 (sec ou humide en permanence), XC2 (humide, rarement sec), XC3 (humidité modérée), XC4 (alternativement humide et sec). Tableau 4.1, clause 4.2.",
        "ground_truth_clauses": ["4.2"],
        "category": "cover", "difficulty": "easy", "in_scope": True, "lang": "fr",
    },
    {
        "question": "Quelles sont les classes d'exposition liées aux chlorures d'origine non marine ?",
        "ground_truth": "XD1 (humidité modérée), XD2 (humide, rarement sec), XD3 (alternativement humide et sec). Tableau 4.1, clause 4.2.",
        "ground_truth_clauses": ["4.2"],
        "category": "cover", "difficulty": "easy", "in_scope": True, "lang": "fr",
    },
    {
        "question": "Quelle classe de résistance minimale du béton le Tableau 4.3N recommande-t-il pour la classe d'exposition XD3 ?",
        "ground_truth": "Le Tableau 4.3N recommande une classe de résistance minimale ≥ C45/55 pour XD3/XS2/XS3 (avec possibilité de minoration d'une classe). Clause 4.4.1.2.",
        "ground_truth_clauses": ["4.4.1.2"],
        "category": "cover", "difficulty": "medium", "in_scope": True, "lang": "fr",
    },
    {
        "question": "Quelle classe de résistance minimale du béton le Tableau 4.3N recommande-t-il pour XC2 ?",
        "ground_truth": "Le Tableau 4.3N recommande une classe de résistance minimale ≥ C35/45 pour XC2/XC3. Clause 4.4.1.2.",
        "ground_truth_clauses": ["4.4.1.2"],
        "category": "cover", "difficulty": "medium", "in_scope": True, "lang": "fr",
    },
    {
        "question": "Quelle est la classe structurale recommandée pour une durée d'utilisation de 50 ans ?",
        "ground_truth": "La classe structurale recommandée pour une durée de 50 ans est S4 (clause 4.4.1.2, Tableau 4.3N).",
        "ground_truth_clauses": ["4.4.1.2"],
        "category": "cover", "difficulty": "easy", "in_scope": True, "lang": "fr",
    },
    {
        "question": "Combien de classes structurales sont définies dans l'Eurocode 2 ?",
        "ground_truth": "Six classes structurales sont définies : S1 à S6. Elles sont utilisées avec les classes d'exposition pour déterminer l'enrobage minimal. Clause 4.4.1.2, Tableau 4.3N.",
        "ground_truth_clauses": ["4.4.1.2"],
        "category": "cover", "difficulty": "easy", "in_scope": True, "lang": "fr",
    },
    {
        "question": "Quelles sont les trois exigences que l'enrobage minimal doit satisfaire simultanément ?",
        "ground_truth": "L'enrobage minimal doit satisfaire : (1) les exigences d'adhérence (cmin,b), (2) les conditions environnementales (cmin,dur), (3) un minimum de 10 mm. La valeur retenue est le maximum. Clause 4.4.1.2 équation (4.2).",
        "ground_truth_clauses": ["4.4.1.2"],
        "category": "cover", "difficulty": "medium", "in_scope": True, "lang": "fr",
    },

    # Matériaux (Section 3)
    {
        "question": "Quelle est la valeur du coefficient partiel γc pour le béton en situation durable ?",
        "ground_truth": "Le coefficient partiel pour le béton est γc = 1,5 en situation durable/transitoire. Tableau 2.1N, clause 2.4.2.4.",
        "ground_truth_clauses": ["2.4.2.4"],
        "category": "materials", "difficulty": "easy", "in_scope": True, "lang": "fr",
    },
    {
        "question": "Quelle est la valeur du coefficient partiel γs pour l'acier de béton armé en situation durable ?",
        "ground_truth": "Le coefficient partiel pour l'acier de béton armé est γs = 1,15 en situation durable/transitoire. Tableau 2.1N, clause 2.4.2.4.",
        "ground_truth_clauses": ["2.4.2.4"],
        "category": "materials", "difficulty": "easy", "in_scope": True, "lang": "fr",
    },
    {
        "question": "Quelle est la valeur du coefficient partiel γc en situation accidentelle ?",
        "ground_truth": "En situation accidentelle, le coefficient partiel pour le béton est γc = 1,2. Tableau 2.1N, clause 2.4.2.4.",
        "ground_truth_clauses": ["2.4.2.4"],
        "category": "materials", "difficulty": "medium", "in_scope": True, "lang": "fr",
    },
    {
        "question": "Quelle est la déformation ultime de compression εcu2 pour les bétons jusqu'à C50/60 ?",
        "ground_truth": "Pour les bétons jusqu'à C50/60, la déformation ultime εcu2 est de 3,5 ‰. Tableau 3.1, clause 3.1.2.",
        "ground_truth_clauses": ["3.1.2"],
        "category": "materials", "difficulty": "medium", "in_scope": True, "lang": "fr",
    },
    {
        "question": "Quelle est la résistance caractéristique en compression fck pour le béton C30/37 ?",
        "ground_truth": "Pour le béton C30/37, la résistance caractéristique en compression est fck = 30 MPa. Tableau 3.1, clause 3.1.2.",
        "ground_truth_clauses": ["3.1.2"],
        "category": "materials", "difficulty": "easy", "in_scope": True, "lang": "fr",
    },
    {
        "question": "Quelle est la résistance moyenne en traction fctm pour le béton C25/30 ?",
        "ground_truth": "Pour le béton C25/30, la résistance moyenne en traction est fctm = 2,6 MPa. Tableau 3.1, clause 3.1.2.",
        "ground_truth_clauses": ["3.1.2"],
        "category": "materials", "difficulty": "medium", "in_scope": True, "lang": "fr",
    },
    {
        "question": "Quel est le module d'élasticité sécant Ecm pour le béton C30/37 ?",
        "ground_truth": "Pour le béton C30/37, le module d'élasticité sécant est Ecm = 33 GPa. Tableau 3.1, clause 3.1.2.",
        "ground_truth_clauses": ["3.1.2"],
        "category": "materials", "difficulty": "medium", "in_scope": True, "lang": "fr",
    },
    {
        "question": "Quelle est la formule de la résistance de calcul en compression fcd ?",
        "ground_truth": "La résistance de calcul en compression est fcd = αcc × fck / γc, où αcc tient compte des effets à long terme (valeur recommandée 1,0) et γc = 1,5. Clause 3.1.6.",
        "ground_truth_clauses": ["3.1.6"],
        "category": "materials", "difficulty": "medium", "in_scope": True, "lang": "fr",
    },

    # Bases de calcul (Section 2)
    {
        "question": "Quelle est la différence entre un Principe et une Règle d'Application dans l'Eurocode 2 ?",
        "ground_truth": "Les Principes sont des exigences obligatoires identifiées par P après le numéro de paragraphe (ex. (1)P). Les Règles d'Application sont des méthodes généralement acceptées conformes aux Principes. Clause 1.4.",
        "ground_truth_clauses": ["1.4"],
        "category": "design_basis", "difficulty": "easy", "in_scope": True, "lang": "fr",
    },

    # Analyse structurale (Section 5)
    {
        "question": "Quelles sont les méthodes d'analyse structurale reconnues par l'Eurocode 2 ?",
        "ground_truth": "L'Eurocode 2 reconnaît : l'analyse élastique linéaire, l'analyse élastique linéaire avec redistribution limitée, l'analyse plastique et l'analyse non linéaire. Clause 5.1.1.",
        "ground_truth_clauses": ["5.1.1"],
        "category": "analysis", "difficulty": "medium", "in_scope": True, "lang": "fr",
    },
    {
        "question": "Comment calcule-t-on la portée efficace d'une poutre selon l'Eurocode 2 ?",
        "ground_truth": "La portée efficace est leff = ln + a1 + a2, où ln est la portée entre nus d'appuis et a1, a2 sont les distances jusqu'au centre d'appui effectif. Clause 5.3.2.2.",
        "ground_truth_clauses": ["5.3.2.2"],
        "category": "analysis", "difficulty": "medium", "in_scope": True, "lang": "fr",
    },

    # États-limites ultimes (Section 6)
    {
        "question": "Quelle est la formule de la résistance au cisaillement VRd,c pour les éléments sans armature d'effort tranchant ?",
        "ground_truth": "VRd,c = [CRd,c × k × (100 × ρl × fck)^(1/3) + k1 × σcp] × bw × d avec une valeur minimale. CRd,c, k1 et vmin sont déterminés nationalement. Clause 6.2.2.",
        "ground_truth_clauses": ["6.2.2"],
        "category": "uls", "difficulty": "hard", "in_scope": True, "lang": "fr",
    },
    {
        "question": "Quelle est la valeur recommandée de CRd,c dans la formule de résistance au cisaillement ?",
        "ground_truth": "La valeur recommandée de CRd,c est 0,18/γc = 0,18/1,5 = 0,12. Clause 6.2.2 NOTE.",
        "ground_truth_clauses": ["6.2.2"],
        "category": "uls", "difficulty": "medium", "in_scope": True, "lang": "fr",
    },
    {
        "question": "Quelle est la section minimale d'armature d'effort tranchant ρw,min ?",
        "ground_truth": "La section minimale d'armature transversale est ρw,min = 0,08 × √fck / fyk. Clause 9.2.2.",
        "ground_truth_clauses": ["9.2.2"],
        "category": "uls", "difficulty": "medium", "in_scope": True, "lang": "fr",
    },

    # États-limites de service (Section 7)
    {
        "question": "Quelle est la largeur de fissure maximale pour la classe d'exposition XC1 sous charges quasi-permanentes ?",
        "ground_truth": "La largeur de fissure maximale recommandée wmax est de 0,3 mm pour les classes XC1 à XC4 sous la combinaison quasi-permanente. Clause 7.3.1, Tableau 7.1N.",
        "ground_truth_clauses": ["7.3.1"],
        "category": "sls", "difficulty": "medium", "in_scope": True, "lang": "fr",
    },
    {
        "question": "Quelle est la contrainte de compression maximale admissible pour éviter la fissuration longitudinale ?",
        "ground_truth": "Pour éviter la fissuration longitudinale, la contrainte de compression ne doit pas dépasser 0,6 × fck sous la combinaison caractéristique. Clause 7.2.",
        "ground_truth_clauses": ["7.2"],
        "category": "sls", "difficulty": "medium", "in_scope": True, "lang": "fr",
    },

    # Dispositions constructives (Sections 8-9)
    {
        "question": "Quel est le diamètre minimal du mandrin de pliage pour des barres de diamètre ≤ 16 mm ?",
        "ground_truth": "Le diamètre minimal du mandrin est 4 × φ pour les barres de diamètre ≤ 16 mm. Clause 8.3.",
        "ground_truth_clauses": ["8.3"],
        "category": "detailing", "difficulty": "medium", "in_scope": True, "lang": "fr",
    },
    {
        "question": "Quelle est la section minimale d'armature longitudinale pour une poutre ?",
        "ground_truth": "La section minimale d'armature pour une poutre est As,min = 0,26 × fctm/fyk × bt × d ≥ 0,0013 × bt × d. Clause 9.2.1.1.",
        "ground_truth_clauses": ["9.2.1.1"],
        "category": "detailing", "difficulty": "medium", "in_scope": True, "lang": "fr",
    },
    {
        "question": "Quel est l'espacement maximal des armatures transversales dans une poutre ?",
        "ground_truth": "L'espacement longitudinal maximal des armatures transversales est sl,max = 0,75 × d × (1 + cot α). Clause 9.2.2.",
        "ground_truth_clauses": ["9.2.2"],
        "category": "detailing", "difficulty": "medium", "in_scope": True, "lang": "fr",
    },
    {
        "question": "Quelle est la distance minimale entre barres parallèles individuelles ?",
        "ground_truth": "La distance libre doit être ≥ max(k1 × φ, dg + k2, 20 mm) où dg est la taille maximale des granulats. Recommandé : k1=1, k2=5mm. Clause 8.2.",
        "ground_truth_clauses": ["8.2"],
        "category": "detailing", "difficulty": "medium", "in_scope": True, "lang": "fr",
    },
    {
        "question": "Quelle est la formule de la longueur d'ancrage de base pour une barre droite ?",
        "ground_truth": "La longueur d'ancrage de base est lb,rqd = (φ/4) × (σsd / fbd), où fbd est la résistance d'adhérence de calcul. Clause 8.4.3.",
        "ground_truth_clauses": ["8.4.3"],
        "category": "detailing", "difficulty": "hard", "in_scope": True, "lang": "fr",
    },
    {
        "question": "Quel est le taux maximum d'armature en zone comprimée d'une poutre ?",
        "ground_truth": "La section maximale d'armature en compression ne doit pas dépasser 0,04 × Ac (4 % de la section brute). Clause 9.2.1.1.",
        "ground_truth_clauses": ["9.2.1.1"],
        "category": "detailing", "difficulty": "easy", "in_scope": True, "lang": "fr",
    },

    # Requêtes multi-clauses (difficile)
    {
        "question": "Pour une poutre en classe d'exposition XC2, classe structurale S4 — quel est l'enrobage minimal et quelle classe de béton est requise ?",
        "ground_truth": "Pour XC2/XC3, S4 : enrobage minimal cmin,dur = 25 mm (Tableau 4.4N). Classe de béton requise : ≥ C35/45 (Tableau 4.3N). Clause 4.4.1.2.",
        "ground_truth_clauses": ["4.4.1.2"],
        "category": "cross_clause", "difficulty": "hard", "in_scope": True, "lang": "fr",
    },
    {
        "question": "Quel enrobage nominal doit-on spécifier sur les plans pour une dalle en XC3, durée de projet 50 ans ?",
        "ground_truth": "Pour XC2/XC3, S4 (50 ans) : cmin,dur=25mm. Pour une dalle (réduction S3 possible) : cmin,dur=20mm. Avec Δcdev=10mm : cnom=30mm. Clauses 4.4.1.1, 4.4.1.2, 4.4.1.3.",
        "ground_truth_clauses": ["4.4.1.1", "4.4.1.2", "4.4.1.3"],
        "category": "cross_clause", "difficulty": "hard", "in_scope": True, "lang": "fr",
    },

    # Hors champ (doivent être refusées)
    {
        "question": "Quelle est la formule de la charge de vent selon l'Eurocode 1 ?",
        "ground_truth": "Hors champ — les charges de vent sont dans l'Eurocode 1 (EN 1991), pas l'Eurocode 2.",
        "ground_truth_clauses": [],
        "category": "out_of_scope", "difficulty": "easy", "in_scope": False, "lang": "fr",
    },
    {
        "question": "Quel facteur de comportement q utiliser pour un portique béton en zone sismique 3 ?",
        "ground_truth": "Hors champ — les facteurs de comportement sont dans l'Eurocode 8 (EN 1998).",
        "ground_truth_clauses": [],
        "category": "out_of_scope", "difficulty": "easy", "in_scope": False, "lang": "fr",
    },
    {
        "question": "Quel rapport eau/ciment maximal pour la classe d'exposition XC1 ?",
        "ground_truth": "Non directement dans l'EC2 — les rapports eau/ciment sont dans l'EN 206-1, référencé par la clause 4.1 mais les valeurs sont dans la norme matériau séparée.",
        "ground_truth_clauses": [],
        "category": "out_of_scope", "difficulty": "medium", "in_scope": False, "lang": "fr",
    },

    # ── ENGLISH QUESTIONS (10) ────────────────────────────────────────────

    {
        "question": "What is the formula for nominal concrete cover?",
        "ground_truth": "cnom = cmin + Δcdev, where cmin is the minimum cover and Δcdev is the allowance for execution deviation (EC2 clause 4.4.1.1).",
        "ground_truth_clauses": ["4.4.1.1"],
        "category": "cover", "difficulty": "easy", "in_scope": True, "lang": "en",
    },
    {
        "question": "What partial safety factor should I use for concrete in persistent design situations?",
        "ground_truth": "The partial safety factor for concrete gamma_c is 1.5 for persistent and transient design situations. From Table 2.1N, clause 2.4.2.4.",
        "ground_truth_clauses": ["2.4.2.4"],
        "category": "materials", "difficulty": "easy", "in_scope": True, "lang": "en",
    },
    {
        "question": "What partial safety factor applies to reinforcing steel?",
        "ground_truth": "The partial safety factor for reinforcing steel gamma_s is 1.15 for persistent and transient design situations. Table 2.1N, clause 2.4.2.4.",
        "ground_truth_clauses": ["2.4.2.4"],
        "category": "materials", "difficulty": "easy", "in_scope": True, "lang": "en",
    },
    {
        "question": "What is the minimum cover for exposure class XC2, structural class S4?",
        "ground_truth": "For XC2/XC3 and structural class S4, the minimum cover cmin,dur is 25 mm, from Table 4.4N of clause 4.4.1.2.",
        "ground_truth_clauses": ["4.4.1.2"],
        "category": "cover", "difficulty": "medium", "in_scope": True, "lang": "en",
    },
    {
        "question": "What minimum concrete strength class is required for exposure class XD3?",
        "ground_truth": "Table 4.3N recommends a minimum strength class of C45/55 for XD3/XS2/XS3. Clause 4.4.1.2.",
        "ground_truth_clauses": ["4.4.1.2"],
        "category": "cover", "difficulty": "medium", "in_scope": True, "lang": "en",
    },
    {
        "question": "What are the exposure classes for carbonation-induced corrosion?",
        "ground_truth": "XC1 (dry or permanently wet), XC2 (wet rarely dry), XC3 (moderate humidity), XC4 (cyclic wet and dry). Table 4.1, clause 4.2.",
        "ground_truth_clauses": ["4.2"],
        "category": "cover", "difficulty": "easy", "in_scope": True, "lang": "en",
    },
    {
        "question": "How do I calculate nominal cover from minimum cover?",
        "ground_truth": "Nominal cover = minimum cover + execution tolerance allowance Δcdev (recommended 10 mm). Formula: cnom = cmin + Δcdev. Clauses 4.4.1.1 and 4.4.1.3.",
        "ground_truth_clauses": ["4.4.1.1", "4.4.1.3"],
        "category": "cover", "difficulty": "easy", "in_scope": True, "lang": "en",
    },
    {
        "question": "What is the ultimate compressive strain εcu2 for concrete up to C50/60?",
        "ground_truth": "For concrete classes up to C50/60, εcu2 = 3.5 permille. From Table 3.1, clause 3.1.2.",
        "ground_truth_clauses": ["3.1.2"],
        "category": "materials", "difficulty": "medium", "in_scope": True, "lang": "en",
    },
    {
        "question": "What is the formula for wind load on a tall structure?",
        "ground_truth": "Out of scope — wind loads are covered by Eurocode 1 (EN 1991), not Eurocode 2.",
        "ground_truth_clauses": [],
        "category": "out_of_scope", "difficulty": "easy", "in_scope": False, "lang": "en",
    },
    {
        "question": "What behavior factor should I use for seismic design?",
        "ground_truth": "Out of scope — seismic behavior factors are defined in Eurocode 8 (EN 1998), not Eurocode 2.",
        "ground_truth_clauses": [],
        "category": "out_of_scope", "difficulty": "easy", "in_scope": False, "lang": "en",
    },
]
