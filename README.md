# StructRAG
 
**Evaluation of a Hierarchical RAG (Retrieval-Augmented Generation) Architecture Applied to Civil Engineering Standards (Eurocode 2 / ACI 318)**
 
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-in%20development-yellow.svg)]()
[![RAG](https://img.shields.io/badge/architecture-Hierarchical%20RAG-orange.svg)]()
[![ChromaDB](https://img.shields.io/badge/vector%20store-ChromaDB-6A0DAD.svg)](https://www.trychroma.com/)
[![Evaluation](https://img.shields.io/badge/evaluation-RAGAS-9cf.svg)](https://github.com/explodinggradients/ragas)
[![Domain](https://img.shields.io/badge/domain-Structural%20Engineering-red.svg)]()
[![Reproducibility](https://img.shields.io/badge/reproducible-yes-brightgreen.svg)]()
 
> A research project investigating whether a hierarchy-aware retrieval pipeline can reduce hallucination risk and improve retrieval accuracy when an LLM is asked to answer questions grounded in structural engineering standards.
 
---
 
## Table of Contents
 
- [Overview](#overview)
- [Project Personnel](#project-personnel)
- [Motivation](#motivation)
- [Problem With Naive RAG on Standards](#problem-with-naive-rag-on-standards)
- [Research Gaps Addressed](#research-gaps-addressed)
- [Objectives](#objectives)
- [Methodology](#methodology)
- [Expected Results](#expected-results)
- [Evaluation Metrics](#evaluation-metrics)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Deliverables](#deliverables)
- [Risks and Limitations](#risks-and-limitations)
- [Data and Code Availability](#data-and-code-availability)
- [References](#references)
- [Status](#status)
---
 
## Overview
 
StructRAG evaluates whether **hierarchical chunking** — indexing a document in a way that preserves its structural nesting (chapter → section → clause → sub-clause → formula/table) — improves retrieval accuracy and reduces hallucination when a Retrieval-Augmented Generation system is queried on structural engineering knowledge.
 
The corpus is deliberately heterogeneous rather than limited to the raw standards: **Eurocode 2** and **ACI 318** provide the normative clauses, while supporting **textbooks and worked-example references** provide applied context — solved design problems, commentary, and derivations that the codes themselves state tersely or omit. This mix is central to the hypothesis, not incidental to it: a practicing engineer rarely consults a code clause in isolation, and a RAG system meant to be useful should be evaluated on its ability to retrieve and reconcile both the normative rule and its applied interpretation.
 
The core hypothesis is simple but consequential for practice: structural standards are not flat text, and neither is their surrounding literature. A clause on minimum reinforcement is only correct in the context of the section, exposure class, and cross-references it sits under — and a worked example applying that clause is only correctly retrieved if the pipeline understands which code, edition, and section it corresponds to. Flattening that hierarchy during chunking — the default behavior of most RAG pipelines — is a plausible source of retrieval errors that could propagate into structural design mistakes if such a tool were ever used in practice.

## Project Personnel

* **Supervisor:**
    * **Afif Beji, Eng., M.Sc.**
        * *Role:* Assistant Professor & Project Supervisor
        * *Institution:* ESPRIT, School of Engineering
* **Interns:**
    * **Mohamed Khelil**
    * **Manef Belkadhi**
    * **Rached Hadj Amor**
        * *Role:* Engineering Students (resp. 3IA / 3A / 4SE)
        * *Institution:* ESPRIT, School of Engineering

---
 
## Motivation
 
Structural engineers routinely consult codes like Eurocode 2 and ACI 318 to verify design compliance, and LLM-based assistants are an increasingly attractive way to speed up that lookup. But these standards are dense, cross-referential, and safety-critical: a wrong minimum cover requirement or an hallucinated formula is not a cosmetic error, it is a potential structural safety issue. A RAG system's value here depends entirely on whether it retrieves the *correct, complete, and correctly contextualized* clause — not just a plausible-sounding one.
 
## Problem With Naive RAG on Standards
 
Standard RAG pipelines typically chunk documents by fixed token windows or paragraph boundaries. Applied to Eurocode 2 or ACI 318, this risks:
 
- Splitting a clause from the exposure class or design condition it depends on.
- Losing the parent-section context that disambiguates similarly worded clauses across chapters.
- Retrieving a formula without the accompanying constraints on its applicability (e.g., unit systems, member type, load case).
Extending the corpus to textbooks and worked examples introduces a second, related failure mode: **cross-document disambiguation**. A worked example illustrating "minimum shear reinforcement" may reference a specific code edition, use different notation than the standard itself, or apply a national annex variant. A naive pipeline that treats all chunks as interchangeable text risks retrieving a worked example that solves a superficially similar but substantively different problem (e.g., a different exposure class or a different code edition), which is arguably a more dangerous failure mode than retrieving nothing, since the answer will look complete and confident.
 
These failure modes are structural to how the documents are organized, not just noise in the retrieval model — which is why a hierarchy-aware chunking strategy, paired with explicit source-and-edition metadata across all document types, is the central object of study here, rather than a marginal optimization.
 
## Research Gaps Addressed
 
| Gap | Common Practice | This Project |
|---|---|---|
| **Flat chunking of hierarchical documents** | Fixed-size or paragraph-level chunking, agnostic to document structure | Hierarchical chunking that preserves chapter/section/clause nesting and cross-references |
| **Lack of domain-specific RAG benchmarks** | General-purpose QA benchmarks not representative of code-compliance queries | Custom QA validation dataset built specifically around Eurocode 2 / ACI 318 compliance questions |
| **Under-specified hallucination evaluation** | Qualitative or anecdotal claims of "reduced hallucination" | Quantitative evaluation using the RAGAS framework (faithfulness, context precision/recall) |
| **Single-standard evaluation** | Most RAG-for-codes studies target one standard in isolation | Comparative evaluation across two structurally different standards (Eurocode 2, ACI 318) |
| **Codes evaluated in isolation from applied literature** | RAG-for-codes studies typically index only the normative text | Corpus deliberately combines normative standards with textbooks and worked examples, testing retrieval across heterogeneous, cross-referencing document types |
 
## Objectives
 
- Develop a RAG pipeline optimized for complex, hierarchically structured civil engineering documents, spanning normative standards and applied reference material.
- Design a hierarchical chunking method that preserves the contextual integrity of structural standards (Eurocode 2, ACI 318) and their related textbooks and worked examples.
- Improve retrieval accuracy for critical information required for structural design (clauses, formulas, applicability conditions, and their applied illustrations).
- Mitigate the risk of LLM hallucination through structure-aware retrieval and explicit source/edition metadata, rather than prompting alone.
- Quantitatively evaluate the resulting system using the RAGAS framework.

## Methodology
 
**Phase 1 — Corpus Acquisition and Source Classification**
Acquire Eurocode 2, ACI 318, and a curated set of related textbooks and worked-example references. Classify each source by type (normative standard, textbook, worked example) and metadata (edition/year, national annex where applicable), since this typing drives both the chunking strategy and later cross-document disambiguation.
 
**Phase 2 — Structural Parsing per Document Type**
Parse the native hierarchy of each source into a structured intermediate representation: parts/sections/clauses/sub-clauses/tables/formulas for the standards, and chapter/example/step structure for textbooks and worked examples. Each document type requires its own parsing logic, since a worked example's internal structure (problem statement → given data → solution steps → result) differs fundamentally from a code's clause structure.
 
**Phase 3 — Hierarchical Chunking Pipeline**
Implement a chunking strategy that respects each document's native hierarchy: chunks are built around complete clauses or complete example steps, annotated with their full ancestor path (e.g., `Eurocode 2 > Part 1 > Section 6 > 6.2 Shear > 6.2.2`, or `Textbook X > Chapter 4 > Worked Example 4.3 > Step 2`), so retrieval preserves context a flat chunker would discard.
 
**Phase 4 — Cross-Document Linking**
Where a textbook or worked example explicitly references a code clause (e.g., "per EC2 §6.2.2"), establish an explicit link between the two chunks in the index, so the pipeline can retrieve the normative clause and its applied illustration together rather than treating them as unrelated text.
 
**Phase 5 — Indexing and Retrieval**
Embed all chunks into ChromaDB as the project's local, persistent vector store, with metadata filtering to allow retrieval to respect scope (e.g., restrict to a given code, edition, exposure class, or member type) alongside semantic similarity.
 
**Phase 6 — Baseline Comparison**
Implement a naive flat-chunking RAG baseline (fixed-size chunks, no hierarchy or source-type metadata) using the same embedding model, LLM, and full corpus, to isolate the effect of hierarchical chunking and cross-document linking specifically.
 
**Phase 7 — QA Benchmark Construction**
Build a validation benchmark of civil-engineering compliance questions (e.g., minimum reinforcement ratios, cover requirements, load combination factors), each paired with a ground-truth answer traceable to a specific clause and, where applicable, a corresponding worked example. Worked examples with known correct numerical answers are a natural source of verifiable QA pairs and are used deliberately for this reason.
 
**Phase 8 — Quantitative Evaluation with RAGAS**
Evaluate both the hierarchical and flat-chunking pipelines on the benchmark using the RAGAS framework, scoring faithfulness, answer relevance, context precision, and context recall.
 
```text
Query: "What is the minimum concrete cover for a beam in exposure class XC2 per Eurocode 2?"
 
Flat RAG retrieval:         [clause text only, exposure class table on a separate page, not retrieved]
Hierarchical RAG retrieval: [clause text + linked exposure class table + parent section context
                             + linked worked example applying the same clause]
```
 
## Expected Results
 
| Configuration | Purpose | Expected Direction |
|---|---|---|
| Flat-chunking RAG (baseline) | Establish naive retrieval performance | Lower context precision/recall on cross-referential queries |
| Hierarchical RAG, codes only | Isolate the effect of hierarchy on normative text alone | Higher context precision/recall vs. baseline |
| Hierarchical RAG + linked textbooks/examples (main contribution) | Test whether applied context further improves grounding | Further improvement in faithfulness and answer completeness, particularly on "how is this applied" queries |
| Eurocode 2 vs. ACI 318 comparison | Test generalization across document structures | Improvement expected in both, magnitude may differ by document formatting style |
 
These are **directional hypotheses**, not observed results — the project's purpose is precisely to test whether hierarchical chunking outperforms the flat baseline, and by how much, rather than to assume a specific numeric improvement in advance.
 
## Evaluation Metrics
 
- **Faithfulness** (RAGAS) — whether generated answers are grounded in retrieved context, not fabricated
- **Context Precision** (RAGAS) — whether retrieved chunks are actually relevant to the query
- **Context Recall** (RAGAS) — whether all necessary context was retrieved
- **Answer Relevance** (RAGAS) — whether the generated answer actually addresses the question
- Manual expert review of a held-out sample, given the safety-critical nature of the domain

## Repository Structure
 
```
structrag/
├── data/
│   ├── raw_standards/          # Eurocode 2 / ACI 318 source documents (licensing permitting)
│   ├── raw_textbooks/          # Related textbooks / worked-example references (licensing permitting)
│   ├── source_manifest.json    # Tracks source type, edition, license terms per document
│   └── qa_benchmark/           # Custom compliance QA validation dataset
├── parsing/
│   ├── standards_parser.py     # Extracts native hierarchy from codes (sections, clauses, tables)
│   ├── textbook_parser.py      # Extracts chapter/example/step structure from textbooks
│   └── hierarchy_schema.py     # Shared data model for nested document structure
├── chunking/
│   ├── hierarchical_chunker.py
│   └── flat_chunker.py         # Baseline for comparison
├── linking/
│   └── cross_reference_linker.py  # Links worked examples/textbook passages to code clauses
├── retrieval/
│   ├── indexer.py              # ChromaDB collection construction and embedding ingestion
│   ├── chroma_client.py        # Local persistent ChromaDB client (metadata schema, collections)
│   └── retriever.py            # Metadata-aware retrieval logic (filters by source type, edition, exposure class)
├── rag_pipeline/
│   ├── generation.py
│   └── app.py                  # Functional RAG application entry point
├── evaluation/
│   ├── ragas_runner.py
│   └── benchmark_report.py
├── reports/
│   └── figures/
├── requirements.txt
├── README.md
└── LICENSE
```
 
## Deliverables
 
- A Python pipeline for processing and hierarchically indexing structural engineering standards.
- A functional RAG application.
- A validation benchmark dataset (QA) specialized in civil engineering compliance.
- A research manuscript detailing the quantitative evaluation (utilizing the RAGAS framework).

## Risks and Limitations
 
- **Document licensing is the primary practical risk, and it now applies to a broader corpus.** Eurocode 2 and ACI 318 are sold by CEN and ACI respectively; textbooks and worked-example references are typically copyrighted commercial publications as well. Every source added to the corpus needs its licensing terms checked individually — access via an institutional library or personal purchase does not automatically grant redistribution or public-repository rights. This is addressed procedurally via the `source_manifest.json` tracking file, but it remains a real constraint on what can be publicly released, not just documented.
- **Cross-document linking is a new technical risk.** Reliably matching a textbook's informal citation style (e.g., "see EC2 6.2.2" vs. "per §6.2.2 of the code") to the correct normative clause is nontrivial and may require a mix of rule-based matching and manual verification, particularly across national annexes or older textbook editions referencing superseded code versions.
- **Ground-truth QA construction** for a technical, safety-critical domain requires care; benchmark questions and answers — including those derived from worked examples — should ideally be reviewed by someone with structural engineering domain knowledge, not generated purely automatically.
- **Generalization across sources:** Eurocode 2, ACI 318, and the selected textbooks differ in formatting conventions, notation, and structure, so parsing logic will likely need source-specific adaptation rather than a single universal parser.
- **Safety caveat:** this project is a research evaluation of a retrieval architecture, not a certified design tool, and outputs should not be used for actual structural design decisions without independent verification by a qualified engineer.

## Data and Code Availability
 
Eurocode 2 and ACI 318 are subject to licensing restrictions from their respective standards bodies (CEN and ACI). The textbooks and worked-example references used to extend the corpus are, in most cases, also commercially copyrighted material. This project will:
 
- Maintain a `source_manifest.json` documenting the license status of every source in the corpus.
- Avoid redistributing full standard or textbook text in the public repository.
- Release the QA benchmark with citations to specific clauses and worked-example locations (e.g., "Textbook Y, Example 4.3") rather than reproduced source text.
- Release all original code — parsers, chunkers, linker, retrieval, and evaluation pipeline — publicly, independent of the source documents' licensing status.
Where possible, open-access national annexes or publicly available code commentary will be prioritized over commercial textbooks to maximize what can be shared for reproducibility.
 
## References
 
- [RAGAS: Retrieval Augmented Generation Assessment](https://github.com/explodinggradients/ragas)
- Eurocode 2 — *Design of concrete structures* (EN 1992-1-1), CEN
- ACI 318 — *Building Code Requirements for Structural Concrete*, American Concrete Institute
- Supporting textbooks and worked-example references (specific titles to be finalized based on institutional access and licensing review — see [Data and Code Availability](#data-and-code-availability))

## Status
 
This repository is in active development. Corpus classification and structural parsing across standards and textbooks (Phases 1–2) are the current focus; cross-document linking, the flat-chunking baseline, and RAGAS-based quantitative evaluation will be added incrementally as milestones are completed.
 
---
 
<sub>Maintained as part of an academic research internship. Contributions, issues, and methodological critique are welcome via GitHub Issues.</sub>
