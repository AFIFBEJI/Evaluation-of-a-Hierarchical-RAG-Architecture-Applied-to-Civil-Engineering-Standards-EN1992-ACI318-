# StructRAG — Hierarchical RAG for Civil Engineering Standards

> Evaluation of a Hierarchical RAG Architecture Applied to Civil Engineering Standards (Eurocode 2 / ACI 318)

---

## What this project is about

Structural engineers consult standards like Eurocode 2 and ACI 318 constantly — to verify cover requirements, check partial safety factors, confirm minimum reinforcement ratios. These standards are dense, cross-referential, and safety-critical. A wrong formula or a hallucinated clause number is not a cosmetic error; it is a potential structural safety issue.

LLM-based assistants are increasingly attractive for speeding up code lookups. But standard RAG pipelines chunk documents by fixed token windows, completely ignoring the document's structure. Applied to Eurocode 2, this means:

- A clause gets split from the applicability condition it depends on
- A formula is retrieved without the constraints that define when it applies
- A NOTE containing a national annex value ends up in a different chunk than the clause it modifies

**This project tests whether respecting the document's native hierarchy during indexing produces meaningfully better retrieval — and whether that improvement can be measured quantitatively.**

The core comparison is:

| Pipeline | Strategy |
|---|---|
| **Hierarchical RAG** (main contribution) | One chunk = one complete clause + all its child paragraphs, notes, and tables. The full ancestor path is prepended to the chunk text. |
| **Flat RAG** (baseline) | Fixed 512-token sliding window, no structure awareness. |

Both pipelines use the same embedding model, the same LLM, and the same corpus. Any difference in answer quality is attributable solely to the chunking strategy.

---

## Project structure

```
structrag/
├── parsing/
│   ├── hierarchy_schema.py       # shared Node dataclass used by all parsers
│   ├── standards_parser.py       # EC2 PDF parser (font-size + bold heading detection)
│   └── table_extractor.py        # pdfplumber table extraction for numeric tables
├── chunking/
│   ├── hierarchical_chunker.py   # clause-boundary chunker (main contribution)
│   └── flat_chunker.py           # sliding-window baseline
├── retrieval/
│   ├── chroma_client.py          # ChromaDB persistent client setup
│   ├── indexer.py                # batch upsert with full metadata
│   ├── retriever.py              # metadata-filtered vector search
│   ├── hybrid_retriever.py       # BM25 + vector RRF fusion
│   └── inject_tables.py          # injects extracted table chunks into ChromaDB
├── rag_pipeline/
│   ├── generation.py             # OpenAI chat completions backend
│   ├── groq_generation.py        # Groq backend with domain classifier
│   └── app.py                    # CLI / REPL entry point
├── evaluation/
│   └── __init__.py               # placeholder — RAGAS runner to be added
├── data/
│   ├── source_manifest.json      # document registry (license, edition, parse status)
│   ├── parsed/                   # output of standards_parser.py (gitignored)
│   └── chunks/                   # output of chunkers (gitignored)
tests/
├── rag_eval.py                   # 14-question evaluation harness
├── rag_eval_results.txt          # v1 results (k=5, vector only)
├── rag_eval_results_v2.txt       # v2 results (k=8, hybrid + tables)
├── inspect_chunks.py             # chunk content inspector
└── probe_tables.py               # pdfplumber table structure probe
run_pipeline.py                   # master runner: parse -> chunk -> index -> test
requirements.txt
```

---

## How to run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your API key

Create a `.env` file in the project root (already in `.gitignore`):

```
GROQ_API_KEY=gsk_your_key_here
GROQ_MODEL=openai/gpt-oss-120b
```

Get a key at [console.groq.com](https://console.groq.com).

### 3. Add the EC2 PDF

Place the Eurocode 2 PDF in the project root. The expected filename is:
```
FA039724-NF EN 1992-1-1 (octobre 2005) Eurocode 2.pdf
```

### 4. Run the full pipeline

```bash
# Full run: parse -> chunk -> index -> smoke test
python run_pipeline.py

# Skip parsing if nodes JSON already exists
python run_pipeline.py --from 2

# Re-index only (chunks already built)
python run_pipeline.py --from 4
```

### 5. Query the system

```bash
# Retrieval only — no API key needed
python structrag/rag_pipeline/app.py --no-llm --query "minimum cover for XC2"

# Full RAG with Groq
python structrag/rag_pipeline/app.py --groq --query "What is the shear resistance formula?"

# Interactive REPL
python structrag/rag_pipeline/app.py --groq
```

### 6. Run the evaluation harness

```bash
python tests/rag_eval.py --out tests/my_results.txt
```

---

## What is done

### Parsing pipeline
- **EC2 PDF fully parsed** into 2,132 structured nodes using PyMuPDF
  - 12 sections, 86 clauses, 275 subclauses, 1,326 paragraphs, 266 notes, 167 image blocks
  - Cross-references extracted automatically per paragraph (`cross_refs` field)
- **5 critical tables extracted** with pdfplumber and injected as searchable chunks:
  - Table 2.1N — partial safety factors (γc = 1.5, γs = 1.15)
  - Table 3.1 — concrete strength and deformation properties (εcu2 values)
  - Table 4.1 — all exposure classes (XC, XD, XS, XF, XA)
  - Table 4.3N — structural classification S1–S6 vs exposure class
  - Table 4.4N — minimum cover values (mm) for every S-class × exposure class

### Chunking
- **Hierarchical chunker** — 467 chunks, each a complete clause with all child content, ancestor path prepended
- **Flat baseline chunker** — 315 chunks, sliding 512-token window, no structure awareness

### Retrieval
- **ChromaDB** — persistent local vector store, two collections (`ec2_hierarchical`, `ec2_flat`)
- **Embedding model** — `all-MiniLM-L6-v2` (multilingual, 22M params)
- **Hybrid retriever** — BM25 + cosine similarity fused with Reciprocal Rank Fusion (k=8)
  - BM25 catches exact symbolic queries ("γc", "1,5", "XC2") that embeddings miss
  - Vector search catches semantic queries ("shear resistance for members without reinforcement")
- **Metadata filtering** — filter by `clause_number`, `level`, `source_id` before semantic ranking

### RAG pipeline
- **OpenAI backend** — `generation.py` with citation-enforcing system prompt
- **Groq backend** — `groq_generation.py` with:
  - Two-layer domain classifier (keyword scan + embedding similarity)
  - Empty-context guard (never calls LLM with no results)
  - 429 rate-limit handling
  - Configurable model via `GROQ_MODEL` env var
- **App** — CLI with `--groq`, `--no-llm`, `--no-compare` flags + interactive REPL

### Evaluation
- **14-question harness** covering: cover requirements, partial safety factors, exposure classes, structural classification, stress-strain models, out-of-scope rejection
- **v1 results** (k=5, vector only, no tables): 3/14 correct answers
- **v2 results** (k=8, hybrid BM25+vector, tables injected): 7/14 correct answers
- Confirmed 1 question (w/c ratio for XC1) is legitimately not in EC2 — those values live in EN 206-1
- **50-question RAGAS benchmark** (49 scored after deduplication) — 40 French / 9 English questions across 5 categories (cover, materials, reinforcement, shear, general) and 3 difficulty levels
- **4 LLMs benchmarked** simultaneously: `openai/gpt-oss-120b` (Groq), `z-ai/glm-5.2` (NVIDIA), `meta/llama-3.1-70b-instruct` (NVIDIA), `nvidia/llama-3.3-nemotron-super-49b-v1` (NVIDIA)
- **Hallucination guardrail** validated: 0% hallucination rate across all 4 models (clause-level citation check against 2,132 known nodes)
- **Out-of-scope refusal** measured: models correctly refused questions outside the EC2 corpus

### Infrastructure
- Source manifest with license tracking per document
- `.gitignore` excluding PDF, API keys, ChromaDB DB, and large generated files
- All code pushed to `work` branch on GitHub

---

## What is not done

### Parent-child (small-to-big) chunking
Currently every chunk is one fixed unit. The architecture should be restructured so **small child chunks** (individual paragraphs, single table rows) are indexed for retrieval precision, and **large parent chunks** (full clauses or sections) are used as the generation context. You search on children, generate from parents. This is the single biggest remaining architectural gap and would directly fix the remaining failures on Q6 (εcu2 strain) and Q8 (characteristic vs design values).

### Cross-encoder reranking
No reranking layer exists between retrieval and generation. After hybrid retrieval returns top-k candidates, a cross-encoder like `bge-reranker-large` or Cohere Rerank should score each (query, chunk) pair and reorder before passing to the LLM. This is specifically good at boosting short, definitionally dense chunks that embeddings underweight.

### Structured table query routing
The numeric tables (4.3N, 4.4N) are currently in ChromaDB as flat markdown text and retrieved by similarity. They should be stored as queryable structures (dict or dataframe) so that queries like "cover for XC2, structural class S4" route to a direct cell lookup (`table[S4][XC2] = 25mm`) instead of asking the LLM to read a markdown table. The pdfplumber extraction already produces clean row/column data — this is a routing layer problem, not a data problem.

### ACI 318
Only EC2 is indexed. `source_manifest.json` has the ACI 318 placeholder entry. The PDF has not been acquired, no parser exists for it, and no ACI chunks are in ChromaDB.

### Textbook parser
`textbook_parser.py` does not exist. The `Node` schema supports `chapter`, `example`, `step`, and `given`/`solution`/`result` types, but no textbook has been parsed. Textbooks are important for the cross-document retrieval hypothesis — the system should be able to retrieve a normative clause and its applied worked example together.

### Cross-document linker
`structrag/linking/cross_reference_linker.py` does not exist. The `cross_refs` field is populated per node during parsing (e.g. a paragraph mentioning "see 6.2.3" stores `["6.2.3"]`), but no code connects a textbook citation like "per EC2 §6.2.2" to the actual EC2 clause chunk in the index via `linked_node_ids`.

---

## Progress summary

| Item | Status |
|---|---|
| EC2 PDF parsed into structured nodes (2,132) | ✅ Done |
| Critical numeric tables extracted (5 tables) | ✅ Done |
| Hierarchical chunker (467 chunks) | ✅ Done |
| Flat baseline chunker (315 chunks) | ✅ Done |
| Parent-child chunking (313 parents, 1818 children) | ✅ Done |
| ChromaDB indexing with full metadata | ✅ Done |
| Hybrid BM25 + vector retrieval | ✅ Done |
| Cross-encoder reranking (ms-marco-MiniLM-L-6-v2) | ✅ Done |
| Groq RAG pipeline with domain classifier | ✅ Done |
| Gemini RAG pipeline (gemini-3.6-flash) | ✅ Done |
| NVIDIA RAG pipeline | ✅ Done |
| Multi-model comparison (4 models, RAGAS-scored) | ✅ Done |
| Hallucination guardrail (clause-level, 0% rate all models) | ✅ Done |
| 14-question evaluation harness | ✅ Done |
| Eval improvement: 3/14 → 9/14 | ✅ Done |
| 50-question RAGAS benchmark (40 FR + 9 EN) | ✅ Done |
| RAGAS scores (faithfulness, recall, precision, relevancy) | ✅ Done |
| Source manifest + .gitignore + GitHub push | ✅ Done |
| ACI 318 ingestion | ❌ Not done |
| Textbook parser | ❌ Not done |
| Cross-document linker | ❌ Not done |

---

## RAGAS Benchmark Results (49 Questions — 40 FR / 9 EN)

Full benchmark evaluated with a unified LLM judge (`meta/llama-3.1-70b-instruct`) scoring 4 RAGAS metrics per question. Results in [`structrag/evaluation/ragas_comparison.json`](structrag/evaluation/ragas_comparison.json).

| Rank | Model | Provider | Faithfulness | Ans. Relevancy | Ctx Precision | Ctx Recall | Halluc% | Refusal% |
|------|-------|----------|:---:|:---:|:---:|:---:|:---:|:---:|
| 🥇 1 | `z-ai/glm-5.2` | NVIDIA | **0.682** | **0.636** | **0.682** | **0.632** | 0% | 40% |
| 🥈 2 | `openai/gpt-oss-120b` | Groq | 0.659 | 0.466 | 0.494 | 0.352 | 0% | 100% |
| 🥉 3 | `nvidia/llama-3.3-nemotron-super-49b-v1` | NVIDIA | 0.602 | 0.364 | 0.416 | 0.336 | 0% | 100% |
| 4 | `meta/llama-3.1-70b-instruct` | NVIDIA | 0.548 | 0.425 | 0.414 | 0.302 | 0% | 100% |

> **Hallucination rate = 0%** across all 4 models. The hallucination guardrail validated all cited clause numbers against the 2,132 known EC2 nodes — no invented citations detected.

> **Refusal rate** measures correctly rejected out-of-scope questions. `gpt-oss-120b` and both NVIDIA models show 100% refusal rate because the domain classifier filtered borderline questions too aggressively. `glm-5.2` at 40% indicates it answered more boundary questions.

**Key finding:** `z-ai/glm-5.2` (NVIDIA) leads on all 4 RAGAS metrics, achieving the best context utilisation (Context Recall 0.632) and answer relevancy (0.636). Its lower refusal rate suggests it interprets out-of-scope queries more liberally, contributing to higher scored question count.

To switch model: set `GROQ_MODEL`, `GEMINI_MODEL`, or `NVIDIA_MODEL` in `.env`.

---



| Layer | Tool |
|---|---|
| PDF parsing | PyMuPDF (fitz) |
| Table extraction | pdfplumber |
| Vector store | ChromaDB (local persistent) |
| Embedding model | all-MiniLM-L6-v2 (sentence-transformers) |
| BM25 search | rank-bm25 |
| LLM (primary) | Groq — openai/gpt-oss-120b |
| LLM (alternative) | OpenAI — gpt-4o-mini |
| Evaluation | RAGAS (custom LLM judge — `meta/llama-3.1-70b-instruct`) |
| Language | Python 3.11 |

---

## Team

**Supervisor:** Afif Beji, Eng., M.Sc. — Assistant Professor, ESPRIT School of Engineering

**Interns:**
- Mohamed Khelil (3IA)
- Manef Belkadhi (3A)
- Rached Hadj Amor (4SE)

ESPRIT School of Engineering, Tunisia

---

## Safety note

This system is a research evaluation of a retrieval architecture, not a certified design tool. Outputs must not be used for actual structural design decisions without independent verification by a qualified structural engineer.
