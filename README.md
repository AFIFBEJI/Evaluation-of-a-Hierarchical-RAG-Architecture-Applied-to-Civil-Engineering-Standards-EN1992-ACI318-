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

### RAGAS evaluation
The eval harness runs questions and prints answers but does not compute RAGAS metrics. No `faithfulness`, `answer_relevance`, `context_recall`, or `context_precision` scores exist yet. The `ragas` package is in `requirements.txt` and `structrag/evaluation/` is a stub. Formalizing the evaluation with RAGAS turns "we improved 3→7" into a defensible, citable number.

### Hallucination guardrail
Nothing currently checks whether a clause number cited in the model's answer actually exists in the corpus. A lightweight post-generation check against the known clause index would flag invented citations like "per clause 7.2.4.1" if that clause doesn't exist in `ec2_2004_nf_nodes.json`. Cheap to build, important for a safety-critical domain.

### ACI 318
Only EC2 is indexed. `source_manifest.json` has the ACI 318 placeholder entry. The PDF has not been acquired, no parser exists for it, and no ACI chunks are in ChromaDB.

### Textbook parser
`textbook_parser.py` does not exist. The `Node` schema supports `chapter`, `example`, `step`, and `given`/`solution`/`result` types, but no textbook has been parsed. Textbooks are important for the cross-document retrieval hypothesis — the system should be able to retrieve a normative clause and its applied worked example together.

### Cross-document linker
`structrag/linking/cross_reference_linker.py` does not exist. The `cross_refs` field is populated per node during parsing (e.g. a paragraph mentioning "see 6.2.3" stores `["6.2.3"]`), but no code connects a textbook citation like "per EC2 §6.2.2" to the actual EC2 clause chunk in the index via `linked_node_ids`.

### QA benchmark (ground truth)
No set of questions with verified clause-level ground-truth answers exists. This is a prerequisite for a meaningful RAGAS evaluation. Questions derived from worked examples with known numerical answers are the natural source.

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
| Multi-model comparison (5 models tested) | ✅ Done |
| Hallucination guardrail (6/6 tests pass) | ✅ Done |
| 14-question evaluation harness | ✅ Done |
| Eval improvement: 3/14 → 9/14 | ✅ Done |
| RAGAS runner (8-question dataset, framework ready) | ✅ Done |
| Source manifest + .gitignore + GitHub push | ✅ Done |
| ACI 318 ingestion | ❌ Not done |
| Textbook parser | ❌ Not done |
| Cross-document linker | ❌ Not done |
| QA benchmark (ground truth, 50+ questions) | ❌ Not done |
| RAGAS scores (faithfulness, recall, precision) | ❌ Not done (needs datasets pkg fix) |

---

## Model Rankings (tested on 3 EC2 questions)

| Rank | Model | Provider | Context | Total time | Notes |
|------|-------|----------|---------|-----------|-------|
| 🥇 1 | `gemini-3.6-flash` | Gemini | 1M tokens | 17.4s | Fastest, free tier, largest context |
| 🥈 2 | `llama-3.3-70b-versatile` | Groq | 131K | 29.7s | Strong open model, free tier |
| 🥉 3 | `openai/gpt-oss-120b` | Groq | 131K | 63.3s | Good quality, free tier |
| 4 | `openai/gpt-oss-20b` | Groq | 131K | 64.6s | Fastest Groq, weaker reasoning |
| 5 | `meta/llama-3.1-70b-instruct` | NVIDIA | 128K | 89.0s | Trial credits needed |

**Recommended:** `gemini-3.6-flash` for production (1M context, fastest, free).
**Fallback:** `llama-3.3-70b-versatile` on Groq (free, no Google dependency).

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
| Evaluation | RAGAS (planned) |
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
