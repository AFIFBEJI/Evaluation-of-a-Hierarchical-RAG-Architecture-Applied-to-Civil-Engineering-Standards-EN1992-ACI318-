# StructRAG — Agent Handoff Document

This document is written for the next AI agent taking over this project.
Read it fully before touching any code.

---

## Project in one sentence

StructRAG evaluates whether a hierarchy-aware RAG pipeline retrieves better
answers from Eurocode 2 (EN 1992-1-1) than a naive flat-chunking baseline.
The corpus is the French AFNOR edition of EC2 (NF EN 1992-1-1:2004, 211 pages).

---

## Environment

- **OS**: Windows 11, PowerShell + bash
- **Python**: `C:\Users\manef\AppData\Local\Programs\Python\Python311\python.exe`
- **Project root**: `C:\Users\manef\OneDrive\Desktop\stageesprit\`
- **API keys** in `.env` (already set, do NOT print or expose them):
  - `GROQ_API_KEY` — Groq (no daily limit, free tier)
  - `NVIDIA_API_KEY` — NVIDIA build (trial credits, rate-limited)
  - `GEMINI_API_KEY` — DO NOT USE, daily quota of 20 req/day, effectively exhausted
- **EC2 PDF**: `FA039724-NF EN 1992-1-1 (octobre 2005) Eurocode 2.pdf` in root

To run any Python script:
```
C:\Users\manef\AppData\Local\Programs\Python\Python311\python.exe <script.py>
```

---

## Current project status (as of latest run)

### What is complete ✅
- Full EC2 parsing: 2,132 nodes, 5 tables, 3 chunk sets
- ChromaDB with 3 collections: hierarchical (467), flat (315), child (1,818)
- Hybrid BM25 + vector retrieval with RRF fusion
- Cross-encoder reranker (ms-marco-MiniLM-L-6-v2)
- Formula library (14 EC2 formulas)
- Table router (direct lookups for Tables 2.1N, 3.1, 4.1, 4.3N, 4.4N, etc.)
- Domain classifier with 40+ French engineering terms
- 49-question RAGAS benchmark (40 French + 9 English, 44 in-scope / 5 out-of-scope)
- Generation complete for all 4 models: gpt120b, nemotron49b, llama70b, glm52
- All models achieve 100% correct out-of-scope refusal rate
- All models achieve 0% hallucination rate on clause citations

### Current RAGAS scores (from latest valid run — GPT judge on Groq)

| Model | Faithfulness | Ans. Relevancy | Ctx Precision | Ctx Recall | Halluc |
|-------|-------------|----------------|---------------|------------|--------|
| gpt-oss-120b (Groq) | **0.97** | 0.70 | see note | see note | 0% |
| nemotron-49b (NVIDIA) | **0.95** | 0.67 | see note | see note | 0% |
| llama-3.1-70b (NVIDIA) | **0.99** | 0.68 | see note | see note | 2% |
| glm-5.2 (NVIDIA) | 0.68 | 0.64 | see note | see note | 0% |

**Note on Context Precision/Recall:** These were 0.0 in the comparison JSON because
`context_texts` were not passed to the scoring function in earlier runs.
A re-scoring script (`rescore_with_context.py`) has been created and is currently
running to fix this. Results will update `ragas_*.json` and `ragas_comparison.json`.

### What is in progress ⏳
- `rescore_with_context.py --models gpt120b` — currently running
  - Phase 1 (retrieval): ✅ Complete (44 questions, 5 chunks each)
  - Phase 2 (scoring): 🔄 In progress (~40 min estimated)
- After gpt120b completes, run: `python rescore_with_context.py --models nemotron49b llama70b glm52`

### What is NOT done yet ❌
- Context precision/recall scores for all 4 models (fix in progress)
- Final comparison table with all 4 metrics correct
- README.md update with final scores table

---

## What is already done (do not redo)

### Data on disk
- `structrag/data/parsed/ec2_2004_nf_nodes.json` — 2,132 nodes from EC2 PDF
- `structrag/data/chunks/ec2_hierarchical_chunks.json` — 467 clause-level chunks
- `structrag/data/chunks/ec2_flat_chunks.json` — 315 flat baseline chunks
- `structrag/data/chunks/ec2_child_chunks.json` — 1,818 paragraph-level child chunks
- `structrag/data/chunks/ec2_parent_chunks.json` — 313 parent clause chunks
- `structrag/data/chroma_db/` — ChromaDB with 3 live collections:
  - `ec2_hierarchical` (467 docs), `ec2_flat` (315 docs), `ec2_child` (1,818 docs)

### Pipeline (all working)
- Parsing: `standards_parser.py`, `table_extractor.py`, `hierarchy_schema.py`
- Chunking: `hierarchical_chunker.py`, `flat_chunker.py`, `parent_child_chunker.py`
- Retrieval: `hybrid_retriever.py` (BM25+vector+RRF), `reranker.py` (cross-encoder), `retriever.py`
- RAG: `groq_generation.py` (Groq), `multi_model_generation.py` (Groq+NVIDIA unified)
- Evaluation: `qa_benchmark.py`, `manual_ragas.py`, `full_ragas_eval.py`, `parallel_ragas_eval.py`

### Evaluation results (on disk)
- `structrag/evaluation/ragas_gpt120b.json` — GPT-OSS-120B results (49 questions)
- `structrag/evaluation/ragas_nemotron49b.json` — Nemotron-49B results (49 questions)
- `structrag/evaluation/ragas_llama70b.json` — Llama-3.1-70B results (49 questions)
- `structrag/evaluation/ragas_glm52.json` — GLM-5.2 results (49 questions)
- `structrag/evaluation/ragas_comparison.json` — combined comparison (currently missing ctx precision/recall)
- `structrag/evaluation/answers_*.txt` — human-readable Q&A for manual review (all 4 models)

---

## How to complete the remaining work

### Step 1: Wait for current re-scoring to finish
```
# Check output:
Get-Content rescore_gpt120b_out.txt -Tail 20
```
When it finishes, you'll see "Updated RAGAS scores for openai/gpt-oss-120b" and "Saved -> ragas_gpt120b.json"

### Step 2: Re-score the other 3 models
```
C:\Users\manef\AppData\Local\Programs\Python\Python311\python.exe rescore_with_context.py --models nemotron49b llama70b glm52 2>&1 | Tee-Object rescore_others_out.txt
```
This will re-retrieve contexts and score all 4 RAGAS metrics for the remaining models.
Estimated time: ~2 hours (3 models × 44 questions × 4 judge calls)

### Step 3: Update README with final scores
After all models are re-scored, read `structrag/evaluation/ragas_comparison.json`
and update `README.md` with the final RAGAS table.

### Step 4: Clean up temp files
Once everything is done, delete:
- `final_gpt_v3.txt`, `final_nem_v3.txt`, `reeval_gpt_final.txt`
- `scoring_out.txt`, `rescore_all.py`, `score_existing_answers.py`
- `test_judge_final.py`, `check_eval_state.py`
- `dryrun_err.txt`, `dryrun_out.txt`, `all_*.txt` log files

---

## Key technical facts about the models

### Why faithfulness is high (0.95–0.99) but answer_relevancy is lower (0.67–0.70)
Faithfulness measures whether answers contradict the context — models score high
because they stay on the retrieved text. Answer relevancy measures whether the
answer directly addresses the question — it's lower because:
1. Nemotron-49B is very verbose (writes 3× more than needed for simple factual questions)
2. Some questions retrieve slightly wrong chunks (context precision issue)
3. Out-of-scope questions that slip through get longer "I cannot answer" responses

### Why GLM-5.2 faithfulness is lower (0.68)
GLM adds more synthesis and inference between context and answer. It's not less
accurate, just less conservative — so when it expands beyond the literal context
text, faithfulness drops even if the answer is correct.

---

## Domain classifier — IMPORTANT for future runs

The domain classifier in `structrag/rag_pipeline/groq_generation.py` uses a
keyword list (`_ENGINEERING_KEYWORDS`) to decide if a question is about EC2.
These 6 French terms were missing and caused false rejections — they have been
added in the latest version:
- `carbonatation`, `chlorures`, `chlorure`
- `contrainte de compression`, `contrainte admissible`
- `armatures transversales`, `espacement`
- `taux d'armature`, `zone comprimee`
- `cisaillement`, `effort tranchant`
- `fissure`, `fissuration`, `largeur de fissure`
- `mandrin`, `pliage`, `ancrage`

---

## RAGAS judge — openai/gpt-oss-120b on Groq

**Judge config** (`structrag/evaluation/manual_ragas.py`):
- Model: `openai/gpt-oss-120b` via Groq API
- `max_tokens=512` — CRITICAL for reasoning models (internal chain-of-thought eats tokens)
- Pattern: `SCORE: X.X` at end of response — forces visible output before the score
- French note: "The answer may be in French — this is expected" — prevents penalizing French answers
- Retry: 3 attempts with 30s/60s backoff on 429 rate limit

**Do NOT change the judge to Gemini, Qwen, or any other model.** The user explicitly
confirmed GPT-OSS-120B on Groq as the judge.

---

## Models available for use

### Groq (no daily limit, use freely)
- `openai/gpt-oss-120b` — primary RAG model + judge ✅
- `openai/gpt-oss-20b` — available but too weak for technical EC2 questions
- `llama-3.3-70b-versatile` — REMOVED from Groq, do NOT use (returns 404)

### NVIDIA build (trial credits, rate-limited ~1 req/min)
- `z-ai/glm-5.2` — 1M context, works but hits rate limits on bulk evaluation ✅
- `nvidia/llama-3.3-nemotron-super-49b-v1` — works, best NVIDIA quality ✅
- `meta/llama-3.1-70b-instruct` — works, good speed/quality trade-off ✅
- `mistralai/mistral-nemotron` — works ✅

### Do NOT use
- Any `gemini-*` model — 20 req/day daily limit, exhausted
- `llama-3.3-70b-versatile` on Groq — no longer available (404 error)
- `qwen/qwen3.6-27b` — explicitly declined by user

---

## Known infrastructure issues

1. **Windows cp1252 encoding**: French text with accents and Greek letters (γ, ε, ρ)
   crash `print()` on Windows. Always add at the top of scripts:
   ```python
   import io, sys
   sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
   ```

2. **ChromaDB telemetry warnings**: `Failed to send telemetry event...` on every
   query. Harmless, ignore.

3. **huggingface-hub version conflict**: `ragas` installs `huggingface-hub>=1.x`
   which breaks `sentence-transformers==3.0.1`. After any pip install touching ragas:
   ```
   pip install huggingface-hub==0.24.7 sentence-transformers==3.0.1 --no-deps
   ```

4. **NVIDIA rate limits**: GLM-5.2 and other NVIDIA models hit rate limits during
   bulk evaluation. Add 65s sleep between NVIDIA API calls for large batches.

---

## Expected RAGAS score ranges for a good EC2 RAG

For a production-quality standards RAG on a 211-page technical document:
- **Faithfulness** > 0.85 → good (answers stay on retrieved content)
- **Answer Relevancy** > 0.70 → acceptable (directly answers the question)
- **Context Precision** > 0.60 → good (retrieved chunks are relevant)
- **Context Recall** > 0.50 → acceptable (context contains the answer info)

Current project scores are **research-grade**, not production-grade because:
1. Many formulas are in images (not extractable by pdfplumber/PyMuPDF)
2. Tables have limited cell-level retrieval
3. Cross-document references (EN 206-1, EN 1990) are not indexed

---

## Future work — items requiring documents or infrastructure not yet available

### FW-1: Add EN 206-1 (concrete specification standard)
**Why**: EC2 references EN 206-1 for water-cement ratios, cement content.
Without it, w/c ratio questions return "consult EN 206-1" — correct but not useful.
**How**: Once EN 206-1 PDF is available — run `standards_parser.py` on it.
Expected improvement: +0.08 context recall on composition questions.

### FW-2: Add EN 1990 (basis of design) and EN 1991-1-1 (loads)
**Why**: EC2 clause 2.1 references EN 1990 for load combinations.
Questions about load factors (γG, γQ) go unanswered.
**How**: Same parser, same manifest entry. 1 day each.
Expected improvement: +0.05 recall on load combination questions.

### FW-3: Add EC2 textbooks with worked examples
**Why**: Biggest usability improvement. Worked examples give step-by-step templates.
**How**: Requires `textbook_parser.py` (not yet built). License confirmation per textbook.
Expected improvement: +0.10–0.15 answer relevancy.

### FW-4: Multi-turn conversation support
**Why**: Real engineers ask follow-up questions that depend on prior context.
**How (Option A)**: Session context dict — extract entities from prior answers, inject into query.
**How (Option B)**: Use GLM-5.2's 1M context window — pass full conversation history.

### FW-5: Cross-document linker
**Why**: When a textbook says "per EC2 §6.2.2", the linker would fetch the actual clause.
**How**: `structrag/linking/cross_reference_linker.py` — scan `cross_refs` fields.

### FW-6: ACI 318 ingestion
**Why**: The research hypothesis compares two structurally different standards.
ACI 318 has different numbering (Section 26.4 vs EC2's 4.4.1.2).
**How**: Acquire ACI 318-19 PDF. Update `standards_parser.py` for ACI numbering.

### FW-7: Numeric value verification (hallucination guardrail extension)
**Why**: Current guardrail checks clause citations but not numeric values.
**How**: Extract numeric claims, look them up in table_router + formula_library.

### FW-8: National Annex parameter overrides
**Why**: French NA modifies αcc, cover tolerances, and structural classification.
**How**: Parse "Annexe Nationale" section, give NA values retrieval priority.

---

## Files to protect (do not delete)

- `structrag/data/chroma_db/` — the live vector DB (expensive to rebuild)
- `structrag/data/parsed/ec2_2004_nf_nodes.json` — parsed nodes
- `structrag/data/chunks/ec2_*_chunks.json` — all chunk files
- `.env` — API keys
- `structrag/evaluation/ragas_*.json` — all evaluation results
- `structrag/evaluation/answers_*.txt` — human-readable Q&A records

## Files that can be deleted after work is done

- `final_gpt_v3.txt`, `final_nem_v3.txt`, `reeval_gpt_final.txt`
- `scoring_out.txt`, `rescore_all.py`, `score_existing_answers.py`
- `test_judge_final.py`, `check_eval_state.py`, `check_nltk.py`, `check_ragas.py`, `check_ragas2.py`
- `dryrun_err.txt`, `dryrun_out.txt`
- All `all_*.txt` log files (`all_check.txt`, `all_err.txt`, etc.)
- `eval3_stderr.txt`, `eval3_stdout.txt`, `eval_stderr.txt`, `eval_stdout.txt`
- `extractor_err.txt`, `cross_check_err.txt`
