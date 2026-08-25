"""
structrag/evaluation/full_ragas_eval.py
-----------------------------------------
Full RAGAS evaluation using the 50-question QA benchmark.

Tests the top 3 models:
  1. gemini-3.6-flash   (Gemini, 1M ctx, fastest)
  2. z-ai/glm-5.2       (NVIDIA build, 1M ctx)
  3. openai/gpt-oss-120b (Groq, 131K ctx, best quality in earlier tests)

For each model, computes:
  - faithfulness       : answer grounded in retrieved context
  - answer_relevancy   : answer addresses the question
  - context_precision  : retrieved chunks are relevant
  - context_recall     : context contains ground-truth information

Also reports:
  - per-question hallucination flag
  - out-of-scope rejection rate
  - correct refusal rate for out-of-scope questions

Usage
-----
    python structrag/evaluation/full_ragas_eval.py
    python structrag/evaluation/full_ragas_eval.py --models groq gemini
    python structrag/evaluation/full_ragas_eval.py --quick   # first 10 Qs only
"""

from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env
for _l in (PROJECT_ROOT / ".env").read_text(encoding="utf-8").splitlines():
    _l = _l.strip()
    if _l and not _l.startswith("#") and "=" in _l:
        _k, _v = _l.split("=", 1); os.environ[_k.strip()] = _v.strip()

# Only redirect stdout when run directly, not when imported
if __name__ == "__main__":
    import io
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from structrag.evaluation.qa_benchmark import QA_BENCHMARK
from structrag.retrieval.hybrid_retriever import retrieve_hybrid
from structrag.retrieval.retriever import retrieve_child
from structrag.chunking.parent_child_chunker import resolve_parents
from structrag.retrieval.reranker import rerank_with_fallback
from structrag.rag_pipeline.multi_model_generation import generate_multi
from structrag.evaluation.hallucination_guard import check_hallucination
from structrag.rag_pipeline.groq_generation import is_engineering_query
from structrag.retrieval.table_router import enrich_with_table_router
from structrag.retrieval.formula_library import get_formula_for_query, make_formula_chunk
from structrag.retrieval.symbol_dictionary import enrich_with_symbols   # G2
from structrag.retrieval.query_rewriter import rewrite_query             # G3

def enrich_context(query: str, chunks: list[dict], source_id: str = "ec2_2004_nf") -> list[dict]:
    """
    Enrich retrieved chunks with:
    1. Table router — direct cell lookups
    2. Symbol dictionary (G2) — inject exact symbol definitions
    3. Formula library — inject full formula blocks as fallback
    """
    # Table router
    chunks = enrich_with_table_router(query, chunks, source_id)

    # G2: symbol dictionary — precise, non-contaminating
    chunks = enrich_with_symbols(query, chunks, source_id)

    # Formula library — append (symbol dict takes priority now)
    formula_text = get_formula_for_query(query)
    if formula_text:
        import re as _re
        clause_match = _re.search(r'\b(\d+\.\d+(?:\.\d+)*)\b', formula_text)
        clause = clause_match.group(1) if clause_match else "unknown"
        formula_chunk = make_formula_chunk(formula_text, clause, source_id)
        existing_ids = {c["chunk_id"] for c in chunks}
        if formula_chunk["chunk_id"] not in existing_ids:
            chunks = chunks + [formula_chunk]  # append, not prepend

    return chunks

# ---------------------------------------------------------------------------
# Models to evaluate
# ---------------------------------------------------------------------------
# Models to evaluate — no Gemini, Groq + NVIDIA only
MODELS = {
    "gpt120b":       ("openai/gpt-oss-120b",                     "groq"),
    "nemotron49b":   ("nvidia/llama-3.3-nemotron-super-49b-v1",  "nvidia"),
    "nemotron49bv2": ("nvidia/llama-3.3-nemotron-super-49b-v1.5","nvidia"),
    "llama70b":      ("meta/llama-3.3-70b-instruct",              "nvidia"),
}


def _detect_provider_for_key(model_id: str) -> str:
    m = model_id.lower()
    if m.startswith("z-ai/") or m.startswith("meta/") or m.startswith("nvidia/"):
        return "nvidia"
    if m.startswith("gemini"): return "gemini"
    return "groq"

PARENT_PATH = PROJECT_ROOT / "structrag" / "data" / "chunks" / "ec2_parent_chunks.json"
SOURCE_ID   = "ec2_2004_nf"
N_RETRIEVE  = 20
N_RERANK    = 8

# ---------------------------------------------------------------------------
# Per-question pipeline
# ---------------------------------------------------------------------------

def run_question(question: str, model_id: str, provider: str, in_scope: bool) -> dict:
    """Run one question through the full pipeline. Returns a result dict."""
    t0 = time.time()

    # Domain check
    domain_ok = is_engineering_query(question)

    if not in_scope:
        # For out-of-scope: check that the model correctly refuses
        if not domain_ok:
            return {
                "answer": "[CORRECTLY REJECTED by domain classifier]",
                "context_texts": [],
                "correctly_refused": True,
                "elapsed_s": round(time.time() - t0, 2),
                "hallucination_clean": True,
                "hallucinated_clauses": [],
            }
        # Domain classifier let it through — still try and record
    
    if not domain_ok and in_scope:
        # False negative — domain classifier wrongly rejected an in-scope question
        return {
            "answer": "[WRONGLY REJECTED by domain classifier]",
            "context_texts": [],
            "correctly_refused": False,
            "elapsed_s": round(time.time() - t0, 2),
            "hallucination_clean": True,
            "hallucinated_clauses": [],
        }

    # Retrieval: G3 query rewriting (LLM-powered) + parent-child + hybrid supplement
    retrieval_query = rewrite_query(question, use_llm=True)   # G3: LLM translation enabled

    child_results = retrieve_child(retrieval_query, n_results=N_RETRIEVE, source_id=SOURCE_ID)
    relevant_children = [c for c in child_results if c.get("score", 999) < 0.8]
    if relevant_children and PARENT_PATH.exists():
        context_chunks = resolve_parents(relevant_children, str(PARENT_PATH))
    else:
        context_chunks = []

    hybrid = retrieve_hybrid(retrieval_query, n_results=N_RETRIEVE, source_id=SOURCE_ID)
    existing_ids = {c["chunk_id"] for c in context_chunks}
    for h in hybrid:
        if h["chunk_id"] not in existing_ids:
            context_chunks.append(h)
            existing_ids.add(h["chunk_id"])

    # Rerank on ORIGINAL question (not the rewritten query)
    context_chunks = rerank_with_fallback(question, context_chunks, top_n=N_RERANK)
    context_texts  = [c["text"] for c in context_chunks]

    # ── Table router + Symbol dictionary (G2) + Formula library ──────────
    context_chunks = enrich_context(question, context_chunks, SOURCE_ID)

    if not context_chunks and not in_scope:
        return {
            "answer": "[No relevant context — out of scope]",
            "context_texts": [],
            "correctly_refused": True,
            "elapsed_s": round(time.time() - t0, 2),
            "hallucination_clean": True,
            "hallucinated_clauses": [],
        }

    # Generation
    try:
        result = generate_multi(
            question=question, chunks=context_chunks,
            model=model_id, provider=provider,
            temperature=0.01,   # prevent NVIDIA response caching on consecutive similar questions
        )
        answer = result.answer
        # Light throttle between questions to respect free tier RPM/TPM limits
        import time as _time
        _time.sleep(2)
    except Exception as e:
        answer = f"[ERROR: {e}]"

    # Hallucination check
    guard = check_hallucination(answer, source_id=SOURCE_ID)

    # For out-of-scope: check if the answer is a refusal
    correctly_refused = False
    if not in_scope:
        refusal_phrases = [
            "not contain enough", "out of scope", "not covered",
            "consult", "eurocode 8", "eurocode 1", "EN 206", "en 1998",
            "does not cover", "not addressed", "separate standard",
        ]
        correctly_refused = any(p.lower() in answer.lower() for p in refusal_phrases)

    return {
        "answer":               answer,
        "context_texts":        context_texts,
        "correctly_refused":    correctly_refused,
        "elapsed_s":            round(time.time() - t0, 2),
        "hallucination_clean":  guard.is_clean,
        "hallucinated_clauses": guard.hallucinated,
    }


# ---------------------------------------------------------------------------
# Evaluate one model
# ---------------------------------------------------------------------------

def evaluate_model(
    model_key: str,
    questions: list[dict],
    output_dir: Path,
) -> dict:
    model_id, provider = MODELS[model_key]
    print(f"\n{'='*65}")
    print(f"  Evaluating: {model_id} ({provider})")
    print(f"  Questions : {len(questions)}")
    print(f"{'='*65}")

    all_questions   : list[str]        = []
    all_answers     : list[str]        = []
    all_contexts    : list[list[str]]  = []
    all_ground_truths: list[str]       = []
    per_q_results   : list[dict]       = []

    in_scope_total  = sum(1 for q in questions if q["in_scope"])
    out_scope_total = len(questions) - in_scope_total
    correct_refusals = 0
    hallucination_count = 0

    for i, item in enumerate(questions, 1):
        q  = item["question"]
        gt = item["ground_truth"]

        print(f"  Q{i:02d}/{len(questions)}: {q[:70]}...", end=" ", flush=True)
        t0 = time.time()

        res = run_question(q, model_id, provider, item["in_scope"])

        elapsed = res["elapsed_s"]
        print(f"({elapsed}s)", flush=True)

        if not item["in_scope"]:
            if res["correctly_refused"]:
                correct_refusals += 1
            status = "REFUSED" if res["correctly_refused"] else "NOT REFUSED"
            print(f"         Out-of-scope -> {status}")
        else:
            if not res["hallucination_clean"]:
                hallucination_count += 1
                print(f"         HALLUCINATION: {res['hallucinated_clauses']}")

        all_questions.append(q)
        all_answers.append(res["answer"])
        all_contexts.append(res["context_texts"] or ["[no context]"])
        all_ground_truths.append(gt)

        per_q_results.append({
            "question":          q,
            "ground_truth":      gt,
            "gt_clauses":        item["ground_truth_clauses"],
            "category":          item["category"],
            "difficulty":        item["difficulty"],
            "in_scope":          item["in_scope"],
            "lang":              item.get("lang", "en"),
            "answer":            res["answer"],
            "n_context_chunks":  len(res["context_texts"]),
            "context_texts":     [c[:500] for c in res["context_texts"][:5]],  # store for RAGAS scoring
            "elapsed_s":         elapsed,
            "hallucination_clean": res["hallucination_clean"],
            "hallucinated_clauses": res["hallucinated_clauses"],
            "correctly_refused": res.get("correctly_refused", False),
        })

    # ── Write human-readable answer record per model ──────────────────────
    readable_path = output_dir / f"answers_{model_key}.txt"
    with open(readable_path, "w", encoding="utf-8") as rf:
        rf.write(f"StructRAG — Answer Record\n")
        rf.write(f"Model   : {model_id}\n")
        rf.write(f"Provider: {provider}\n")
        rf.write(f"Date    : {time.strftime('%Y-%m-%d %H:%M')}\n")
        rf.write(f"Questions: {len(questions)}\n")
        rf.write("=" * 70 + "\n\n")
        for i, qr in enumerate(per_q_results, 1):
            rf.write(f"Q{i:02d}. [{qr['lang'].upper()} | {qr['category']} | {qr['difficulty']}]\n")
            rf.write(f"Question     : {qr['question']}\n")
            rf.write(f"Ground truth : {qr['ground_truth']}\n")
            rf.write(f"GT clauses   : {', '.join(qr['gt_clauses']) or 'N/A'}\n")
            rf.write(f"Model answer : {qr['answer']}\n")
            rf.write(f"Hallucination: {'CLEAN' if qr['hallucination_clean'] else 'WARNING: ' + str(qr['hallucinated_clauses'])}\n")
            if not qr["in_scope"]:
                rf.write(f"Refusal      : {'CORRECT' if qr['correctly_refused'] else 'MISSED'}\n")
            rf.write(f"Time         : {qr['elapsed_s']}s\n")
            rf.write("-" * 70 + "\n\n")
    print(f"  Human-readable answers -> {readable_path.name}")

    # RAGAS scoring — using manual Gemini judge (avoids langchain version conflicts)
    ragas_scores = {}
    print("\n  Computing RAGAS scores (Gemini judge)...")
    try:
        from structrag.evaluation.manual_ragas import compute_ragas_scores

        # Only score in-scope questions
        in_scope_indices = [i for i, q in enumerate(questions) if q["in_scope"]]
        qs  = [all_questions[i]    for i in in_scope_indices]
        ans = [all_answers[i]      for i in in_scope_indices]
        ctx = [all_contexts[i]     for i in in_scope_indices]
        gts = [all_ground_truths[i] for i in in_scope_indices]

        ragas_result = compute_ragas_scores(qs, ans, ctx, gts, verbose=True)
        ragas_scores = {k: v for k, v in ragas_result.items() if k != "per_question"}

        print(f"\n  RAGAS Scores (in-scope questions only, n={len(qs)}):")
        for m, s in ragas_scores.items():
            print(f"    {m:22s}: {s:.4f}")

        # Attach per-question RAGAS to per_q_results
        for j, idx in enumerate(in_scope_indices):
            if j < len(ragas_result.get("per_question", [])):
                per_q_results[idx].update(ragas_result["per_question"][j])

    except Exception as e:
        print(f"  RAGAS scoring failed: {e}")
        ragas_scores = {"error": str(e)}

    # Summary stats
    refusal_rate = correct_refusals / out_scope_total if out_scope_total else None
    halluc_rate  = hallucination_count / in_scope_total if in_scope_total else None

    print(f"\n  Summary:")
    print(f"    In-scope questions   : {in_scope_total}")
    print(f"    Out-of-scope         : {out_scope_total}")
    print(f"    Correct refusals     : {correct_refusals}/{out_scope_total} ({refusal_rate:.0%})" if refusal_rate is not None else "")
    print(f"    Hallucinations       : {hallucination_count}/{in_scope_total} ({halluc_rate:.0%})" if halluc_rate is not None else "")

    output = {
        "model_id":          model_id,
        "provider":          provider,
        "n_questions":       len(questions),
        "n_in_scope":        in_scope_total,
        "n_out_of_scope":    out_scope_total,
        "correct_refusals":  correct_refusals,
        "refusal_rate":      round(refusal_rate, 4) if refusal_rate is not None else None,
        "hallucination_count": hallucination_count,
        "hallucination_rate":  round(halluc_rate, 4) if halluc_rate is not None else None,
        "ragas_scores":      ragas_scores,
        "per_question":      per_q_results,
    }

    # Save
    out_file = output_dir / f"ragas_{model_key}.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved -> {out_file.name}")

    return output


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full RAGAS evaluation")
    parser.add_argument("--models", nargs="+",
                        choices=list(MODELS.keys()) + ["all"],
                        default=["gpt120b", "glm52", "mistral", "nemotron49b"],
                        help="Model preset keys to evaluate")
    parser.add_argument("--model-id", type=str, default=None,
                        help="Override model ID (used by parallel runner)")
    parser.add_argument("--provider", type=str, default=None,
                        help="Override provider (used by parallel runner)")
    parser.add_argument("--quick", action="store_true",
                        help="Use first 10 questions only (for testing)")
    parser.add_argument("--category", type=str, default=None,
                        help="Filter to one category (e.g. cover, materials)")
    args = parser.parse_args()

    questions = QA_BENCHMARK
    if args.category:
        questions = [q for q in questions if q["category"] == args.category]
    if args.quick:
        questions = questions[:10]

    print(f"\nStructRAG Full RAGAS Evaluation")

    output_dir = PROJECT_ROOT / "structrag" / "evaluation"
    all_results = {}

    # If called with explicit model-id (from parallel runner), evaluate just that one
    if args.model_id:
        provider = args.provider or _detect_provider_for_key(args.model_id)
        # Find or create a key for this model
        key = next((k for k, (m, p) in MODELS.items() if m == args.model_id), "custom")
        MODELS[key] = (args.model_id, provider)
        model_keys = [key]
    else:
        model_keys = args.models if args.models != ["all"] else list(MODELS.keys())

    print(f"Models    : {model_keys}")
    print(f"Questions : {len(questions)}")
    print(f"In-scope  : {sum(1 for q in questions if q['in_scope'])}")
    print(f"Out-scope : {sum(1 for q in questions if not q['in_scope'])}")

    for model_key in model_keys:
        if model_key not in MODELS:
            print(f"Unknown model key: {model_key}, skipping")
            continue
        result = evaluate_model(model_key, questions, output_dir)
        all_results[model_key] = result

    # Final comparison table
    print(f"\n{'='*65}")
    print("  FINAL COMPARISON")
    print(f"{'='*65}")
    header = f"  {'Model':<20} {'Faith':>8} {'AnswRel':>8} {'CtxPrec':>8} {'CtxRec':>8} {'Halluc':>8} {'Refusal':>8}"
    print(header)
    print("  " + "-" * 63)
    for key, res in all_results.items():
        s = res.get("ragas_scores", {})
        model_id = res.get("model_id") or key
        if "error" not in s and s:
            print(f"  {model_id[:20]:<20} "
                  f"{s.get('faithfulness', 0) or 0:>8.4f} "
                  f"{s.get('answer_relevancy', 0) or 0:>8.4f} "
                  f"{s.get('context_precision', 0) or 0:>8.4f} "
                  f"{s.get('context_recall', 0) or 0:>8.4f} "
                  f"{(res.get('hallucination_rate') or 0):>8.2%} "
                  f"{(res.get('refusal_rate') or 0):>8.2%}")
        else:
            err = s.get('error', 'unknown')[:40] if isinstance(s, dict) else str(s)[:40]
            print(f"  {model_id[:20]:<20} [RAGAS failed: {err}]")

    # Save combined
    combined_path = output_dir / "ragas_comparison.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nFull results -> {combined_path}")
