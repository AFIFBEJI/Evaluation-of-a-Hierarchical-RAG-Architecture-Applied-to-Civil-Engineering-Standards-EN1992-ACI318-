"""
tests/test_multi_model.py
--------------------------
Tests all configured LLM backends on the same 3 questions and ranks them.
Run after all API keys are set in .env.

Output -> tests/multi_model_results.txt
"""
import sys, os, time, json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

# Load .env
for line in (BASE / ".env").read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1); os.environ[k.strip()] = v.strip()

out = open(BASE / "tests" / "multi_model_results.txt", "w", encoding="utf-8")

def w(s=""):
    try: print(str(s))
    except: pass
    out.write(str(s) + "\n")

DIVIDER = "=" * 65
SUB = "-" * 65

# Questions that specifically test different retrieval depths
TEST_QUESTIONS = [
    "What partial safety factor applies to concrete in persistent design situations?",
    "What minimum cover is required for structural class S4 and exposure class XC2?",
    "How do I calculate nominal cover from minimum cover?",
]

# Models to test — (model_id, provider, description)
MODELS_TO_TEST = [
    ("openai/gpt-oss-120b",        "groq",   "Groq GPT-OSS 120B — 131K ctx"),
    ("llama-3.3-70b-versatile",    "groq",   "Groq Llama 3.3 70B — 131K ctx"),
    ("gemini-2.5-flash",           "gemini", "Gemini 2.5 Flash — 1M ctx"),
    ("meta/llama-3.1-70b-instruct","nvidia", "NVIDIA Llama 3.1 70B — 128K ctx"),
    ("openai/gpt-oss-20b",         "groq",   "Groq GPT-OSS 20B — 131K ctx (fastest)"),
]

w(DIVIDER)
w("  StructRAG Multi-Model Comparison")
w(DIVIDER)

try:
    from structrag.retrieval.hybrid_retriever import retrieve_hybrid
    from structrag.chunking.parent_child_chunker import resolve_parents
    from structrag.retrieval.reranker import rerank_with_fallback
    from structrag.rag_pipeline.multi_model_generation import generate_multi
    from structrag.evaluation.hallucination_guard import check_hallucination

    parent_path = BASE / "structrag/data/chunks/ec2_parent_chunks.json"

    # Pre-retrieve contexts once (same for all models — fair comparison)
    w("\nPre-retrieving contexts for all questions...")
    all_contexts = []
    for q in TEST_QUESTIONS:
        from structrag.retrieval.retriever import retrieve_child
        children = retrieve_child(q, n_results=16, source_id="ec2_2004_nf")
        if children and parent_path.exists():
            chunks = resolve_parents(children, str(parent_path))
        else:
            chunks = retrieve_hybrid(q, n_results=8, source_id="ec2_2004_nf")
        chunks = rerank_with_fallback(q, chunks, top_n=5)
        all_contexts.append(chunks)
        w(f"  Q: {q[:60]}... -> {len(chunks)} context chunks")

    # Test each model
    results_summary = []

    for model_id, provider, description in MODELS_TO_TEST:
        w(f"\n{DIVIDER}")
        w(f"  Model: {description}")
        w(DIVIDER)

        model_results = {"model": model_id, "provider": provider,
                         "description": description, "questions": []}
        total_time = 0
        hallucination_count = 0

        for q, chunks in zip(TEST_QUESTIONS, all_contexts):
            w(f"\n  Q: {q}")
            w(SUB)

            t0 = time.time()
            try:
                result = generate_multi(
                    question=q, chunks=chunks,
                    model=model_id, provider=provider,
                )
                answer = result.answer
                elapsed = round(time.time() - t0, 2)
                total_time += elapsed

                guard = check_hallucination(answer)
                if not guard.is_clean:
                    hallucination_count += 1

                w(f"  Answer ({elapsed}s):")
                for line in answer.splitlines()[:8]:
                    w(f"    {line}")
                w(f"  Hallucination check: {guard.summary()}")

                model_results["questions"].append({
                    "question": q, "answer": answer,
                    "elapsed_s": elapsed,
                    "hallucination_clean": guard.is_clean,
                    "hallucinated_clauses": guard.hallucinated,
                })

            except Exception as e:
                elapsed = round(time.time() - t0, 2)
                w(f"  ERROR ({elapsed}s): {e}")
                model_results["questions"].append({
                    "question": q, "answer": f"ERROR: {e}",
                    "elapsed_s": elapsed, "hallucination_clean": True,
                    "hallucinated_clauses": [],
                })

        model_results["total_time_s"] = round(total_time, 2)
        model_results["hallucination_count"] = hallucination_count
        results_summary.append(model_results)

    # ── Final ranking ──────────────────────────────────────────────────────
    w(f"\n{DIVIDER}")
    w("  RANKING SUMMARY")
    w(DIVIDER)
    w(f"  {'Model':<40} {'Total time':>12} {'Hallucinations':>15}")
    w(SUB)

    # Sort: fewer hallucinations first, then faster
    ranked = sorted(results_summary,
                    key=lambda x: (x["hallucination_count"], x["total_time_s"]))
    for rank, r in enumerate(ranked, 1):
        w(f"  {rank}. {r['description']:<38} "
          f"{r['total_time_s']:>10.1f}s  "
          f"{r['hallucination_count']:>12}")

    # Save full results
    results_file = BASE / "tests" / "multi_model_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, ensure_ascii=False, indent=2)
    w(f"\nFull results saved -> {results_file.name}")

except Exception as e:
    import traceback
    w(f"FATAL ERROR: {e}")
    w(traceback.format_exc())

out.close()
print("Done -> tests/multi_model_results.txt")
