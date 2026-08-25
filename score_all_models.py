"""
score_all_models.py  (v2 — single combined prompt per model×question)
----------------------------------------------------------------------
Scores all 4 metrics in ONE judge call per model×question.
This cuts API calls from 5 → 1, making the full run ~5× faster.

Progress is saved after every scored question — safe to restart.

Run: python score_all_models.py
"""
from __future__ import annotations
import io, sys, json, os, time, re
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

for _l in (PROJECT_ROOT / ".env").read_text(encoding="utf-8").splitlines():
    _l = _l.strip()
    if _l and not _l.startswith("#") and "=" in _l:
        _k, _v = _l.split("=", 1); os.environ[_k.strip()] = _v.strip()

EVAL_DIR      = PROJECT_ROOT / "structrag" / "evaluation"
SOURCE_ID     = "ec2_2004_nf"
N_RETRIEVE    = 20
N_RERANK      = 8
PARENT_PATH   = PROJECT_ROOT / "structrag" / "data" / "chunks" / "ec2_parent_chunks.json"
JUDGE_MODEL   = "meta/llama-3.3-70b-instruct"   # NVIDIA — separate from Groq RAG model
PROGRESS_FILE = PROJECT_ROOT / "scoring_progress.json"
SLEEP_OK      = 3    # seconds between successful calls
SLEEP_BASE    = 30   # base wait on rate limit

MODELS = {
    "gpt120b":     "ragas_gpt120b.json",
    "nemotron49b": "ragas_nemotron49b.json",
    "llama70b":    "ragas_llama70b.json",
    "mistral":     "ragas_mistral.json",      # replaces glm52 (EOL 2026-08-21)
}

ERROR_PREFIXES = (
    "[ERROR", "[Groq rate", "[NVIDIA rate", "[Gemini",
    "[WRONGLY REJECTED", "[CORRECTLY REJECTED", "[No match",
    "[No chunk", "[No relev", "[Groq max",
)

# ── Judge: single combined call ────────────────────────────────────────────

def call_judge(prompt: str) -> str:
    """Call llama-3.3-70b on NVIDIA as judge — doesn't compete with Groq rate limits."""
    api_key = os.environ.get("NVIDIA_API_KEY", "")
    for attempt in range(7):
        try:
            from openai import OpenAI
            resp = OpenAI(
                base_url="https://integrate.api.nvidia.com/v1",
                api_key=api_key, timeout=60.0,
            ).chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                temperature=0.0,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            err = str(e)
            if "429" in err or "rate_limit" in err.lower():
                wait = min(SLEEP_BASE * (attempt + 1), 210)
                print(f" [RL {wait}s]", end="", flush=True)
                time.sleep(wait)
            else:
                print(f" [ERR {err[:30]}]", end="", flush=True)
                return ""
    return ""

def score_one(question: str, answer: str, contexts: list[str], ground_truth: str) -> dict:
    """One judge call returns all 4 RAGAS scores."""
    ctx_str = " | ".join(c[:120] for c in contexts[:3]) or "[no context]"

    prompt = (
        f"You are evaluating a RAG system answer about Eurocode 2 (French civil engineering standard).\n"
        f"The answer may be in French — this is normal and correct.\n\n"
        f"QUESTION: {question[:200]}\n\n"
        f"CONTEXT (retrieved passages): {ctx_str[:400]}\n\n"
        f"GROUND TRUTH: {ground_truth[:200]}\n\n"
        f"ANSWER: {answer[:300]}\n\n"
        f"Score each metric from 0.0 to 1.0:\n"
        f"1. FAITHFULNESS: Is the answer consistent with the context? (0=contradicts, 1=fully grounded)\n"
        f"2. RELEVANCY: Does the answer directly address the question? (0=irrelevant, 1=fully answers)\n"
        f"3. CONTEXT_PRECISION: Is the retrieved context relevant to the question? (0=irrelevant, 1=all relevant)\n"
        f"4. CONTEXT_RECALL: Does the context contain info needed for the ground truth? (0=missing, 1=fully covered)\n\n"
        f"Respond ONLY with these 4 lines (nothing else):\n"
        f"FAITHFULNESS: X.X\n"
        f"RELEVANCY: X.X\n"
        f"CONTEXT_PRECISION: X.X\n"
        f"CONTEXT_RECALL: X.X"
    )

    response = call_judge(prompt)
    time.sleep(SLEEP_OK)

    def parse(label: str) -> float:
        m = re.search(rf'{label}:\s*(0\.\d+|1\.0|0\.0|[01])', response, re.IGNORECASE)
        if m:
            return min(1.0, max(0.0, float(m.group(1))))
        # fallback: any decimal number on the line
        lines = [l for l in response.splitlines() if label.lower() in l.lower()]
        if lines:
            nums = re.findall(r'0\.\d+|1\.0|0\.0', lines[0])
            if nums: return float(nums[0])
        return 0.5

    if not response:
        return {"f": 0.5, "ar": 0.5, "cp": 0.5, "cr": 0.5}

    return {
        "f":  parse("FAITHFULNESS"),
        "ar": parse("RELEVANCY"),
        "cp": parse("CONTEXT_PRECISION"),
        "cr": parse("CONTEXT_RECALL"),
    }

# ── Retrieval ──────────────────────────────────────────────────────────────

def retrieve(question: str) -> list[str]:
    try:
        from structrag.retrieval.hybrid_retriever import retrieve_hybrid
        from structrag.retrieval.retriever import retrieve_child
        from structrag.chunking.parent_child_chunker import resolve_parents
        from structrag.retrieval.reranker import rerank_with_fallback
        from structrag.retrieval.table_router import enrich_with_table_router
        from structrag.retrieval.formula_library import get_formula_for_query, make_formula_chunk

        children = retrieve_child(question, n_results=N_RETRIEVE, source_id=SOURCE_ID)
        relevant = [c for c in children if c.get("score", 999) < 0.8]
        chunks   = resolve_parents(relevant, str(PARENT_PATH)) if relevant and PARENT_PATH.exists() else []

        hybrid = retrieve_hybrid(question, n_results=N_RETRIEVE, source_id=SOURCE_ID)
        seen   = {c["chunk_id"] for c in chunks}
        for h in hybrid:
            if h["chunk_id"] not in seen:
                chunks.append(h); seen.add(h["chunk_id"])

        chunks = rerank_with_fallback(question, chunks, top_n=N_RERANK)
        chunks = enrich_with_table_router(question, chunks, SOURCE_ID)

        formula = get_formula_for_query(question)
        if formula:
            import re as _re
            m = _re.search(r'\b(\d+\.\d+(?:\.\d+)*)\b', formula)
            fc = make_formula_chunk(formula, m.group(1) if m else "unknown", SOURCE_ID)
            if fc["chunk_id"] not in {c["chunk_id"] for c in chunks}:
                chunks = [fc] + chunks

        return [c["text"] for c in chunks[:5]]
    except Exception as e:
        print(f"\n  [retrieval error: {e}]")
        return []

# ── Progress helpers ───────────────────────────────────────────────────────

def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
    return {}

def save_progress(prog: dict):
    PROGRESS_FILE.write_text(json.dumps(prog, ensure_ascii=False, indent=2), encoding="utf-8")

# ── Main ───────────────────────────────────────────────────────────────────

def main():
    # Load all model JSONs
    model_data: dict[str, dict] = {}
    for key, fname in MODELS.items():
        path = EVAL_DIR / fname
        if path.exists():
            model_data[key] = json.loads(path.read_text(encoding="utf-8"))
            print(f"Loaded {key}: {len(model_data[key]['per_question'])} questions")
        else:
            print(f"MISSING: {fname}")

    if not model_data:
        print("No files found."); return

    first      = next(iter(model_data.values()))
    questions  = first["per_question"]
    n          = len(questions)
    progress   = load_progress()

    in_scope_idx = [i for i, q in enumerate(questions) if q.get("in_scope", True)]
    print(f"\nProgress: {len(progress)} contexts already saved")
    print(f"In-scope questions: {len(in_scope_idx)}")

    # Phase 1: Retrieve (reuse existing contexts from progress file)
    needs_retrieval = [i for i in in_scope_idx if str(i) not in progress]
    if needs_retrieval:
        print(f"\nPhase 1: Retrieving context for {len(needs_retrieval)} questions...")
        for i in needs_retrieval:
            q_text = questions[i]["question"]
            print(f"  Q{i+1}/{n}: {q_text[:60]}...", end=" ", flush=True)
            ctx = retrieve(q_text)
            progress[str(i)] = {"context": ctx, "scores": {}}
            save_progress(progress)
            print(f"({len(ctx)} chunks)")
            time.sleep(0.3)
        print("  Retrieval complete.\n")
    else:
        print("Phase 1: All contexts already retrieved (from progress file).\n")

    # Phase 2: Score — 1 judge call per model×question
    todo = [(i, key) for i in in_scope_idx for key in model_data
            if key not in progress.get(str(i), {}).get("scores", {})]

    print(f"Phase 2: {len(todo)} model×question pairs to score")
    print(f"  1 judge call each → ~{len(todo) * (SLEEP_OK + 2) // 60 + 1} min minimum\n")

    for idx, (i, key) in enumerate(todo):
        q_item = questions[i]
        q_text = q_item["question"]
        gt     = q_item["ground_truth"]
        ctx    = progress[str(i)]["context"] or ["[no context]"]

        per_q  = model_data[key]["per_question"]
        answer = per_q[i]["answer"] if i < len(per_q) else ""

        is_err = not answer or any(answer.startswith(p) for p in ERROR_PREFIXES)

        print(f"  [{idx+1}/{len(todo)}] Q{i+1} [{key}]: {q_text[:50]}...", end=" ", flush=True)

        if is_err:
            scores = {"f": 0.0, "ar": 0.0, "cp": 0.0, "cr": 0.0}
            print("SKIP (error answer)")
        else:
            scores = score_one(q_text, answer, ctx, gt)
            print(f"f={scores['f']:.2f} ar={scores['ar']:.2f} "
                  f"cp={scores['cp']:.2f} cr={scores['cr']:.2f}")

        progress[str(i)]["scores"][key] = scores
        save_progress(progress)

    # Phase 3: Write updated JSONs and final table
    print("\n\nPhase 3: Writing results...")
    for key, data in model_data.items():
        per_q = data["per_question"]
        scored_items = []

        for i, q_item in enumerate(per_q):
            if not q_item.get("in_scope", True):
                continue
            s = progress.get(str(i), {}).get("scores", {}).get(key)
            if s:
                q_item.update({
                    "faithfulness":      s["f"],
                    "answer_relevancy":  s["ar"],
                    "context_precision": s["cp"],
                    "context_recall":    s["cr"],
                    "context_texts":     [c[:500] for c in progress[str(i)]["context"][:5]],
                })
                scored_items.append(q_item)

        def avg(lst): return round(sum(lst)/len(lst), 4) if lst else 0.0
        data["ragas_scores"] = {
            "faithfulness":      avg([q["faithfulness"]      for q in scored_items]),
            "answer_relevancy":  avg([q["answer_relevancy"]  for q in scored_items]),
            "context_precision": avg([q["context_precision"] for q in scored_items]),
            "context_recall":    avg([q["context_recall"]    for q in scored_items]),
        }

        (EVAL_DIR / MODELS[key]).write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    # Save comparison
    comparison = {k: json.loads((EVAL_DIR / v).read_text(encoding="utf-8"))
                  for k, v in MODELS.items() if (EVAL_DIR / v).exists()}
    (EVAL_DIR / "ragas_comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")

    # Print table
    print(f"\n{'='*82}")
    print("  FINAL RAGAS SCORES")
    print(f"{'='*82}")
    print(f"  {'Model':<32} {'Faith':>7} {'AnswRel':>8} {'CtxPrec':>9} {'CtxRec':>8} {'Halluc':>7}")
    print("  " + "-"*72)
    for key, data in comparison.items():
        s   = data.get("ragas_scores", {})
        mid = data.get("model_id", key)
        print(f"  {mid[:32]:<32} "
              f"{s.get('faithfulness',0):>7.4f} "
              f"{s.get('answer_relevancy',0):>8.4f} "
              f"{s.get('context_precision',0):>9.4f} "
              f"{s.get('context_recall',0):>8.4f} "
              f"{(data.get('hallucination_rate') or 0):>7.2%}")
    print(f"\nSaved: ragas_comparison.json | Progress: scoring_progress.json")

if __name__ == "__main__":
    main()
