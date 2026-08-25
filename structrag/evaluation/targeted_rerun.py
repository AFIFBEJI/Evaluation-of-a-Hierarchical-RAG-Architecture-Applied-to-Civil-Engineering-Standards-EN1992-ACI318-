"""
structrag/evaluation/targeted_rerun.py
----------------------------------------
Reruns only the questions that failed in a previous evaluation:
  - [NVIDIA rate limit — retry later]  → re-ask GLM
  - [WRONGLY REJECTED by domain classifier] → re-ask all models (keywords now fixed)
  - [Groq rate limit] → re-ask the Groq model

Does NOT rerun questions that already have real answers.
Updates the existing JSON files in place.

Usage
-----
    python structrag/evaluation/targeted_rerun.py --model glm52
    python structrag/evaluation/targeted_rerun.py --model gpt120b
    python structrag/evaluation/targeted_rerun.py --all
"""

from __future__ import annotations
import argparse, io, json, os, sys, time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Load .env
for _l in (PROJECT_ROOT / ".env").read_text(encoding="utf-8").splitlines():
    _l = _l.strip()
    if _l and not _l.startswith("#") and "=" in _l:
        _k, _v = _l.split("=", 1)
        os.environ[_k.strip()] = _v.strip()

from structrag.evaluation.manual_ragas import compute_ragas_scores, _is_error_answer
from structrag.evaluation.full_ragas_eval import run_question, MODELS
from structrag.evaluation.qa_benchmark import QA_BENCHMARK
from structrag.retrieval.table_router import enrich_with_table_router

EVAL_DIR = PROJECT_ROOT / "structrag" / "evaluation"

# Create a lookup: question text → benchmark item
BENCHMARK_BY_QUESTION = {item["question"]: item for item in QA_BENCHMARK}


def rerun_model(model_key: str, dry_run: bool = False) -> None:
    """Rerun failed questions for one model and update its JSON."""
    if model_key not in MODELS:
        print(f"Unknown model key: {model_key}")
        return

    model_id, provider = MODELS[model_key]
    json_path = EVAL_DIR / f"ragas_{model_key}.json"

    if not json_path.exists():
        print(f"No existing results for {model_key} at {json_path}")
        return

    data = json.loads(json_path.read_text(encoding="utf-8"))
    per_q = data.get("per_question", [])

    # Find failed questions
    failed_indices = []
    for i, q_result in enumerate(per_q):
        answer = q_result.get("answer", "")
        if _is_error_answer(answer):
            failed_indices.append(i)

    print(f"\nModel: {model_id}")
    print(f"Total questions: {len(per_q)}")
    print(f"Failed questions to rerun: {len(failed_indices)}")

    if not failed_indices:
        print("Nothing to rerun.")
        return

    if dry_run:
        for i in failed_indices:
            q = per_q[i]
            print(f"  Would rerun Q{i+1}: {q['question'][:60]}")
            print(f"    Old answer: {q.get('answer', '')[:60]}")
        return

    # Rerun each failed question
    reruns_done = 0
    for idx in failed_indices:
        q_result = per_q[idx]
        question  = q_result["question"]
        in_scope  = q_result.get("in_scope", True)

        print(f"\n  Rerunning Q{idx+1}: {question[:65]}...")

        # Get benchmark item for ground_truth
        benchmark_item = BENCHMARK_BY_QUESTION.get(question, {})
        gt = benchmark_item.get("ground_truth", "")

        # Wait between NVIDIA calls to respect rate limits
        if provider == "nvidia" and reruns_done > 0:
            wait = 65
            print(f"    [Waiting {wait}s for NVIDIA rate limit...]")
            time.sleep(wait)

        new_result = run_question(question, model_id, provider, in_scope)
        new_answer = new_result["answer"]
        new_contexts = new_result.get("context_texts", [])

        print(f"    New answer: {new_answer[:100]}")

        # Score with fixed judge
        if not _is_error_answer(new_answer) and in_scope:
            scored = compute_ragas_scores(
                questions=[question],
                answers=[new_answer],
                contexts=[new_contexts or ["[no context]"]],
                ground_truths=[gt],
                verbose=False,
            )
            pq_scores = scored.get("per_question", [{}])[0]
        else:
            pq_scores = {
                "faithfulness": 0.0, "answer_relevancy": 0.0,
                "context_precision": 0.0, "context_recall": 0.0,
            }

        # Update the entry in per_question
        per_q[idx]["answer"]           = new_answer
        per_q[idx]["n_context_chunks"] = len(new_contexts)
        per_q[idx]["elapsed_s"]        = new_result.get("elapsed_s", 0)
        per_q[idx]["faithfulness"]     = pq_scores.get("faithfulness", 0.0)
        per_q[idx]["answer_relevancy"] = pq_scores.get("answer_relevancy", 0.0)
        per_q[idx]["context_precision"]= pq_scores.get("context_precision", 0.0)
        per_q[idx]["context_recall"]   = pq_scores.get("context_recall", 0.0)
        per_q[idx]["hallucination_clean"] = new_result.get("hallucination_clean", True)
        per_q[idx]["hallucinated_clauses"] = new_result.get("hallucinated_clauses", [])

        reruns_done += 1
        print(f"    Scores: faith={pq_scores.get('faithfulness',0):.2f} "
              f"rel={pq_scores.get('answer_relevancy',0):.2f} "
              f"prec={pq_scores.get('context_precision',0):.2f} "
              f"rec={pq_scores.get('context_recall',0):.2f}")

    # Recompute aggregate RAGAS scores
    in_scope_results = [q for q in per_q if q.get("in_scope", True)]
    def avg(key): 
        vals = [q.get(key, 0.0) for q in in_scope_results if key in q]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    data["ragas_scores"] = {
        "faithfulness":      avg("faithfulness"),
        "answer_relevancy":  avg("answer_relevancy"),
        "context_precision": avg("context_precision"),
        "context_recall":    avg("context_recall"),
    }

    # Count hallucinations
    data["hallucination_count"] = sum(1 for q in per_q if not q.get("hallucination_clean", True))
    data["hallucination_rate"]  = round(data["hallucination_count"] / len(per_q), 4)

    # Save updated JSON
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nUpdated {json_path.name}")
    print(f"New RAGAS scores:")
    for m, s in data["ragas_scores"].items():
        print(f"  {m:22s}: {s:.4f}")


def rebuild_comparison() -> None:
    """Rebuild ragas_comparison.json from all individual model JSONs."""
    comparison = {}
    for key in MODELS:
        json_path = EVAL_DIR / f"ragas_{key}.json"
        if json_path.exists():
            comparison[key] = json.loads(json_path.read_text(encoding="utf-8"))

    out_path = EVAL_DIR / "ragas_comparison.json"
    out_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")

    # Print summary table
    print(f"\n{'='*75}")
    print(f"  {'Model':<25} {'Faith':>8} {'AnswRel':>8} {'CtxPrec':>8} {'CtxRec':>8} {'Halluc':>7}")
    print(f"  {'-'*73}")
    for key, data in comparison.items():
        s = data.get("ragas_scores", {})
        if "error" not in s and s:
            model_id = data.get("model_id", key)
            print(f"  {model_id[:25]:<25} "
                  f"{s.get('faithfulness',0):>8.4f} "
                  f"{s.get('answer_relevancy',0):>8.4f} "
                  f"{s.get('context_precision',0):>8.4f} "
                  f"{s.get('context_recall',0):>8.4f} "
                  f"{data.get('hallucination_rate',0):>7.1%}")
    print(f"\nSaved -> {out_path.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Targeted rerun of failed questions")
    parser.add_argument("--model",    type=str, choices=list(MODELS.keys()), default=None)
    parser.add_argument("--all",      action="store_true", help="Rerun all models")
    parser.add_argument("--dry-run",  action="store_true", help="Show what would be rerun, don't call APIs")
    parser.add_argument("--compare",  action="store_true", help="Just rebuild comparison table, no rerun")
    args = parser.parse_args()

    if args.compare:
        rebuild_comparison()
    elif args.all:
        for key in MODELS:
            rerun_model(key, dry_run=args.dry_run)
        rebuild_comparison()
    elif args.model:
        rerun_model(args.model, dry_run=args.dry_run)
        rebuild_comparison()
    else:
        parser.print_help()
