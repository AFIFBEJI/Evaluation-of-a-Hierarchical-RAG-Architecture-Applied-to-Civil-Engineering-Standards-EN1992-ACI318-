"""
structrag/evaluation/ragas_runner.py
-------------------------------------
RAGAS-based quantitative evaluation of the StructRAG pipeline.

Metrics computed
----------------
- faithfulness       : is the answer grounded in the retrieved context?
- answer_relevance   : does the answer actually address the question?
- context_precision  : are the retrieved chunks relevant to the question?
- context_recall     : does the retrieved context contain the ground-truth answer?

Dataset format
--------------
The evaluation dataset is a list of dicts:
    {
        "question":        str,   # the question asked
        "ground_truth":    str,   # the correct answer (from the standard)
        "ground_truth_clauses": list[str],  # clause numbers that must appear
    }

This module:
1. Runs each question through the full pipeline (hybrid retrieval + LLM)
2. Collects (question, answer, contexts, ground_truth) tuples
3. Passes them to RAGAS for scoring
4. Saves results to evaluation/ragas_results.json

Usage
-----
    python structrag/evaluation/ragas_runner.py
    python structrag/evaluation/ragas_runner.py --model gemini-2.5-flash
    python structrag/evaluation/ragas_runner.py --provider nvidia
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env
for _line in (PROJECT_ROOT / ".env").read_text(encoding="utf-8").splitlines():
    _line = _line.strip()
    if _line and not _line.startswith("#") and "=" in _line:
        _k, _v = _line.split("=", 1)
        os.environ[_k.strip()] = _v.strip()

from structrag.retrieval.hybrid_retriever import retrieve_hybrid
from structrag.chunking.parent_child_chunker import resolve_parents
from structrag.rag_pipeline.multi_model_generation import generate_multi
from structrag.evaluation.hallucination_guard import check_hallucination

# ---------------------------------------------------------------------------
# Built-in QA dataset (ground truth from EC2)
# ---------------------------------------------------------------------------

QA_DATASET = [
    {
        "question": "What is the nominal concrete cover formula?",
        "ground_truth": (
            "The nominal cover is cnom = cmin + Δcdev, where cmin is the minimum cover "
            "and Δcdev is the allowance for deviation in execution (clause 4.4.1.1)."
        ),
        "ground_truth_clauses": ["4.4.1.1", "4.4.1.3"],
    },
    {
        "question": "What partial safety factor applies to concrete in persistent design situations?",
        "ground_truth": (
            "The partial safety factor for concrete gamma_c is 1.5 for persistent and "
            "transient design situations, as given in Table 2.1N of clause 2.4.2.4."
        ),
        "ground_truth_clauses": ["2.4.2.4"],
    },
    {
        "question": "What partial safety factor applies to reinforcing steel?",
        "ground_truth": (
            "The partial safety factor for reinforcing steel gamma_s is 1.15 for "
            "persistent and transient design situations (Table 2.1N, clause 2.4.2.4)."
        ),
        "ground_truth_clauses": ["2.4.2.4"],
    },
    {
        "question": "What are the exposure classes for corrosion induced by carbonation?",
        "ground_truth": (
            "Exposure classes XC1 (dry or permanently wet), XC2 (wet, rarely dry), "
            "XC3 (moderate humidity), and XC4 (cyclic wet and dry) — Table 4.1, clause 4.2."
        ),
        "ground_truth_clauses": ["4.2"],
    },
    {
        "question": "What minimum cover is required for structural class S4 in exposure class XC2?",
        "ground_truth": (
            "For structural class S4 and exposure class XC2/XC3, the minimum cover "
            "cmin,dur is 25 mm, per Table 4.4N of clause 4.4.1.2."
        ),
        "ground_truth_clauses": ["4.4.1.2"],
    },
    {
        "question": "What is the recommended structural class for a standard 50-year design life?",
        "ground_truth": (
            "The recommended structural class for a design service life of 50 years is S4, "
            "as defined in clause 4.4.1.2 and Table 4.3N."
        ),
        "ground_truth_clauses": ["4.4.1.2"],
    },
    {
        "question": "What concrete strength class is recommended for exposure class XD3?",
        "ground_truth": (
            "For exposure class XD3/XS2/XS3, Table 4.3N recommends a minimum concrete "
            "strength class of C45/55 (with possible reduction of one class under certain conditions)."
        ),
        "ground_truth_clauses": ["4.4.1.2"],
    },
    {
        "question": "What is the compressive strain at maximum stress (εc2) for C30/37 concrete?",
        "ground_truth": (
            "For concrete class C30/37, the strain at maximum stress εc2 is 2.0 permille "
            "and the ultimate compressive strain εcu2 is 3.5 permille, from Table 3.1."
        ),
        "ground_truth_clauses": ["3.1.2"],
    },
]


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run_ragas_evaluation(
    model: str = None,
    provider: str = None,
    n_results: int = 8,
    source_id: str = "ec2_2004_nf",
    output_path: str = None,
    use_parent_child: bool = True,
) -> dict:
    """
    Run the full evaluation pipeline and compute RAGAS scores.

    Returns a dict with per-question results and aggregate RAGAS scores.
    """
    if model is None:
        model = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

    parent_path = (
        PROJECT_ROOT / "structrag" / "data" / "chunks" / "ec2_parent_chunks.json"
    )

    print(f"\nStructRAG RAGAS Evaluation")
    print(f"Model    : {model}")
    print(f"Provider : {provider or 'auto-detect'}")
    print(f"k        : {n_results}")
    print(f"Questions: {len(QA_DATASET)}")
    print(f"Mode     : {'parent-child' if use_parent_child else 'hierarchical'}")
    print("=" * 60)

    questions:    list[str] = []
    answers:      list[str] = []
    contexts:     list[list[str]] = []
    ground_truths: list[str] = []
    per_question_results: list[dict] = []

    for i, item in enumerate(QA_DATASET, 1):
        q  = item["question"]
        gt = item["ground_truth"]
        gt_clauses = item.get("ground_truth_clauses", [])

        print(f"\nQ{i:02d}: {q}")
        t0 = time.time()

        # ── Retrieval ─────────────────────────────────────────────────────
        if use_parent_child and parent_path.exists():
            from structrag.retrieval.retriever import retrieve_child
            child_results = retrieve_child(q, n_results=n_results, source_id=source_id)
            if child_results:
                context_chunks = resolve_parents(child_results, str(parent_path))
            else:
                context_chunks = retrieve_hybrid(q, n_results=n_results, source_id=source_id)
        else:
            context_chunks = retrieve_hybrid(q, n_results=n_results, source_id=source_id)

        # ── Reranking ─────────────────────────────────────────────────────
        from structrag.retrieval.reranker import rerank_with_fallback
        context_chunks = rerank_with_fallback(q, context_chunks, top_n=5)

        context_texts = [c["text"] for c in context_chunks]

        # ── Generation ────────────────────────────────────────────────────
        try:
            result = generate_multi(
                question=q,
                chunks=context_chunks,
                model=model,
                provider=provider,
            )
            answer = result.answer
        except Exception as e:
            answer = f"[ERROR: {e}]"
            print(f"  Generation error: {e}")

        elapsed = time.time() - t0

        # ── Hallucination check ───────────────────────────────────────────
        guard = check_hallucination(answer, source_id=source_id)

        print(f"  Answer ({elapsed:.1f}s): {answer[:150]}...")
        print(f"  Hallucination check: {guard.summary()}")

        questions.append(q)
        answers.append(answer)
        contexts.append(context_texts)
        ground_truths.append(gt)

        per_question_results.append({
            "question":         q,
            "answer":           answer,
            "ground_truth":     gt,
            "gt_clauses":       gt_clauses,
            "context_chunks":   len(context_chunks),
            "context_preview":  [c["text"][:100] for c in context_chunks[:2]],
            "elapsed_s":        round(elapsed, 2),
            "hallucination":    {
                "is_clean":     guard.is_clean,
                "cited":        guard.cited,
                "hallucinated": guard.hallucinated,
            },
        })

    # ── RAGAS scoring ─────────────────────────────────────────────────────
    ragas_scores = {}
    try:
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
        from datasets import Dataset

        print("\n\nComputing RAGAS scores...")

        ragas_data = Dataset.from_dict({
            "question":   questions,
            "answer":     answers,
            "contexts":   contexts,
            "ground_truth": ground_truths,
        })

        ragas_result = evaluate(
            ragas_data,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        )

        ragas_scores = {
            "faithfulness":       round(float(ragas_result["faithfulness"]),       4),
            "answer_relevancy":   round(float(ragas_result["answer_relevancy"]),   4),
            "context_precision":  round(float(ragas_result["context_precision"]),  4),
            "context_recall":     round(float(ragas_result["context_recall"]),     4),
        }
        print(f"\nRAGAS Scores:")
        for metric, score in ragas_scores.items():
            print(f"  {metric:22s}: {score:.4f}")

    except Exception as e:
        print(f"\nRAGAS scoring failed: {e}")
        print("Install datasets: pip install datasets")
        ragas_scores = {"error": str(e)}

    # ── Save results ──────────────────────────────────────────────────────
    output = {
        "model":       model,
        "provider":    provider or "auto",
        "n_questions": len(QA_DATASET),
        "ragas_scores": ragas_scores,
        "per_question": per_question_results,
    }

    if output_path is None:
        output_path = str(
            PROJECT_ROOT / "structrag" / "evaluation" /
            f"ragas_{(model or 'unknown').replace('/', '_').replace('-', '_')}.json"
        )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved -> {output_path}")

    return output


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAGAS evaluation for StructRAG")
    parser.add_argument("--model",    type=str, default=None)
    parser.add_argument("--provider", type=str, default=None,
                        choices=["groq", "gemini", "nvidia"])
    parser.add_argument("--k",        type=int, default=8)
    parser.add_argument("--no-parent-child", action="store_true")
    args = parser.parse_args()

    run_ragas_evaluation(
        model=args.model,
        provider=args.provider,
        n_results=args.k,
        use_parent_child=not args.no_parent_child,
    )
