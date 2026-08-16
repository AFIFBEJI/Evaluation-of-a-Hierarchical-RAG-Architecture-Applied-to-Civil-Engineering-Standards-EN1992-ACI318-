"""
tests/test_guardrail_reranker.py
---------------------------------
Tests hallucination guardrail and reranker — no API keys needed.
"""
import sys, os, json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

out = open(BASE / "tests" / "guardrail_reranker_result.txt", "w", encoding="utf-8")
def w(s=""):
    out.write(str(s) + "\n")

try:
    # ── 1. Hallucination guard ─────────────────────────────────────────────
    w("=== Hallucination Guardrail Tests ===\n")
    from structrag.evaluation.hallucination_guard import (
        check_hallucination, extract_cited_clauses
    )

    test_cases = [
        # (answer, expect_clean, description)
        ("The minimum cover is defined in clause 4.4.1.2 and clause 4.4.1.1.", True, "real clauses"),
        ("Per EC2 clause 99.99.99, the cover must be 200mm.", False, "hallucinated clause"),
        ("gamma_c = 1.5 per EC2 2.4.2.4 Table 2.1N.", True, "real clause with table ref"),
        ("The formula is in clause 6.2.2 and §6.2.3.", True, "real subclauses"),
        ("No clause cited — just prose.", True, "no citations"),
        ("Per clause 7.9.4.12 and clause 4.4.1.2, the cover is 25mm.", False, "mixed real+fake"),
    ]

    passed = 0
    for answer, expect_clean, desc in test_cases:
        result = check_hallucination(answer)
        ok = result.is_clean == expect_clean
        status = "PASS" if ok else "FAIL"
        if ok: passed += 1
        w(f"  [{status}] {desc}")
        w(f"         cited={result.cited}  valid={result.valid}  hallucinated={result.hallucinated}")
        w(f"         {result.summary()}")
        w()

    w(f"Hallucination guard: {passed}/{len(test_cases)} passed\n")

    # ── 2. Reranker ────────────────────────────────────────────────────────
    w("=== Reranker Tests ===\n")
    from structrag.retrieval.reranker import rerank_with_fallback

    # Load real chunks for reranking test
    chunks_path = BASE / "structrag/data/chunks/ec2_hierarchical_chunks.json"
    if chunks_path.exists():
        with open(chunks_path, encoding="utf-8") as f:
            all_chunks = json.load(f)

        # Take 10 diverse chunks as candidates
        candidates = all_chunks[:10]
        query = "minimum concrete cover requirement exposure class XC2"

        w(f"Query: {query}")
        w(f"Candidates before reranking: {len(candidates)}")
        w("Top-5 before reranking (original order):")
        for i, c in enumerate(candidates[:5], 1):
            w(f"  [{i}] clause={c.get('clause_number','')}  text={c['text'][:80]}")

        reranked = rerank_with_fallback(query, candidates, top_n=5)

        w(f"\nTop-5 after reranking:")
        for i, c in enumerate(reranked, 1):
            score = c.get('rerank_score', 0)
            score_str = f"{float(score):.4f}" if score != '?' else '?'
            w(f"  [{i}] clause={c.get('clause_number','')}  "
              f"rerank_score={score_str}  "
              f"text={c['text'][:80]}")

        # Verify reranking changed the order
        orig_order = [c["chunk_id"] for c in candidates[:5]]
        new_order  = [c["chunk_id"] for c in reranked]
        order_changed = orig_order != new_order
        w(f"\nOrder changed by reranker: {order_changed}")
        w(f"PASS: Reranker is working" if reranked else "FAIL: No results returned")
    else:
        w("SKIP: chunks file not found")

    w("\nAll tests done.")
except Exception as e:
    import traceback
    w(f"ERROR: {e}")
    w(traceback.format_exc())

out.close()
print("Done -> tests/guardrail_reranker_result.txt")
