"""Quick test of z-ai/glm-5.2 on 3 questions. Writes to tests/glm_test_result.txt"""
import sys, os, time, json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

for _l in (BASE / ".env").read_text(encoding="utf-8").splitlines():
    _l = _l.strip()
    if _l and not _l.startswith("#") and "=" in _l:
        _k, _v = _l.split("=", 1); os.environ[_k.strip()] = _v.strip()

out = open(BASE / "tests" / "glm_test_result.txt", "w", encoding="utf-8")

def w(s=""):
    try: print(str(s))
    except: pass
    out.write(str(s) + "\n")

QUESTIONS = [
    "What partial safety factor applies to concrete in persistent design situations?",
    "What minimum cover is required for structural class S4 and exposure class XC2?",
    "How do I calculate nominal cover from minimum cover?",
]

w("=== GLM-5.2 Test (NVIDIA build) ===\n")

try:
    from structrag.retrieval.hybrid_retriever import retrieve_hybrid
    from structrag.chunking.parent_child_chunker import resolve_parents
    from structrag.retrieval.retriever import retrieve_child
    from structrag.retrieval.reranker import rerank_with_fallback
    from structrag.rag_pipeline.multi_model_generation import generate_multi
    from structrag.evaluation.hallucination_guard import check_hallucination

    parent_path = BASE / "structrag/data/chunks/ec2_parent_chunks.json"

    for q in QUESTIONS:
        w(f"Q: {q}")
        w("-" * 60)
        t0 = time.time()

        # Retrieve
        children = retrieve_child(q, n_results=16, source_id="ec2_2004_nf")
        relevant = [c for c in children if c.get("score", 999) < 0.8]
        if relevant and parent_path.exists():
            chunks = resolve_parents(relevant, str(parent_path))
        else:
            chunks = retrieve_hybrid(q, n_results=8, source_id="ec2_2004_nf")
        chunks = rerank_with_fallback(q, chunks, top_n=5)

        # Generate with GLM
        result = generate_multi(
            question=q, chunks=chunks,
            model="z-ai/glm-5.2", provider="nvidia",
        )
        elapsed = round(time.time() - t0, 2)
        guard = check_hallucination(result.answer)

        w(f"Time   : {elapsed}s")
        w(f"Sources: {result.sources}")
        w(f"Answer : {result.answer[:400]}")
        w(f"Guard  : {guard.summary()}")
        w()

    w("Done.")

except Exception as e:
    import traceback
    w(f"ERROR: {e}")
    w(traceback.format_exc())

out.close()
print("Done -> tests/glm_test_result.txt")
