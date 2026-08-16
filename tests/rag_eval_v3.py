"""
tests/rag_eval_v3.py
--------------------
Evaluation harness v3 — uses the complete pipeline:
  - Parent-child retrieval (small child chunks -> large parent context)
  - Hybrid BM25 + vector search
  - Cross-encoder reranking (top-20 -> top-5)
  - Hallucination guardrail on every answer
  - Configurable model (Groq / Gemini / NVIDIA)

Usage
-----
    python tests/rag_eval_v3.py
    python tests/rag_eval_v3.py --model gemini-2.5-flash --provider gemini
    python tests/rag_eval_v3.py --model llama-3.3-70b-versatile

Output -> tests/rag_eval_results_v3.txt
"""
from __future__ import annotations
import argparse, os, sys, time, json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

# Load .env
for _l in (BASE / ".env").read_text(encoding="utf-8").splitlines():
    _l = _l.strip()
    if _l and not _l.startswith("#") and "=" in _l:
        _k, _v = _l.split("=", 1); os.environ[_k.strip()] = _v.strip()

from structrag.rag_pipeline.groq_generation import (
    is_engineering_query, _is_engineering_query_keyword,
    DEFAULT_MODEL, RELEVANCE_THRESHOLD,
)
from structrag.retrieval.hybrid_retriever import retrieve_hybrid
from structrag.retrieval.retriever import retrieve_child
from structrag.chunking.parent_child_chunker import resolve_parents
from structrag.retrieval.reranker import rerank_with_fallback
from structrag.rag_pipeline.multi_model_generation import generate_multi
from structrag.evaluation.hallucination_guard import check_hallucination

QUESTIONS = [
    "What's the minimum concrete cover requirement for reinforcement, and what does it depend on?",
    "What partial safety factor should I use for concrete under persistent and transient design situations?",
    "What partial safety factor applies to reinforcing steel under persistent and transient design situations?",
    "What's the maximum water-cement ratio for exposure class XC1?",
    "What's the minimum concrete strength class required for exposure class XD3?",
    "What ultimate concrete compressive strain is assumed in the parabola-rectangle stress-strain model?",
    "What are exposure classes, and what categories exist (carbonation, chloride, freeze/thaw, etc.)?",
    "What's the difference between characteristic and design values of material strength?",
    "What is structural classification, and how does it relate to concrete cover requirements?",
    "For a beam in exposure class XC2, what cover is required, and what minimum concrete strength class typically goes with that exposure class?",
    "How do I calculate nominal cover from minimum cover, and what allowance should I add for deviation?",
    "What's the formula for wind load on a tall structure?",
    "What behavior factor should I use for seismic design?",
    "What's the required cover for reinforcement in a nuclear containment structure?",
]

N_RESULTS  = 20   # retrieve more, rerank to 5
RERANK_TOP = 5
SOURCE_ID  = "ec2_2004_nf"
DIVIDER    = "=" * 70
SUB_DIV    = "-" * 70

def emit(line="", f=None):
    try: print(line)
    except: pass
    if f: f.write(line + "\n")

def _format_chunk(i, c):
    score = c.get("rerank_score") or c.get("score", "?")
    score_str = f"{float(score):.4f}" if isinstance(score, (int, float)) else str(score)
    clause  = c.get("clause_number", "")
    path    = c.get("ancestor_path", "")[:60]
    preview = c.get("text", "")[:150].replace("\n", " ")
    return (f"  [{i}] rerank={score_str}  clause={clause}\n"
            f"       {path}\n"
            f"       {preview}...")


def run_eval(questions, model, provider, out_file=None):
    parent_path = BASE / "structrag/data/chunks/ec2_parent_chunks.json"
    use_pc = parent_path.exists()

    def w(s=""): emit(s, out_file)

    w(DIVIDER)
    w(f"  StructRAG Evaluation v3")
    w(f"  Model    : {model}")
    w(f"  Provider : {provider or 'auto'}")
    w(f"  Retrieve : {N_RESULTS} candidates -> rerank -> top {RERANK_TOP}")
    w(f"  Mode     : {'parent-child' if use_pc else 'hybrid-only'}")
    w(f"  Questions: {len(questions)}")
    w(DIVIDER)

    for idx, question in enumerate(questions, 1):
        w(); w(DIVIDER)
        w(f"  Q{idx:02d}. {question}")
        w(DIVIDER)

        # ── Domain check ──────────────────────────────────────────────────
        kw_hit    = _is_engineering_query_keyword(question)
        domain_ok = is_engineering_query(question)

        if not domain_ok:
            w(f"  RETRIEVAL FIRED   : NO  (out-of-scope)")
            w(f"  MODEL ANSWER:")
            w(f"  This question is out of scope for this assistant.")
            w()
            continue

        w(f"  RETRIEVAL FIRED   : YES  (keyword={'YES' if kw_hit else 'embedding'})")

        # ── Retrieval ─────────────────────────────────────────────────────
        t0 = time.time()
        if use_pc:
            child_results = retrieve_child(question, n_results=N_RESULTS, source_id=SOURCE_ID)
            # Only use parent-child path if children are actually relevant
            relevant_children = [c for c in child_results if c.get("score", 999) < 0.8]
            if relevant_children:
                context_chunks = resolve_parents(relevant_children, str(parent_path))
            else:
                context_chunks = []
            # Always supplement with hybrid (catches table chunks + fills gaps)
            hybrid = retrieve_hybrid(question, n_results=N_RESULTS, source_id=SOURCE_ID)
            existing_ids = {c["chunk_id"] for c in context_chunks}
            for h in hybrid:
                if h["chunk_id"] not in existing_ids:
                    context_chunks.append(h)
                    existing_ids.add(h["chunk_id"])
        else:
            context_chunks = retrieve_hybrid(question, n_results=N_RESULTS, source_id=SOURCE_ID)

        t_retrieve = time.time() - t0

        if not context_chunks:
            w(f"  RETRIEVAL STATUS  : ZERO RESULTS")
            w(f"  MODEL ANSWER:")
            w(f"  No matching content found in indexed standards.")
            w(); continue

        # ── Reranking ─────────────────────────────────────────────────────
        t1 = time.time()
        context_chunks = rerank_with_fallback(question, context_chunks, top_n=RERANK_TOP)
        t_rerank = time.time() - t1

        w(f"  RETRIEVAL STATUS  : OK  ({len(context_chunks)} after rerank)")
        w(f"  Timings           : retrieve={t_retrieve:.2f}s  rerank={t_rerank:.2f}s")
        w()
        w(f"  RETRIEVED CHUNKS ({len(context_chunks)}):")
        for i, c in enumerate(context_chunks, 1):
            w(_format_chunk(i, c))
        w()

        # ── Generation ────────────────────────────────────────────────────
        w(SUB_DIV)
        w(f"  Calling {model} ...")
        t2 = time.time()
        try:
            result = generate_multi(
                question=question,
                chunks=context_chunks,
                model=model,
                provider=provider,
            )
            answer = result.answer
            t_gen = time.time() - t2
            w(f"  Response time : {t_gen:.1f}s")
            w(f"  Sources cited : {result.sources or 'none'}")
            w()
            w(f"  MODEL ANSWER:")
            for line in answer.splitlines():
                w(f"  {line}")
        except Exception as e:
            answer = f"[ERROR: {e}]"
            w(f"  ERROR: {e}")

        # ── Hallucination check ───────────────────────────────────────────
        guard = check_hallucination(answer, source_id=SOURCE_ID)
        w()
        w(f"  HALLUCINATION CHECK: {guard.summary()}")
        if not guard.is_clean:
            w(f"    Fabricated clauses: {guard.hallucinated}")
        w()

    w(DIVIDER)
    w(f"  Evaluation complete. {len(questions)} questions processed.")
    w(DIVIDER)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    default=None)
    parser.add_argument("--provider", default=None, choices=["groq","gemini","nvidia"])
    parser.add_argument("--n",        type=int, default=None)
    parser.add_argument("--out",      default="tests/rag_eval_results_v3.txt")
    args = parser.parse_args()

    model    = args.model    or os.environ.get("GROQ_MODEL", DEFAULT_MODEL)
    provider = args.provider
    qs       = QUESTIONS[:args.n] if args.n else QUESTIONS

    out_path = BASE / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        run_eval(qs, model, provider, out_file=f)

    print(f"\nSaved -> {out_path}")
