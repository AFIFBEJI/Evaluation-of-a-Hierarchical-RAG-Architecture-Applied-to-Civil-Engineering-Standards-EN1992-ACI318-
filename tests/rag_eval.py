"""
tests/rag_eval.py
-----------------
Test harness for the StructRAG + Groq pipeline.

Runs each question through:
  1. Domain classifier  (did retrieval fire?)
  2. ChromaDB retrieval (how many chunks? zero-result flag?)
  3. Groq generation    (what did the model say?)

Prints structured output per question — no grading, just raw results
for manual review.

Usage
-----
    python tests/rag_eval.py
    python tests/rag_eval.py --n 5          # first 5 questions only
    python tests/rag_eval.py --out results.txt  # also save to file
"""

from __future__ import annotations
import argparse
import os
import sys
import time
from pathlib import Path

# ── project root on path ───────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

# ── load .env before any project imports ──────────────────────────────────
env_path = BASE / ".env"
if env_path.exists():
    for _line in env_path.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ[_k.strip()] = _v.strip()

# ── project imports ────────────────────────────────────────────────────────
from structrag.rag_pipeline.groq_generation import (
    is_engineering_query,
    _is_engineering_query_keyword,
    groq_generate,
    DEFAULT_MODEL,
    RELEVANCE_THRESHOLD,
)
from structrag.retrieval.hybrid_retriever import retrieve_hybrid

# ── test questions ─────────────────────────────────────────────────────────
QUESTIONS = [
    # --- in-scope: should trigger retrieval and return EC2 content ---
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
    # --- out-of-scope: should be blocked by domain classifier ---
    "What's the formula for wind load on a tall structure?",
    "What behavior factor should I use for seismic design?",
    "What's the required cover for reinforcement in a nuclear containment structure?",
]

N_RESULTS   = 8     # k=8 per Fix 4
SOURCE_ID   = "ec2_2004_nf"
DIVIDER     = "=" * 70
SUB_DIV     = "-" * 70

# ── helpers ────────────────────────────────────────────────────────────────

def _retrieval_status(chunks: list[dict]) -> str:
    if not chunks:
        return "ZERO RESULTS"
    relevant = [c for c in chunks if c.get("score", 999) <= RELEVANCE_THRESHOLD]
    if not relevant:
        return f"ZERO RELEVANT  ({len(chunks)} retrieved but all score > {RELEVANCE_THRESHOLD})"
    return f"OK  ({len(relevant)} relevant chunk(s) of {len(chunks)} retrieved)"


def _format_chunk(i: int, c: dict) -> str:
    score   = c.get("score", "?")
    clause  = c.get("clause_number", "")
    path    = c.get("ancestor_path", "")
    page    = c.get("page_number", "")
    preview = c.get("text", "")[:200].replace("\n", " ")
    lines = [f"  [{i}] score={score}"]
    if clause:
        lines.append(f"       clause : {clause}")
    if path:
        lines.append(f"       path   : {path[:80]}")
    if page:
        lines.append(f"       page   : {page}")
    lines.append(f"       text   : {preview}...")
    return "\n".join(lines)


def run_eval(questions: list[str], out_file=None) -> None:
    model = os.environ.get("GROQ_MODEL", DEFAULT_MODEL)

    def emit(line=""):
        safe = line.encode("utf-8", "replace").decode("utf-8")
        try:
            print(safe)
        except UnicodeEncodeError:
            print(safe.encode("ascii", "replace").decode("ascii"))
        if out_file:
            out_file.write(safe + "\n")

    emit(DIVIDER)
    emit(f"  StructRAG Evaluation Harness")
    emit(f"  Model  : {model}")
    emit(f"  Source : {SOURCE_ID}")
    emit(f"  k      : {N_RESULTS} chunks per query")
    emit(f"  Total  : {len(questions)} questions")
    emit(DIVIDER)

    for idx, question in enumerate(questions, 1):
        emit()
        emit(DIVIDER)
        emit(f"  Q{idx:02d}. {question}")
        emit(DIVIDER)

        # ── Step 1: domain classification ─────────────────────────────
        kw_hit    = _is_engineering_query_keyword(question)
        domain_ok = is_engineering_query(question)   # keyword only (fast path)

        if domain_ok:
            emit(f"  RETRIEVAL FIRED   : YES  (keyword={'YES' if kw_hit else 'NO, embedding'})")
        else:
            emit(f"  RETRIEVAL FIRED   : NO   -- domain classifier rejected this query")
            emit(f"  (flagged: out-of-scope question)")
            # Still call groq_generate so we show the model's out-of-scope message
            result = groq_generate(
                question=question,
                n_results=N_RESULTS,
                source_id=SOURCE_ID,
                skip_domain_check=False,
            )
            emit()
            emit(f"  RETRIEVED CHUNKS  : 0 (retrieval did not fire)")
            emit()
            emit(f"  MODEL ANSWER:")
            emit(f"  {result.answer}")
            emit()
            continue

        # ── Step 2: retrieval ──────────────────────────────────────────
        chunks = retrieve_hybrid(
            query=question,
            n_results=N_RESULTS,
            source_id=SOURCE_ID,
        )

        status = _retrieval_status(chunks)
        emit(f"  RETRIEVAL STATUS  : {status}")

        if not chunks:
            emit(f"  (flagged: retrieval returned zero chunks)")
            emit()
            emit(f"  MODEL ANSWER:")
            emit(f"  No matching content in indexed standards for this query.")
            emit()
            continue

        emit()
        emit(f"  RETRIEVED CHUNKS ({len(chunks)}):")
        for i, c in enumerate(chunks, 1):
            emit(_format_chunk(i, c))
        emit()

        # Flag zero-relevant separately from zero-retrieved
        relevant = [c for c in chunks if c.get("score", 999) <= RELEVANCE_THRESHOLD]
        if not relevant:
            emit(f"  (flagged: {len(chunks)} chunks retrieved but none below relevance threshold {RELEVANCE_THRESHOLD})")

        # ── Step 3: Groq generation ────────────────────────────────────
        emit(SUB_DIV)
        emit(f"  Calling Groq ({model}) ...")
        t0 = time.time()
        try:
            result = groq_generate(
                question=question,
                n_results=N_RESULTS,
                source_id=SOURCE_ID,
                skip_domain_check=True,   # already checked above
            )
            elapsed = time.time() - t0
            emit(f"  Response time     : {elapsed:.1f}s")
            emit(f"  Sources cited     : {result.sources or 'none'}")
            emit()
            emit(f"  MODEL ANSWER:")
            # indent every line of the answer
            for line in result.answer.splitlines():
                emit(f"  {line}")
        except Exception as e:
            emit(f"  ERROR calling Groq: {e}")

        emit()

    emit(DIVIDER)
    emit(f"  Evaluation complete. {len(questions)} questions processed.")
    emit(DIVIDER)


# ── entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StructRAG evaluation harness")
    parser.add_argument("--n",   type=int, default=None,
                        help="Run only first N questions")
    parser.add_argument("--out", type=str, default=None,
                        help="Also save output to this file")
    args = parser.parse_args()

    questions = QUESTIONS[:args.n] if args.n else QUESTIONS

    if args.out:
        out_path = BASE / args.out
        with open(out_path, "w", encoding="utf-8") as f:
            run_eval(questions, out_file=f)
        print(f"\nSaved to {out_path}")
    else:
        run_eval(questions, out_file=None)
