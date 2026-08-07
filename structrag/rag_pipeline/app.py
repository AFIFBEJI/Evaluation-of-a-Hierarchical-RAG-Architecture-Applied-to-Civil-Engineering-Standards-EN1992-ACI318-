"""
app.py
------
StructRAG — functional RAG application entry point.

Combines retrieval + generation into one callable pipeline.
Runs both hierarchical and flat retrieval on the same query so you can
compare answers side-by-side during development.

Usage
-----
Interactive mode (REPL):
    python structrag/rag_pipeline/app.py

Single query (CLI):
    python structrag/rag_pipeline/app.py --query "What is the minimum cover for XC2?"

Retrieval-only mode (no LLM / no API key needed):
    python structrag/rag_pipeline/app.py --query "..." --no-llm
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from structrag.retrieval.retriever import retrieve_hierarchical, retrieve_flat
from structrag.rag_pipeline.generation import generate, GenerationResult
from structrag.rag_pipeline.groq_generation import groq_generate

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_RESULTS  = 5      # chunks retrieved per query
SOURCE_ID  = "ec2_2004_nf"
DIVIDER    = "-" * 70


# ---------------------------------------------------------------------------
# Pipeline functions
# ---------------------------------------------------------------------------

def run_hierarchical(query: str, use_llm: bool = True, use_groq: bool = False) -> dict:
    """Retrieve hierarchically + optionally generate an answer."""
    chunks = retrieve_hierarchical(query, n_results=N_RESULTS, source_id=SOURCE_ID)

    result = {"query": query, "chunk_type": "hierarchical", "chunks": chunks}

    if use_llm and chunks:
        if use_groq:
            # groq_generate does its own retrieval internally, but we reuse
            # the chunks already fetched so we don't hit ChromaDB twice.
            gen = groq_generate(
                question=query,
                n_results=N_RESULTS,
                source_id=SOURCE_ID,
                skip_domain_check=True,  # already decided to call this
            )
        else:
            gen = generate(query, chunks, chunk_type="hierarchical")
        result["answer"]  = gen.answer
        result["sources"] = gen.sources
    elif not chunks:
        result["answer"]  = "[No chunks retrieved]"
        result["sources"] = []

    return result


def run_flat(query: str, use_llm: bool = True) -> dict:
    """Retrieve from flat baseline + optionally generate an answer."""
    chunks = retrieve_flat(query, n_results=N_RESULTS, source_id=SOURCE_ID)

    result = {"query": query, "chunk_type": "flat", "chunks": chunks}

    if use_llm and chunks:
        gen = generate(query, chunks, chunk_type="flat")
        result["answer"]  = gen.answer
        result["sources"] = gen.sources
    elif not chunks:
        result["answer"]  = "[No chunks retrieved]"
        result["sources"] = []

    return result


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _print_chunks(chunks: list[dict], label: str) -> None:
    print(f"\n{DIVIDER}")
    print(f"  {label} — Top {len(chunks)} retrieved chunks")
    print(DIVIDER)
    for i, c in enumerate(chunks, 1):
        score  = c.get("score", "?")
        clause = c.get("clause_number", "")
        path   = c.get("ancestor_path", f"pages {c.get('start_page','')}–{c.get('end_page','')}")
        print(f"\n  [{i}] score={score}  {('clause ' + clause) if clause else ''}")
        print(f"       {path[:80]}")
        print(f"       {c['text'][:300].replace(chr(10), ' ')}")


def _print_answer(result: dict) -> None:
    label = result["chunk_type"].upper()
    print(f"\n{DIVIDER}")
    print(f"  {label} ANSWER")
    print(DIVIDER)
    print(result.get("answer", "[no answer]"))
    sources = result.get("sources", [])
    if sources:
        print(f"\n  Sources cited: {', '.join(sources)}")


# ---------------------------------------------------------------------------
# Single query runner
# ---------------------------------------------------------------------------

def run_query(query: str, use_llm: bool = True, compare: bool = True, use_groq: bool = False) -> None:
    """
    Run one query through both pipelines and print results.

    Parameters
    ----------
    query    : the question
    use_llm  : if False, only print retrieved chunks (no API call)
    compare  : if True, run both hierarchical and flat side-by-side
    use_groq : if True, use Groq instead of OpenAI for generation
    """
    print(f"\n{'='*70}")
    print(f"  Query: {query}")
    if use_groq and use_llm:
        import os
        from structrag.rag_pipeline.groq_generation import DEFAULT_MODEL
        model = os.environ.get("GROQ_MODEL", DEFAULT_MODEL)
        print(f"  LLM  : Groq / {model}")
    print(f"{'='*70}")

    # --- Hierarchical ---
    hier = run_hierarchical(query, use_llm=use_llm, use_groq=use_groq)
    _print_chunks(hier["chunks"], "HIERARCHICAL")
    if use_llm:
        _print_answer(hier)

    if compare:
        # --- Flat baseline (always uses OpenAI if LLM is on, for fair comparison) ---
        flat = run_flat(query, use_llm=use_llm)
        _print_chunks(flat["chunks"], "FLAT BASELINE")
        if use_llm:
            _print_answer(flat)


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------

def repl(use_llm: bool = True, use_groq: bool = False) -> None:
    """Interactive question loop."""
    backend = "Groq" if use_groq else "OpenAI"
    print(f"\nStructRAG -- Eurocode 2 assistant  [backend: {backend}]")
    print("Type a question and press Enter. Type 'quit' to exit.\n")
    while True:
        try:
            query = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Bye.")
            break
        run_query(query, use_llm=use_llm, use_groq=use_groq)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StructRAG -- EC2 RAG pipeline")
    parser.add_argument("--query",  type=str, default=None,
                        help="Single query to run (omit for interactive REPL)")
    parser.add_argument("--no-llm", action="store_true",
                        help="Retrieval only -- skip LLM generation (no API key needed)")
    parser.add_argument("--no-compare", action="store_true",
                        help="Run hierarchical only (skip flat baseline)")
    parser.add_argument("--groq", action="store_true",
                        help="Use Groq (GROQ_API_KEY) instead of OpenAI for generation")
    args = parser.parse_args()

    use_llm  = not args.no_llm
    compare  = not args.no_compare
    use_groq = args.groq

    if args.query:
        run_query(args.query, use_llm=use_llm, compare=compare, use_groq=use_groq)
    else:
        repl(use_llm=use_llm, use_groq=use_groq)
