"""
groq_generation.py
------------------
Groq-powered RAG generation for StructRAG.

Drop-in companion to generation.py — same GenerationResult return type,
same prompt structure, same retrieval layer.  Swap the LLM backend from
OpenAI to Groq by calling groq_generate() instead of generate().

Configuration (environment variables)
--------------------------------------
GROQ_API_KEY   — required. Your Groq API key.
                 Set with:  set GROQ_API_KEY=gsk_...   (Windows CMD)
                            $env:GROQ_API_KEY="gsk_..."  (PowerShell)
                 Never hardcode this value.

GROQ_MODEL     — optional. Default: openai/gpt-oss-120b
                 Override with any model available on your Groq plan,
                 e.g.  set GROQ_MODEL=llama-3.3-70b-versatile

Domain classification
---------------------
Before hitting the vector DB, the query is checked against a two-layer
classifier:

  Layer 1 — keyword scan: fast O(n) check against ~60 engineering terms
            (Eurocode, ACI, reinforcement, shear, exposure class, etc.).
            If any keyword matches, the query is immediately classified as
            engineering-relevant.

  Layer 2 — embedding similarity: if no keyword matched, the query is
            embedded and compared (cosine) against a small set of
            representative engineering sentences.  If similarity > 0.35,
            the query is classified as relevant.

  If neither layer triggers: the function returns a clear "out of scope"
  message without querying the DB or calling Groq.

This threshold (0.35) is intentionally loose — false negatives (missing a
relevant query) are more damaging than false positives (sending a borderline
query to the DB).

Error handling
--------------
- 429 RateLimitError  : caught, returns a clear retry message
- Empty context       : detected before the Groq call, returns a
                        "no matching standard found" message
- Missing API key     : raises EnvironmentError with setup instructions
- groq not installed  : raises ImportError with pip install command

Reuses
------
- structrag.retrieval.retriever.retrieve_hierarchical()
- structrag.rag_pipeline.generation.GenerationResult
- structrag.rag_pipeline.generation.build_prompt()
- structrag.rag_pipeline.generation.SYSTEM_PROMPT
"""

from __future__ import annotations
import os
import sys
import time
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Auto-load .env from project root if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass  # dotenv optional — keys can still be set manually in the shell

from structrag.rag_pipeline.generation import (
    GenerationResult,
    build_prompt,
    SYSTEM_PROMPT,
)
from structrag.retrieval.hybrid_retriever import retrieve_hybrid

# ---------------------------------------------------------------------------
# Configuration — read from environment, never hardcoded
# ---------------------------------------------------------------------------
DEFAULT_MODEL       = "openai/gpt-oss-120b"
DEFAULT_MAX_TOKENS  = 1024
DEFAULT_TEMPERATURE = 0.0   # deterministic for reproducibility / RAGAS scoring
DEFAULT_N_RESULTS   = 8     # Fix 4: raised from 5 to 8

# Score threshold: cosine distance (lower = more similar).
# Chunks with distance > this are considered low-relevance.
# ChromaDB returns distances in [0, 2] for cosine; 0 = identical, 2 = opposite.
RELEVANCE_THRESHOLD = 1.2   # generous — keep anything remotely related

# ---------------------------------------------------------------------------
# Domain classifier — Layer 1: keyword scan
# ---------------------------------------------------------------------------
# fmt: off
_ENGINEERING_KEYWORDS: set[str] = {
    # Standards
    "eurocode", "ec2", "aci", "aci318", "aci 318", "en 1992", "en1992",
    "nf en", "bs en", "din en",
    # Structural elements
    "beam", "column", "slab", "footing", "foundation", "wall", "frame",
    "shear", "bending", "torsion", "punching", "deflection", "buckling",
    "reinforcement", "rebar", "stirrup", "tie", "link", "bar", "tendon",
    "prestress", "prestressed", "post-tension", "pre-tension",
    # Materials
    "concrete", "beton", "acier", "steel", "fck", "fyk", "fcm",
    "compressive strength", "tensile strength", "yield strength",
    "modulus", "elasticity", "creep", "shrinkage",
    # Design concepts
    "cover", "enrobage", "exposure class", "classe d'exposition",
    "durability", "durabilite", "fire resistance", "load combination",
    "ultimate limit state", "serviceability", "uls", "sls", "elu", "els",
    "safety factor", "partial factor", "gamma",
    # EC2 clause terms
    "minimum reinforcement", "armature minimale", "anchorage", "ancrage",
    "lap length", "couture", "detailing", "disposition constructive",
    "neutral axis", "axe neutre", "moment resistance", "shear resistance",
    "vrd", "mrd", "ned", "ved", "clause", "section", "article",
    # ACI terms
    "development length", "splice", "confinement", "ductility", "seismic",
    "chapter", "table r", "commentary",
    # Civil / structural general
    "structural", "civil", "construction", "design", "calcul", "verif",
    "verification", "comply", "compliance", "code", "standard", "norm",
}
# fmt: on


def _is_engineering_query_keyword(query: str) -> bool:
    """Return True if any engineering keyword appears in the query."""
    q_lower = query.lower()
    return any(kw in q_lower for kw in _ENGINEERING_KEYWORDS)


# ---------------------------------------------------------------------------
# Domain classifier — Layer 2: embedding similarity
# ---------------------------------------------------------------------------
# Representative engineering sentences used as reference anchors.
_REFERENCE_SENTENCES = [
    "minimum concrete cover for reinforced concrete beam exposure class",
    "shear resistance design value Eurocode 2 clause 6.2",
    "ACI 318 minimum reinforcement ratio bending member",
    "compressive strength of concrete fck characteristic value",
    "ultimate limit state flexion bending moment capacity",
    "lap length anchorage reinforcement bars structural concrete",
]

# Cache the reference embeddings after first load
_ref_embeddings: Optional[list] = None


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two embedding vectors."""
    import math
    dot   = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _is_engineering_query_embedding(query: str) -> bool:
    """
    Embed the query and compare against reference engineering sentences.
    Returns True if max cosine similarity > 0.35.
    Falls back to False if sentence-transformers is unavailable.
    """
    global _ref_embeddings
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")

        if _ref_embeddings is None:
            _ref_embeddings = model.encode(_REFERENCE_SENTENCES).tolist()

        query_emb = model.encode([query])[0].tolist()
        max_sim = max(
            _cosine_similarity(query_emb, ref) for ref in _ref_embeddings
        )
        return max_sim > 0.35

    except Exception:
        # If embedding fails for any reason, let the query through
        return True


def is_engineering_query(query: str) -> bool:
    """
    Two-layer domain classifier.
    Layer 1 (fast): keyword scan.
    Layer 2 (slower): embedding similarity — only if layer 1 misses.
    """
    if _is_engineering_query_keyword(query):
        return True
    return _is_engineering_query_embedding(query)


# ---------------------------------------------------------------------------
# Groq client
# ---------------------------------------------------------------------------

def _get_groq_client():
    """
    Instantiate and return a Groq client.
    Raises ImportError if the groq package is not installed.
    Raises EnvironmentError if GROQ_API_KEY is not set.
    """
    try:
        from groq import Groq
    except ImportError:
        raise ImportError(
            "The 'groq' package is not installed.\n"
            "Install it with:  pip install groq"
        )

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY environment variable is not set.\n"
            "Set it with:\n"
            "  Windows CMD:        set GROQ_API_KEY=gsk_...\n"
            "  Windows PowerShell: $env:GROQ_API_KEY='gsk_...'\n"
            "  Linux/macOS:        export GROQ_API_KEY=gsk_...\n"
            "Get your key at https://console.groq.com"
        )

    return Groq(api_key=api_key)


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def groq_generate(
    question: str,
    n_results: int = DEFAULT_N_RESULTS,
    source_id: Optional[str] = None,
    model: Optional[str] = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    skip_domain_check: bool = False,
) -> GenerationResult:
    """
    Full RAG pipeline using Groq as the LLM backend.

    Steps
    -----
    1. Domain check  — classify the query; reject off-topic queries early.
    2. Retrieve      — query ChromaDB's ec2_hierarchical collection for top-k.
    3. Empty check   — if no relevant chunks found, return gracefully.
    4. Prompt build  — reuse build_prompt() from generation.py.
    5. Groq call     — send to Groq, handle 429 and other errors.
    6. Return        — GenerationResult with answer + sources + context.

    Parameters
    ----------
    question          : The user's natural-language question.
    n_results         : Number of chunks to retrieve from ChromaDB (default 5).
    source_id         : Optional filter to one source (e.g. "ec2_2004_nf").
    model             : Groq model name. Defaults to GROQ_MODEL env var,
                        then DEFAULT_MODEL ("openai/gpt-oss-120b").
    max_tokens        : Max tokens in the Groq response.
    temperature       : 0.0 for deterministic / reproducible answers.
    skip_domain_check : Set True to bypass the domain classifier (e.g. in
                        evaluation scripts that pre-filter queries).

    Returns
    -------
    GenerationResult with:
        answer       — the model's answer (or an error/out-of-scope message)
        sources      — clause numbers cited
        context_used — chunk texts that were passed to the model
        model        — model name actually used
        chunk_type   — always "hierarchical" (Groq integration uses the
                       structure-aware collection only)
    """
    # Resolve model: explicit arg > env var > default
    resolved_model = model or os.environ.get("GROQ_MODEL") or DEFAULT_MODEL

    # ------------------------------------------------------------------
    # Step 1: Domain classification
    # ------------------------------------------------------------------
    if not skip_domain_check and not is_engineering_query(question):
        return GenerationResult(
            answer=(
                "This assistant is specialized in civil and structural engineering "
                "standards (Eurocode 2, ACI 318, and related codes). "
                "Your question does not appear to be related to these topics. "
                "Please ask a question about structural design, concrete design, "
                "reinforcement, exposure classes, load combinations, or similar subjects."
            ),
            sources=[],
            context_used=[],
            model=resolved_model,
            chunk_type="hierarchical",
        )

    # ------------------------------------------------------------------
    # Step 2: Retrieve relevant chunks from ChromaDB (hybrid BM25 + vector)
    # ------------------------------------------------------------------
    chunks = retrieve_hybrid(
        query=question,
        n_results=n_results,
        source_id=source_id,
    )

    # Filter out low-relevance results
    relevant_chunks = [
        c for c in chunks
        if c.get("score", 999) <= RELEVANCE_THRESHOLD
    ]

    # ------------------------------------------------------------------
    # Step 3: Empty context guard — do not call Groq with nothing
    # ------------------------------------------------------------------
    if not relevant_chunks:
        return GenerationResult(
            answer=(
                "No matching content was found in the indexed engineering standards "
                "for your question. "
                "This may mean the relevant standard has not been added to the "
                "database yet (currently only Eurocode 2 NF EN 1992-1-1:2004 is "
                "indexed), or the topic falls outside the indexed sections. "
                "Please verify the question or consult the standard directly."
            ),
            sources=[],
            context_used=[],
            model=resolved_model,
            chunk_type="hierarchical",
        )

    # ------------------------------------------------------------------
    # Step 4: Build the prompt (reuses generation.py's build_prompt)
    # ------------------------------------------------------------------
    user_message, source_labels = build_prompt(question, relevant_chunks)
    context_texts = [c["text"] for c in relevant_chunks]

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_message},
    ]

    # ------------------------------------------------------------------
    # Step 5: Call Groq
    # ------------------------------------------------------------------
    client = _get_groq_client()

    try:
        response = client.chat.completions.create(
            model=resolved_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        answer = response.choices[0].message.content.strip()

    except Exception as e:
        error_type = type(e).__name__
        error_str  = str(e)

        # Rate limit (HTTP 429)
        if "429" in error_str or "rate_limit" in error_str.lower() or "RateLimitError" in error_type:
            return GenerationResult(
                answer=(
                    "Groq rate limit reached. Please wait a moment and try again. "
                    "If this happens frequently, consider reducing n_results or "
                    "switching to a smaller model via the GROQ_MODEL environment variable."
                ),
                sources=source_labels,
                context_used=context_texts,
                model=resolved_model,
                chunk_type="hierarchical",
            )

        # Authentication error
        if "401" in error_str or "authentication" in error_str.lower() or "AuthenticationError" in error_type:
            raise EnvironmentError(
                f"Groq authentication failed. Check that GROQ_API_KEY is correct.\n"
                f"Original error: {error_str}"
            )

        # Any other error — re-raise with context
        raise RuntimeError(
            f"Groq API call failed ({error_type}): {error_str}"
        ) from e

    # ------------------------------------------------------------------
    # Step 6: Return result
    # ------------------------------------------------------------------
    return GenerationResult(
        answer=answer,
        sources=source_labels,
        context_used=context_texts,
        prompt=user_message,
        model=resolved_model,
        chunk_type="hierarchical",
    )


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test Groq RAG generation")
    parser.add_argument(
        "--query", type=str,
        default="What is the minimum concrete cover for a beam in exposure class XC2?",
        help="Question to ask"
    )
    parser.add_argument(
        "--n", type=int, default=5,
        help="Number of chunks to retrieve"
    )
    parser.add_argument(
        "--source", type=str, default="ec2_2004_nf",
        help="Source ID filter (default: ec2_2004_nf)"
    )
    parser.add_argument(
        "--skip-domain-check", action="store_true",
        help="Bypass domain classifier"
    )
    args = parser.parse_args()

    print(f"Query  : {args.query}")
    print(f"Model  : {os.environ.get('GROQ_MODEL', DEFAULT_MODEL)}")
    print(f"Source : {args.source}")
    print()

    result = groq_generate(
        question=args.query,
        n_results=args.n,
        source_id=args.source,
        skip_domain_check=args.skip_domain_check,
    )

    print("=" * 60)
    print("ANSWER")
    print("=" * 60)
    print(result.answer)
    print()
    print(f"Sources cited : {', '.join(result.sources) or 'none'}")
    print(f"Model used    : {result.model}")
    print(f"Context chunks: {len(result.context_used)}")
