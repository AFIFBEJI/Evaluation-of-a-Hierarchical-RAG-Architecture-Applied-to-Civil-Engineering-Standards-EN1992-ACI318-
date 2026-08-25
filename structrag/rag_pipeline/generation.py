"""
generation.py
-------------
RAG generation layer — takes retrieved chunks and a question, builds a
structured prompt, and calls an LLM to produce a grounded answer.

LLM backend
-----------
Uses the OpenAI chat completions API (gpt-4o-mini by default — fast and cheap
for development; swap to gpt-4o for final evaluation).

The API key is read from the OPENAI_API_KEY environment variable.
Never hardcode the key — set it in your shell or a .env file.

Prompt design
-------------
The system prompt enforces three rules that matter for safety-critical domains:
  1. Answer ONLY from the provided context — no hallucination from training data.
  2. Always cite the specific clause number and source.
  3. If the context does not contain enough information, say so explicitly
     rather than guessing.

These rules are what makes the RAGAS faithfulness score meaningful — if the
model ignores the context, faithfulness drops and we can measure it.

Output
------
Returns a GenerationResult dataclass containing:
    answer        str    — the LLM-generated answer
    sources       list   — clause numbers / chunk IDs cited
    context_used  list   — the chunk texts that were passed to the LLM
    prompt        str    — the full prompt (useful for debugging)
"""

from __future__ import annotations
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_MODEL    = "gpt-4o-mini"
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.0     # deterministic — important for reproducibility

SYSTEM_PROMPT = """You are a structural engineering assistant specializing in \
Eurocode 2 (EN 1992-1-1).

Your rules:
1. Answer ONLY using the provided context passages. Do not use training data not in context.
2. NEVER mention clause numbers, article numbers, section numbers, table numbers, \
equation numbers, expression numbers, or standard names like "EN 206-1" or "EC2". \
No references of any kind. Give the direct answer only.
3. When a formula is in the context, ALWAYS write it explicitly. \
Example: x = (As × fyd) / (η × fcd × b × λ). Never describe a formula in words only.
4. Define every symbol used in a formula immediately after it.
5. Stay focused on exactly what was asked. Do not add unrelated conditions or topics.
6. Do NOT add meta-commentary such as "Les symboles utilisés sont:", \
"Il n'y a pas de formule", "Il convient de noter que", \
"Il est recommandé de consulter", or any similar phrases. \
Just answer directly without explaining your own response.
7. Use ALL relevant information from the context. Only say "Je ne dispose pas de \
suffisamment d'informations" if the context is completely unrelated to the question.
8. Be concise. Answer in French if the question is in French."""


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class GenerationResult:
    answer:       str
    sources:      list[str]           = field(default_factory=list)
    context_used: list[str]           = field(default_factory=list)
    prompt:       str                  = ""
    model:        str                  = DEFAULT_MODEL
    chunk_type:   str                  = "hierarchical"   # or "flat"

    def __str__(self) -> str:
        lines = [
            f"Answer:\n{self.answer}",
            f"\nSources: {', '.join(self.sources) or 'none cited'}",
            f"Model: {self.model} | Chunk type: {self.chunk_type}",
            f"Context chunks used: {len(self.context_used)}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Build the prompt from retrieved chunks
# ---------------------------------------------------------------------------

def _build_context_block(chunks: list[dict]) -> tuple[str, list[str]]:
    """
    Format retrieved chunks into a numbered context block for the prompt.
    Returns (context_text, list_of_source_labels).

    For hierarchical chunks: includes clause number + ancestor path as header.
    For flat chunks: includes page range.
    """
    context_parts: list[str] = []
    source_labels: list[str] = []

    for i, chunk in enumerate(chunks, 1):
        chunk_type = (chunk.get("chunk_type") or
                      chunk.get("metadata", {}).get("chunk_type", ""))
        is_hierarchical = chunk_type in ("hierarchical", "parent", "child")

        if is_hierarchical:
            clause = chunk.get("clause_number") or chunk.get("metadata", {}).get("clause_number", "")
            path   = chunk.get("ancestor_path") or chunk.get("metadata", {}).get("ancestor_path", "")
            header = f"[Context {i}] Clause {clause} — {path}" if clause else f"[Context {i}] {path}"
            if clause:
                source_labels.append(clause)
        else:
            start = chunk.get("start_page", "?")
            end   = chunk.get("end_page", "?")
            header = f"[Context {i}] Pages {start}-{end}"
            source_labels.append(f"pp.{start}-{end}")

        context_parts.append(f"{header}\n{chunk['text']}")

    return "\n\n---\n\n".join(context_parts), source_labels


def build_prompt(question: str, chunks: list[dict]) -> tuple[str, list[str]]:
    """Return (user_message_text, source_labels)."""
    context_block, source_labels = _build_context_block(chunks)
    user_message = (
        f"Context passages from the structural engineering standards:\n\n"
        f"{context_block}\n\n"
        f"---\n\n"
        f"Question: {question}\n\n"
        f"Answer based strictly on the context above, citing specific clause numbers:"
    )
    return user_message, source_labels


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def generate(
    question: str,
    chunks: list[dict],
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    chunk_type: str = "hierarchical",
) -> GenerationResult:
    """
    Generate an answer grounded in the retrieved chunks.

    Parameters
    ----------
    question   : the user's question
    chunks     : retrieved chunk dicts from retriever.py
    model      : OpenAI model name
    max_tokens : maximum tokens in the response
    temperature: 0.0 for deterministic answers (recommended for evaluation)
    chunk_type : "hierarchical" or "flat" — recorded in result for tracking
    """
    try:
        import openai
    except ImportError:
        raise ImportError(
            "openai package not installed. Run: pip install openai"
        )

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY environment variable is not set.\n"
            "Set it with:  set OPENAI_API_KEY=sk-...  (Windows)\n"
            "Or create a .env file and load it before running."
        )

    client = openai.OpenAI(api_key=api_key)

    user_message, source_labels = build_prompt(question, chunks)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_message},
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    answer = response.choices[0].message.content.strip()
    context_texts = [c["text"] for c in chunks]

    return GenerationResult(
        answer=answer,
        sources=source_labels,
        context_used=context_texts,
        prompt=user_message,
        model=model,
        chunk_type=chunk_type,
    )


# ---------------------------------------------------------------------------
# Smoke test (no API key needed — just tests prompt building)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    dummy_chunks = [
        {
            "chunk_id":     "ec2_2004_nf_subclause_4.4.1",
            "chunk_type":   "hierarchical",
            "clause_number": "4.4.1",
            "ancestor_path": "Eurocode 2 > 4 Durabilité > 4.4 Enrobage > 4.4.1",
            "text": (
                "[Eurocode 2 > 4 Durabilité > 4.4 Enrobage > 4.4.1]\n"
                "4.4.1  Généralités\n\n"
                "(1)P L'enrobage nominal est l'enrobage minimal plus la tolérance "
                "d'exécution Δcdev.\n"
                "(2)P La valeur minimale de l'enrobage cmin est déterminée par les "
                "exigences de durabilité (Tableau 4.2) et par les exigences d'adhérence."
            ),
            "page_number":  51,
        }
    ]
    question = "What is the minimum concrete cover for a beam in exposure class XC2?"
    prompt, sources = build_prompt(question, dummy_chunks)
    print("=== Prompt preview ===")
    print(prompt[:800])
    print(f"\nSources that would be cited: {sources}")
    print("\n(Set OPENAI_API_KEY to run a live generation)")
