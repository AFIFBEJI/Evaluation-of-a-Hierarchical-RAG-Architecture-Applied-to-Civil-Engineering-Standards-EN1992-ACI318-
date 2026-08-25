"""
structrag/rag_pipeline/multi_model_generation.py
-------------------------------------------------
Unified generation interface supporting multiple LLM backends:
  - Groq       (openai/gpt-oss-120b, llama-3.3-70b-versatile, etc.)
  - Gemini     (gemini-2.5-flash — 1M context window, free tier)
  - NVIDIA     (meta/llama-3.1-70b-instruct via build.nvidia.com)

All backends return the same GenerationResult dataclass, so callers are
backend-agnostic.

Environment variables
---------------------
GROQ_API_KEY    + GROQ_MODEL    (default: openai/gpt-oss-120b)
GEMINI_API_KEY  + GEMINI_MODEL  (default: gemini-2.5-flash)
NVIDIA_API_KEY  + NVIDIA_MODEL  (default: meta/llama-3.1-70b-instruct)

Model rankings for this project (large context + free/trial):
  1. gemini-2.5-flash     — 1M context, free tier (10 RPM), best for large context
  2. openai/gpt-oss-120b  — 131K context, fast (500 t/s), free tier Groq
  3. llama-3.3-70b        — 131K context, strong open model, free tier Groq
  4. meta/llama-3.1-70b   — 128K context, NVIDIA trial credits
  5. openai/gpt-oss-20b   — 131K context, fastest (1000 t/s), weaker reasoning
"""

from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Auto-load .env
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    for line in (PROJECT_ROOT / ".env").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ[k.strip()] = v.strip()

from structrag.rag_pipeline.generation import (
    GenerationResult, build_prompt, SYSTEM_PROMPT,
    DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE,
)

# ---------------------------------------------------------------------------
# Backend: Groq
# ---------------------------------------------------------------------------

def _generate_groq(
    question: str,
    chunks: list[dict],
    model: str,
    max_tokens: int,
    temperature: float,
) -> GenerationResult:
    try:
        from groq import Groq
    except ImportError:
        raise ImportError("pip install groq")

    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not set")

    client = Groq(api_key=api_key)
    user_message, source_labels = build_prompt(question, chunks)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_message},
    ]

    import time as _time
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages,
                max_tokens=max_tokens, temperature=temperature,
            )
            answer = (resp.choices[0].message.content or "").strip()
            return GenerationResult(
                answer=answer, sources=source_labels,
                context_used=[c["text"] for c in chunks],
                prompt=user_message, model=model, chunk_type="hierarchical",
            )
        except Exception as e:
            err = str(e)
            if ("429" in err or "rate_limit" in err.lower()) and attempt < 2:
                wait = (attempt + 1) * 30
                print(f"\n    [Groq rate limit — waiting {wait}s]", end="", flush=True)
                _time.sleep(wait)
                continue
            elif "429" in err or "rate_limit" in err.lower():
                return GenerationResult(
                    answer="[Groq rate limit — please wait and retry]",
                    sources=source_labels, context_used=[c["text"] for c in chunks],
                    model=model, chunk_type="hierarchical",
                )
            raise

    return GenerationResult(
        answer="[Groq max retries exceeded]",
        sources=source_labels, context_used=[c["text"] for c in chunks],
        model=model, chunk_type="hierarchical",
    )


# ---------------------------------------------------------------------------
# Backend: Gemini
# ---------------------------------------------------------------------------

def _generate_gemini(
    question: str,
    chunks: list[dict],
    model: str,
    max_tokens: int,
    temperature: float,
) -> GenerationResult:
    # Try new google.genai SDK first, fall back to deprecated google.generativeai
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY not set")

    user_message, source_labels = build_prompt(question, chunks)
    combined_prompt = f"{SYSTEM_PROMPT}\n\n{user_message}"

    try:
        # New SDK: google-genai
        from google import genai
        from google.genai import types as genai_types

        client = genai.Client(api_key=api_key)
        config = genai_types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        )
        response = client.models.generate_content(
            model=model,
            contents=combined_prompt,
            config=config,
        )
        answer = response.text.strip()

    except ImportError:
        # Fallback: old google.generativeai SDK
        try:
            import google.generativeai as genai_old
        except ImportError:
            raise ImportError("pip install google-genai")

        genai_old.configure(api_key=api_key)
        generation_config = genai_old.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        )
        gem_model = genai_old.GenerativeModel(
            model_name=model,
            generation_config=generation_config,
        )
        response = gem_model.generate_content(combined_prompt)
        answer = response.text.strip()

    except Exception as e:
        if "429" in str(e) or "quota" in str(e).lower():
            return GenerationResult(
                answer="[Gemini quota reached — retry later or switch model]",
                sources=source_labels, context_used=[c["text"] for c in chunks],
                model=model, chunk_type="hierarchical",
            )
        if "404" in str(e) or "no longer available" in str(e).lower():
            return GenerationResult(
                answer=f"[Gemini model '{model}' not available — try gemini-3.6-flash or gemini-3.5-flash-lite]",
                sources=source_labels, context_used=[c["text"] for c in chunks],
                model=model, chunk_type="hierarchical",
            )
        raise

    return GenerationResult(
        answer=answer, sources=source_labels,
        context_used=[c["text"] for c in chunks],
        prompt=user_message, model=model, chunk_type="hierarchical",
    )


# ---------------------------------------------------------------------------
# Backend: NVIDIA build (OpenAI-compatible API)
# ---------------------------------------------------------------------------

def _generate_nvidia(
    question: str,
    chunks: list[dict],
    model: str,
    max_tokens: int,
    temperature: float,
) -> GenerationResult:
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("pip install openai")

    api_key = os.environ.get("NVIDIA_API_KEY", "")
    if not api_key:
        raise EnvironmentError("NVIDIA_API_KEY not set")

    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key,
        timeout=180.0,     # 180s timeout — nemotron can be slow
        max_retries=1,
    )

    user_message, source_labels = build_prompt(question, chunks)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_message},
    ]

    try:
        resp = client.chat.completions.create(
            model=model, messages=messages,
            max_tokens=max_tokens, temperature=temperature,
        )
        answer = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        err = str(e)
        if "429" in err or "rate" in err.lower():
            return GenerationResult(
                answer="[NVIDIA rate limit — retry later]",
                sources=source_labels, context_used=[c["text"] for c in chunks],
                model=model, chunk_type="hierarchical",
            )
        if "timeout" in err.lower() or "timed out" in err.lower() or "Timeout" in type(e).__name__:
            return GenerationResult(
                answer="[NVIDIA timeout — model did not respond in 90s]",
                sources=source_labels, context_used=[c["text"] for c in chunks],
                model=model, chunk_type="hierarchical",
            )
        if "Connection" in err or "connection" in err.lower():
            return GenerationResult(
                answer="[NVIDIA connection error — retry later]",
                sources=source_labels, context_used=[c["text"] for c in chunks],
                model=model, chunk_type="hierarchical",
            )
        raise

    return GenerationResult(
        answer=answer, sources=source_labels,
        context_used=[c["text"] for c in chunks],
        prompt=user_message, model=model, chunk_type="hierarchical",
    )


# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------

def _detect_provider(model: str) -> str:
    """Detect backend from model name prefix."""
    model_lower = model.lower()
    if model_lower.startswith("gemini"):
        return "gemini"
    if model_lower.startswith("z-ai/") or model_lower.startswith("nvidia/") or model_lower.startswith("mistralai/"):
        return "nvidia"
    if model_lower.startswith("openai/") or model_lower.startswith("groq/"):
        return "groq"
    if "/" in model:
        return "nvidia"
    return "groq"   # default


# ---------------------------------------------------------------------------
# Unified generate()
# ---------------------------------------------------------------------------

# Available models ranked for this project
RANKED_MODELS = [
    # (model_id, provider, context_tokens, notes)
    ("gemini-3.6-flash",               "gemini", 1_000_000, "Best context, free tier, GA"),
    ("z-ai/glm-5.2",                   "nvidia", 1_000_000, "GLM-5.2, 1M ctx, NVIDIA build"),
    ("gemini-3.5-flash-lite",          "gemini", 1_000_000, "Free tier, lighter model"),
    ("openai/gpt-oss-120b",            "groq",     131_072, "Default, 500 t/s, free tier"),
    ("llama-3.3-70b-versatile",        "groq",     131_072, "Strong open model, free tier"),
    ("meta/llama-3.1-70b-instruct",    "nvidia",   128_000, "NVIDIA trial credits"),
    ("openai/gpt-oss-20b",             "groq",     131_072, "Fastest, weaker reasoning"),
]


def generate_multi(
    question: str,
    chunks: list[dict],
    model: Optional[str] = None,
    provider: Optional[str] = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
) -> GenerationResult:
    """
    Generate an answer using any configured backend.

    Parameters
    ----------
    question  : the user's question
    chunks    : retrieved context chunks
    model     : model name (auto-detects provider from name if not given)
    provider  : "groq" | "gemini" | "nvidia" (overrides auto-detection)
    max_tokens, temperature : generation parameters
    """
    # Resolve model
    if model is None:
        # Try Groq first (already configured and tested)
        model = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

    # Resolve provider
    if provider is None:
        provider = _detect_provider(model)

    if provider == "gemini":
        return _generate_gemini(question, chunks, model, max_tokens, temperature)
    elif provider == "nvidia":
        return _generate_nvidia(question, chunks, model, max_tokens, temperature)
    else:
        return _generate_groq(question, chunks, model, max_tokens, temperature)
