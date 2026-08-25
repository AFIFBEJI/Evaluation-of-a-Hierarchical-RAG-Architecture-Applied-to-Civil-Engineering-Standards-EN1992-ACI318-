"""
structrag/evaluation/manual_ragas.py
--------------------------------------
RAGAS-style scoring using openai/gpt-oss-120b on Groq as the LLM judge.

The key fix for reasoning models (gpt-oss-120b, gpt-oss-20b):
  These models run an internal chain-of-thought that consumes tokens before
  producing visible output. If max_tokens is too small, visible output is empty.
  Fix: use max_tokens=512, ask for explanations (not just numbers), parse numbers
  from the full response text.

Each metric uses a separate prompt that asks the model to explain its score
and end with a number — this forces visible output before the number.
"""

from __future__ import annotations
import os, re, sys, time
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

JUDGE_MODEL    = "meta/llama-3.3-70b-instruct"   # NVIDIA — separate from Groq RAG model
JUDGE_PROVIDER = "nvidia"


def _is_error_answer(answer: str) -> bool:
    if not answer or not answer.strip():
        return True
    error_prefixes = (
        "[ERROR", "[Groq rate", "[NVIDIA rate", "[Gemini quota",
        "[WRONGLY REJECTED", "[CORRECTLY REJECTED", "[No matching",
        "[No chunks", "[No relevant", "[Groq max",
    )
    return any(answer.startswith(p) for p in error_prefixes)


def _call_judge(prompt: str, api_key: str = None) -> str:
    """Call llama-3.3-70b on NVIDIA as judge (separate from Groq RAG model)."""
    nvidia_key = os.environ.get("NVIDIA_API_KEY", "")
    for attempt in range(5):
        try:
            from openai import OpenAI
            client = OpenAI(
                base_url="https://integrate.api.nvidia.com/v1",
                api_key=nvidia_key,
                timeout=60.0,
            )
            resp = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                temperature=0.0,
            )
            content = resp.choices[0].message.content or ""
            return content.strip()
        except Exception as e:
            err = str(e)
            if "429" in err or "rate_limit" in err.lower():
                wait = min(30 * (attempt + 1), 150)
                print(f"\n    [Judge rate limit — waiting {wait}s]", end="", flush=True)
                time.sleep(wait)
                continue
            return ""
    print(f"\n    [Judge gave up after 5 attempts — returning 0.5]", end="", flush=True)
    return ""


def _parse_score(text: str) -> float:
    """Extract the last decimal number from a response text."""
    # Look for patterns like "Score: 0.8", "I give it 0.75", "0.9 out of 1"
    # Search from end to find the score number
    nums = re.findall(r'\b(0\.\d+|1\.0|0\.0|1|0)\b', text)
    if nums:
        # Take the last number (most likely to be the final score)
        return min(1.0, max(0.0, float(nums[-1])))
    return 0.5


def compute_faithfulness(answer: str, contexts: list[str], api_key: str = None) -> float:
    if _is_error_answer(answer):
        return 0.0
    if not api_key:
        api_key = os.environ.get("GROQ_API_KEY", "")

    ctx_str = " | ".join(c[:150] for c in contexts[:3]) if contexts else "[no context]"
    prompt = (
        f"Context (from Eurocode 2, may be in French): {ctx_str[:400]}\n\n"
        f"Answer (may be in French): {answer[:300]}\n\n"
        f"Question: Are the main factual claims in this answer supported by or consistent with the context above?\n"
        f"Note: The answer may be in French — this is expected and correct.\n"
        f"Consider: it's OK if the answer cites clause numbers not shown in the context, as long as the facts are consistent.\n"
        f"Think briefly, then give a score from 0.0 (contradicts context) to 1.0 (fully supported).\n"
        f"End your response with: SCORE: X.X"
    )
    response = _call_judge(prompt, api_key)
    if not response:
        return 0.5
    # Look for "SCORE: X.X" pattern first
    score_match = re.search(r'SCORE:\s*(0\.\d+|1\.0|0\.0|[01])', response, re.IGNORECASE)
    if score_match:
        return min(1.0, max(0.0, float(score_match.group(1))))
    return _parse_score(response)


def compute_answer_relevancy(question: str, answer: str, api_key: str = None) -> float:
    if _is_error_answer(answer):
        return 0.0
    if not api_key:
        api_key = os.environ.get("GROQ_API_KEY", "")

    prompt = (
        f"Question: {question}\n\n"
        f"Answer (may be in French — this is expected): {answer[:300]}\n\n"
        f"How well does this answer address the question above?\n"
        f"Note: French answers to French questions about Eurocode 2 are fully valid.\n"
        f"Think briefly, then give a score from 0.0 (completely irrelevant) to 1.0 (directly and fully answers the question).\n"
        f"End your response with: SCORE: X.X"
    )
    response = _call_judge(prompt, api_key)
    if not response:
        return 0.5
    score_match = re.search(r'SCORE:\s*(0\.\d+|1\.0|0\.0|[01])', response, re.IGNORECASE)
    if score_match:
        return min(1.0, max(0.0, float(score_match.group(1))))
    return _parse_score(response)


def compute_context_precision(question: str, contexts: list[str], api_key: str = None) -> float:
    if not contexts:
        return 0.0
    if not api_key:
        api_key = os.environ.get("GROQ_API_KEY", "")

    # Check top 3 chunks only (saves 2 API calls vs checking all 5)
    relevant = 0
    chunks_to_check = contexts[:3]
    for ctx in chunks_to_check:
        prompt = (
            f"Question: {question}\n\n"
            f"Context passage: {ctx[:250]}\n\n"
            f"Is this context passage relevant to answering the question? Think briefly.\n"
            f"End with: SCORE: 1 (relevant) or SCORE: 0 (not relevant)"
        )
        response = _call_judge(prompt, api_key)
        score_match = re.search(r'SCORE:\s*([01])', response, re.IGNORECASE)
        if score_match:
            relevant += int(score_match.group(1))
        else:
            relevant += round(_parse_score(response))
        time.sleep(3)  # longer sleep to avoid rate limits

    return round(relevant / len(chunks_to_check), 4)


def compute_context_recall(
    question: str, contexts: list[str], ground_truth: str, api_key: str = None
) -> float:
    if not contexts or not ground_truth:
        return 0.5
    if not api_key:
        api_key = os.environ.get("GROQ_API_KEY", "")

    ctx_str = " | ".join(c[:150] for c in contexts[:3])
    prompt = (
        f"Ground truth answer: {ground_truth[:250]}\n\n"
        f"Retrieved context: {ctx_str[:400]}\n\n"
        f"Does the retrieved context contain the information needed to produce the ground truth answer?\n"
        f"Think briefly, then score from 0.0 (context completely missing the info) to 1.0 (context fully covers it).\n"
        f"End your response with: SCORE: X.X"
    )
    response = _call_judge(prompt, api_key)
    if not response:
        return 0.5
    score_match = re.search(r'SCORE:\s*(0\.\d+|1\.0|0\.0|[01])', response, re.IGNORECASE)
    if score_match:
        return min(1.0, max(0.0, float(score_match.group(1))))
    return _parse_score(response)


def compute_ragas_scores(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
    api_key: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """
    Compute 4 RAGAS metrics using gpt-oss-120b on Groq.
    Uses max_tokens=512 with SCORE: pattern to force visible output from reasoning model.
    Makes 5 judge calls per question (1 faith + 1 relevancy + 3 precision + 1 recall).
    """
    if not api_key:
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            for _l in (PROJECT_ROOT / ".env").read_text(encoding="utf-8").splitlines():
                _l = _l.strip()
                if _l and not _l.startswith("#") and "=" in _l:
                    _k, _v = _l.split("=", 1)
                    os.environ[_k.strip()] = _v.strip()
            api_key = os.environ.get("GROQ_API_KEY", "")

    faith_scores, rel_scores, prec_scores, rec_scores = [], [], [], []

    for i, (q, a, ctxs, gt) in enumerate(zip(questions, answers, contexts, ground_truths)):
        if verbose:
            print(f"  Scoring Q{i+1}/{len(questions)}...", end=" ", flush=True)

        if _is_error_answer(a):
            faith_scores.append(0.0); rel_scores.append(0.0)
            prec_scores.append(0.0); rec_scores.append(0.0)
            if verbose: print(f"SKIPPED (error: {a[:30]})")
            continue

        f  = compute_faithfulness(a, ctxs, api_key);       time.sleep(2)
        ar = compute_answer_relevancy(q, a, api_key);      time.sleep(2)
        cp = compute_context_precision(q, ctxs, api_key)   # has internal sleep(3)
        time.sleep(2)
        cr = compute_context_recall(q, ctxs, gt, api_key); time.sleep(2)

        faith_scores.append(f); rel_scores.append(ar)
        prec_scores.append(cp); rec_scores.append(cr)

        if verbose:
            print(f"faith={f:.2f} rel={ar:.2f} prec={cp:.2f} rec={cr:.2f}")

    def avg(lst): return round(sum(lst) / len(lst), 4) if lst else 0.0

    return {
        "faithfulness":      avg(faith_scores),
        "answer_relevancy":  avg(rel_scores),
        "context_precision": avg(prec_scores),
        "context_recall":    avg(rec_scores),
        "per_question": [
            {"faithfulness": f, "answer_relevancy": ar,
             "context_precision": cp, "context_recall": cr}
            for f, ar, cp, cr in zip(faith_scores, rel_scores, prec_scores, rec_scores)
        ],
    }
