"""
structrag/evaluation/parallel_ragas_eval.py
--------------------------------------------
Runs RAGAS evaluation on multiple models simultaneously using subprocesses.

Models evaluated (no Gemini — removed due to daily quota limits):
  - openai/gpt-oss-120b    (Groq)
  - llama-3.3-70b-versatile (Groq)
  - z-ai/glm-5.2           (NVIDIA build)

RAGAS judge: llama-3.3-70b-versatile on Groq (no daily limit).

Each model runs in its own subprocess writing to its own output file.
All 3 start at the same time — total wall time = slowest model.

Usage
-----
    python structrag/evaluation/parallel_ragas_eval.py
    python structrag/evaluation/parallel_ragas_eval.py --quick   # 10 questions
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PY = r"C:\Users\manef\AppData\Local\Programs\Python\Python311\python.exe"

# Load .env
for _l in (PROJECT_ROOT / ".env").read_text(encoding="utf-8").splitlines():
    _l = _l.strip()
    if _l and not _l.startswith("#") and "=" in _l:
        _k, _v = _l.split("=", 1)
        os.environ[_k.strip()] = _v.strip()

# Models to evaluate — no Gemini
MODELS = [
    {"key": "gpt120b",     "model": "openai/gpt-oss-120b",                    "provider": "groq"},
    {"key": "glm52",       "model": "z-ai/glm-5.2",                           "provider": "nvidia"},
    {"key": "llama70b",    "model": "meta/llama-3.1-70b-instruct",            "provider": "nvidia"},
    {"key": "nemotron49b", "model": "nvidia/llama-3.3-nemotron-super-49b-v1",  "provider": "nvidia"},
]

OUTPUT_DIR = PROJECT_ROOT / "structrag" / "evaluation"


def launch_model(model_cfg: dict, quick: bool) -> subprocess.Popen:
    """Launch full_ragas_eval.py for one model as a subprocess."""
    key      = model_cfg["key"]
    model_id = model_cfg["model"]
    provider = model_cfg["provider"]

    out_file = OUTPUT_DIR / f"ragas_{key}.txt"
    script   = PROJECT_ROOT / "structrag" / "evaluation" / "full_ragas_eval.py"

    args = [
        PY, str(script),
        "--models", key,     # single model key
        "--model-id", model_id,
        "--provider", provider,
    ]
    if quick:
        args.append("--quick")

    proc = subprocess.Popen(
        args,
        stdout=open(out_file, "w", encoding="utf-8"),
        stderr=subprocess.STDOUT,
        cwd=str(PROJECT_ROOT),
    )
    print(f"  Started {model_id} (pid={proc.pid}) -> {out_file.name}")
    return proc


def poll_until_done(procs: list[tuple[str, subprocess.Popen]]) -> None:
    """Poll all processes every 20s and print progress until all finish."""
    done = set()
    total = len(procs)
    start = time.time()

    while len(done) < total:
        time.sleep(20)
        for key, proc in procs:
            if key in done:
                continue
            ret = proc.poll()
            if ret is not None:
                elapsed = round(time.time() - start)
                print(f"  [{elapsed}s] {key} finished (exit={ret})")
                done.add(key)
            else:
                # Print last line of output file as progress indicator
                out_file = OUTPUT_DIR / f"ragas_{key}.txt"
                try:
                    lines = out_file.read_text(encoding="utf-8",
                                               errors="replace").splitlines()
                    last = next(
                        (l for l in reversed(lines) if l.strip() and "telemetry" not in l),
                        "..."
                    )
                    elapsed = round(time.time() - start)
                    print(f"  [{elapsed}s] {key}: {last[:80]}")
                except Exception:
                    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Use first 10 questions only")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mode = "QUICK (10 questions)" if args.quick else "FULL (50 questions)"
    print(f"\nStructRAG Parallel RAGAS Evaluation — {mode}")
    print(f"Models  : {[m['key'] for m in MODELS]}")
    print(f"Judge   : openai/gpt-oss-120b (Groq, no daily limit)")
    print(f"Output  : {OUTPUT_DIR}")
    print(f"Starting all models simultaneously...\n")

    procs = []
    for cfg in MODELS:
        proc = launch_model(cfg, args.quick)
        procs.append((cfg["key"], proc))
        time.sleep(2)   # stagger starts slightly

    print(f"\nAll {len(procs)} models running. Polling every 20s...\n")
    poll_until_done(procs)

    print("\n\n=== All models complete. Results: ===")
    for cfg in MODELS:
        key = cfg["key"]
        json_file = OUTPUT_DIR / f"ragas_{key}.json"
        txt_file  = OUTPUT_DIR / f"ragas_{key}.txt"
        if json_file.exists():
            import json
            data = json.loads(json_file.read_text(encoding="utf-8"))
            s = data.get("ragas_scores", {})
            if "error" not in s:
                print(f"\n  {cfg['model']}")
                print(f"    faithfulness      : {s.get('faithfulness', 0):.4f}")
                print(f"    answer_relevancy  : {s.get('answer_relevancy', 0):.4f}")
                print(f"    context_precision : {s.get('context_precision', 0):.4f}")
                print(f"    context_recall    : {s.get('context_recall', 0):.4f}")
                print(f"    hallucination_rate: {data.get('hallucination_rate', 0):.2%}")
            else:
                print(f"\n  {cfg['model']}: RAGAS error — {s['error'][:60]}")
        else:
            print(f"\n  {cfg['model']}: no JSON output (check {txt_file.name})")
