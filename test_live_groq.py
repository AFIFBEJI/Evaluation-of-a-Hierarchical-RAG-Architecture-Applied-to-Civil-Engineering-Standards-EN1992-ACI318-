import sys, os
from pathlib import Path

BASE = Path(r"C:\Users\manef\OneDrive\Desktop\stageesprit")
sys.path.insert(0, str(BASE))

# Force-read .env directly — bypasses any cached env vars from previous runs
env_path = BASE / ".env"
for line in env_path.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ[k.strip()] = v.strip()

print(f"Key loaded: {os.environ.get('GROQ_API_KEY', 'NOT FOUND')[:12]}...")

from structrag.rag_pipeline.groq_generation import groq_generate

queries = [
    "What is the minimum concrete cover for a beam in exposure class XC2 per Eurocode 2?",
    "What is the shear resistance formula for members without shear reinforcement?",
    "Who won the World Cup in 2022?",
]

out_path = BASE / "groq_live_result.txt"
with open(out_path, "w", encoding="utf-8") as f:
    for q in queries:
        f.write("=" * 60 + "\n")
        f.write(f"QUERY: {q}\n")
        f.write("=" * 60 + "\n")
        try:
            result = groq_generate(question=q, n_results=3, source_id="ec2_2004_nf")
            f.write(f"Model  : {result.model}\n")
            f.write(f"Sources: {result.sources}\n")
            f.write(f"Chunks : {len(result.context_used)}\n\n")
            f.write(f"ANSWER:\n{result.answer}\n\n")
        except Exception as e:
            f.write(f"ERROR: {e}\n\n")

print(f"Done -> {out_path}")
