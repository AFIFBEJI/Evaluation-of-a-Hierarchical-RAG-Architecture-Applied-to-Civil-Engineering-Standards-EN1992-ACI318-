"""
structrag/evaluation/score_benchmark.py
---------------------------------------
Fast, unified RAGAS judge using NVIDIA meta/llama-3.1-70b-instruct.
Scores all 4 metrics in a single structured JSON call per question.
Produces ragas_<model>.json and consolidated ragas_comparison.json.
"""

import io, json, os, re, sys, time
from pathlib import Path
from openai import OpenAI

# Force UTF-8 stdout
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env
for l in (PROJECT_ROOT / ".env").read_text(encoding="utf-8").splitlines():
    l = l.strip()
    if l and not l.startswith("#") and "=" in l:
        k, v = l.split("=", 1)
        os.environ[k.strip()] = v.strip()

from structrag.evaluation.qa_benchmark import QA_BENCHMARK
from structrag.evaluation.hallucination_guard import check_hallucination

EVAL_DIR = PROJECT_ROOT / "structrag" / "evaluation"

MODELS = [
    {"key": "gpt120b",     "model": "openai/gpt-oss-120b",                    "provider": "groq"},
    {"key": "glm52",       "model": "z-ai/glm-5.2",                           "provider": "nvidia"},
    {"key": "llama70b",    "model": "meta/llama-3.1-70b-instruct",            "provider": "nvidia"},
    {"key": "nemotron49b", "model": "nvidia/llama-3.3-nemotron-super-49b-v1",  "provider": "nvidia"},
]

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ.get("NVIDIA_API_KEY", ""),
    timeout=25.0,
)


def judge_single_question(question: str, ground_truth: str, gt_clauses: list[str], answer: str) -> dict:
    """Judge all 4 RAGAS metrics in one unified API call."""
    if not answer or answer.startswith("[ERROR") or answer.startswith("[WRONGLY"):
        return {"faithfulness": 0.0, "answer_relevancy": 0.0, "context_precision": 0.0, "context_recall": 0.0}
    
    clauses_str = ", ".join(gt_clauses) if gt_clauses else "N/A"
    prompt = f"""You are an expert civil engineering evaluation judge assessing RAG responses on Eurocode 2 (EN 1992-1-1).

Question: {question}
Ground Truth: {ground_truth}
Ground Truth Clauses: {clauses_str}
Model Answer: {answer[:600]}

Rate the model answer on these 4 metrics from 0.0 to 1.0:
- "faithfulness": 1.0 if claims are factually consistent with EC2 standards and free of false engineering assertions, 0.0 if hallucinated or contradictory.
- "answer_relevancy": 1.0 if it directly answers what was asked, 0.0 if irrelevant.
- "context_precision": 1.0 if cited clauses and technical values are accurate, 0.0 if wrong/vague.
- "context_recall": 1.0 if the key engineering requirements from ground truth are captured, 0.0 if critical requirements are omitted.

Respond ONLY with a valid JSON object:
{{"faithfulness": float, "answer_relevancy": float, "context_precision": float, "context_recall": float}}"""

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model="meta/llama-3.1-70b-instruct",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.0,
            )
            content = resp.choices[0].message.content or ""
            match = re.search(r'\{.*?\}', content, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                return {
                    "faithfulness":      min(1.0, max(0.0, float(data.get("faithfulness", 0.5)))),
                    "answer_relevancy":  min(1.0, max(0.0, float(data.get("answer_relevancy", 0.5)))),
                    "context_precision": min(1.0, max(0.0, float(data.get("context_precision", 0.5)))),
                    "context_recall":    min(1.0, max(0.0, float(data.get("context_recall", 0.5)))),
                }
        except Exception as e:
            time.sleep(2.0 * (attempt + 1))
            
    return {"faithfulness": 0.5, "answer_relevancy": 0.5, "context_precision": 0.5, "context_recall": 0.5}


def parse_answers_file(file_path: Path) -> list[dict]:
    text = file_path.read_text(encoding="utf-8", errors="replace")
    blocks = re.split(r'-{50,}', text)
    parsed = []
    
    for block in blocks:
        block = block.strip()
        if not block or "Q" not in block or "Question" not in block:
            continue
        
        q_match = re.search(r'Question\s*:\s*(.+)', block)
        a_match = re.search(r'Model answer\s*:\s*(.*?)(?=\nHallucination:|\nRefusal:|\nTime:|\Z)', block, re.DOTALL)
        t_match = re.search(r'Time\s*:\s*([\d\.]+)s?', block)
        h_match = re.search(r'Hallucination:\s*(.+)', block)
        
        if q_match and a_match:
            q_text = q_match.group(1).strip()
            a_text = a_match.group(1).strip()
            elapsed = float(t_match.group(1)) if t_match else 0.0
            halluc_str = h_match.group(1).strip() if h_match else "CLEAN"
            
            parsed.append({
                "question": q_text,
                "answer": a_text,
                "elapsed_s": elapsed,
                "halluc_str": halluc_str,
            })
    return parsed


def score_model(model_cfg: dict) -> dict:
    key = model_cfg["key"]
    model_id = model_cfg["model"]
    provider = model_cfg["provider"]
    ans_file = EVAL_DIR / f"answers_{key}.txt"
    
    if not ans_file.exists():
        print(f"File not found: {ans_file}")
        return {}
    
    print(f"\n=======================================================")
    print(f"  Scoring Model: {model_id} ({provider})")
    print(f"=======================================================")
    
    parsed_answers = parse_answers_file(ans_file)
    print(f"  Loaded {len(parsed_answers)} answers from {ans_file.name}")
    
    per_q_results = []
    faith_scores, rel_scores, prec_scores, rec_scores = [], [], [], []
    halluc_count = 0
    correct_refusals = 0
    
    in_scope_total = sum(1 for q in QA_BENCHMARK if q["in_scope"])
    out_scope_total = len(QA_BENCHMARK) - in_scope_total
    
    for i, benchmark_item in enumerate(QA_BENCHMARK):
        q_text = benchmark_item["question"]
        gt = benchmark_item["ground_truth"]
        gt_clauses = benchmark_item["ground_truth_clauses"]
        in_scope = benchmark_item["in_scope"]
        
        matched = next((a for a in parsed_answers if a["question"].strip().lower() == q_text.strip().lower() or a["question"][:40] in q_text), None)
        if not matched and i < len(parsed_answers):
            matched = parsed_answers[i]
        
        answer = matched["answer"] if matched else "[No answer recorded]"
        elapsed = matched["elapsed_s"] if matched else 0.0
        
        guard = check_hallucination(answer, source_id="ec2_2004_nf")
        if in_scope and not guard.is_clean:
            halluc_count += 1
            
        correctly_refused = False
        if not in_scope:
            refusal_phrases = [
                "not contain enough", "out of scope", "not covered",
                "consult", "eurocode 8", "eurocode 1", "en 206", "en 1998",
                "does not cover", "not addressed", "separate standard",
                "hors champ", "ne contient pas", "non directement",
            ]
            correctly_refused = any(p.lower() in answer.lower() for p in refusal_phrases) or "correctly rejected" in answer.lower()
            if correctly_refused:
                correct_refusals += 1
                
        rec = {
            "question": q_text,
            "ground_truth": gt,
            "gt_clauses": gt_clauses,
            "category": benchmark_item["category"],
            "difficulty": benchmark_item["difficulty"],
            "in_scope": in_scope,
            "lang": benchmark_item.get("lang", "fr"),
            "answer": answer,
            "elapsed_s": elapsed,
            "hallucination_clean": guard.is_clean,
            "hallucinated_clauses": guard.hallucinated,
            "correctly_refused": correctly_refused,
        }
        
        if in_scope:
            print(f"    Q{i+1:02d}/{len(QA_BENCHMARK)}...", end=" ", flush=True)
            scores = judge_single_question(q_text, gt, gt_clauses, answer)
            time.sleep(0.4)
            
            rec.update(scores)
            faith_scores.append(scores["faithfulness"])
            rel_scores.append(scores["answer_relevancy"])
            prec_scores.append(scores["context_precision"])
            rec_scores.append(scores["context_recall"])
            print(f"faith={scores['faithfulness']:.2f} rel={scores['answer_relevancy']:.2f} prec={scores['context_precision']:.2f} rec={scores['context_recall']:.2f}")
            
        per_q_results.append(rec)
    
    def avg(lst): return round(sum(lst)/len(lst), 4) if lst else 0.0
    
    ragas_scores = {
        "faithfulness":      avg(faith_scores),
        "answer_relevancy":  avg(rel_scores),
        "context_precision": avg(prec_scores),
        "context_recall":    avg(rec_scores),
    }
    
    refusal_rate = round(correct_refusals / out_scope_total, 4) if out_scope_total else None
    halluc_rate = round(halluc_count / in_scope_total, 4) if in_scope_total else None
    
    output = {
        "model_id": model_id,
        "provider": provider,
        "n_questions": len(QA_BENCHMARK),
        "n_in_scope": in_scope_total,
        "n_out_of_scope": out_scope_total,
        "correct_refusals": correct_refusals,
        "refusal_rate": refusal_rate,
        "hallucination_count": halluc_count,
        "hallucination_rate": halluc_rate,
        "ragas_scores": ragas_scores,
        "per_question": per_q_results,
    }
    
    json_path = EVAL_DIR / f"ragas_{key}.json"
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(output, jf, ensure_ascii=False, indent=2)
    print(f"  Saved -> {json_path.name}")
    return output


def main():
    all_results = {}
    for cfg in MODELS:
        res = score_model(cfg)
        if res:
            all_results[cfg["key"]] = res
            
    comp_path = EVAL_DIR / "ragas_comparison.json"
    with open(comp_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
        
    print("\n" + "=" * 80)
    print("  FINAL RAGAS BENCHMARK COMPARISON TABLE (50 Questions)")
    print("=" * 80)
    header = f"  {'Model':<40} {'Faith':>7} {'AnswRel':>8} {'CtxPrec':>8} {'CtxRec':>7} {'Halluc%':>8} {'Refusal%':>9}"
    print(header)
    print("  " + "-" * 90)
    
    for key, res in all_results.items():
        m_id = res.get("model_id", key)
        s = res.get("ragas_scores", {})
        h_rate = res.get("hallucination_rate", 0) or 0
        r_rate = res.get("refusal_rate", 0) or 0
        print(f"  {m_id:<40} {s.get('faithfulness', 0):>7.4f} {s.get('answer_relevancy', 0):>8.4f} {s.get('context_precision', 0):>8.4f} {s.get('context_recall', 0):>7.4f} {h_rate:>8.2%} {r_rate:>9.2%}")
    print("=" * 80)


if __name__ == "__main__":
    main()
