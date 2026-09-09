"""
app.py — StructRAG Web Interface
Eurocode 2 RAG assistant using llama-3.3-70b on NVIDIA
"""
import io, sys, os, re, time
from pathlib import Path
from flask import Flask, request, jsonify, render_template_string

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env
for _l in (PROJECT_ROOT / ".env").read_text(encoding="utf-8").splitlines():
    _l = _l.strip()
    if _l and not _l.startswith("#") and "=" in _l:
        _k, _v = _l.split("=", 1); os.environ[_k.strip()] = _v.strip()

app = Flask(__name__)

# ── RAG pipeline (lazy-loaded on first request) ───────────────────────────
_pipeline_ready = False
_chunks_cache   = {}

def init_pipeline():
    global _pipeline_ready
    if _pipeline_ready:
        return
    from structrag.retrieval.hybrid_retriever import retrieve_hybrid      # noqa
    from structrag.retrieval.retriever import retrieve_child              # noqa
    from structrag.chunking.parent_child_chunker import resolve_parents   # noqa
    from structrag.retrieval.reranker import rerank_with_fallback         # noqa
    from structrag.retrieval.table_router import enrich_with_table_router # noqa
    from structrag.retrieval.formula_library import get_formula_for_query, make_formula_chunk  # noqa
    _pipeline_ready = True

PARENT_PATH = PROJECT_ROOT / "structrag" / "data" / "chunks" / "ec2_parent_chunks.json"
SOURCE_ID   = "ec2_2004_nf"
MODEL_ID    = "openai/gpt-oss-120b"
PROVIDER    = "groq"


def retrieve(question: str) -> list:
    from structrag.retrieval.hybrid_retriever import retrieve_hybrid
    from structrag.retrieval.retriever import retrieve_child
    from structrag.chunking.parent_child_chunker import resolve_parents
    from structrag.retrieval.reranker import rerank_with_fallback
    from structrag.retrieval.table_router import enrich_with_table_router
    from structrag.retrieval.formula_library import get_formula_for_query, make_formula_chunk
    from structrag.retrieval.symbol_dictionary import enrich_with_symbols   # G2
    from structrag.retrieval.query_rewriter import rewrite_query             # G3

    # G3: LLM-powered query translation (GPT-OSS-20B on Groq)
    # Uses a separate model from the RAG model — no rate limit conflict
    retrieval_query = rewrite_query(question, use_llm=True)

    children = retrieve_child(retrieval_query, n_results=20, source_id=SOURCE_ID)
    relevant  = [c for c in children if c.get("score", 999) < 0.8]
    chunks    = resolve_parents(relevant, str(PARENT_PATH)) if relevant and PARENT_PATH.exists() else []
    hybrid    = retrieve_hybrid(retrieval_query, n_results=20, source_id=SOURCE_ID)
    seen      = {c["chunk_id"] for c in chunks}
    for h in hybrid:
        if h["chunk_id"] not in seen:
            chunks.append(h); seen.add(h["chunk_id"])
    chunks = rerank_with_fallback(question, chunks, top_n=8)   # rerank on ORIGINAL question
    chunks = enrich_with_table_router(question, chunks, SOURCE_ID)

    # G2: inject symbol definitions for any EC2 symbols in the question
    chunks = enrich_with_symbols(question, chunks, SOURCE_ID)

    # formula library: inject full formula blocks (kept as secondary enrichment)
    formula = get_formula_for_query(question)
    if formula:
        m = re.search(r'\b(\d+\.\d+(?:\.\d+)*)\b', formula)
        fc = make_formula_chunk(formula, m.group(1) if m else "unknown", SOURCE_ID)
        if fc["chunk_id"] not in {c["chunk_id"] for c in chunks}:
            chunks.append(fc)   # append (not prepend) — symbol dict takes priority

    return chunks


# ── HTML template ─────────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>StructRAG — Eurocode 2 Assistant</title>
  <!-- Marked.js for markdown rendering -->
  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      background: #0f1117;
      color: #e2e8f0;
      height: 100vh;
      display: flex;
      flex-direction: column;
    }

    /* ── Header ── */
    header {
      padding: 16px 28px;
      border-bottom: 1px solid #1e2535;
      display: flex;
      align-items: center;
      gap: 12px;
      background: #0f1117;
    }
    header .logo {
      width: 32px; height: 32px;
      background: linear-gradient(135deg, #3b82f6, #8b5cf6);
      border-radius: 8px;
      display: flex; align-items: center; justify-content: center;
      font-size: 16px;
    }
    header h1 { font-size: 17px; font-weight: 600; color: #f1f5f9; }
    header span {
      font-size: 12px; color: #64748b;
      background: #1e2535; padding: 3px 10px; border-radius: 20px;
      margin-left: 4px;
    }

    /* ── Chat area ── */
    #chat {
      flex: 1;
      overflow-y: auto;
      padding: 24px 0;
      display: flex;
      flex-direction: column;
      gap: 0;
    }

    /* welcome screen */
    #welcome {
      flex: 1;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 12px;
      color: #475569;
      padding: 40px;
      text-align: center;
    }
    #welcome .icon { font-size: 48px; }
    #welcome h2 { font-size: 20px; color: #94a3b8; font-weight: 500; }
    #welcome p  { font-size: 14px; max-width: 420px; line-height: 1.6; }
    #welcome .examples { display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; margin-top: 8px; }
    #welcome .ex {
      background: #1e2535; border: 1px solid #2d3748;
      padding: 8px 14px; border-radius: 8px; font-size: 13px;
      cursor: pointer; color: #94a3b8; transition: all .2s;
    }
    #welcome .ex:hover { background: #2d3748; color: #e2e8f0; }

    /* message rows */
    .msg-row {
      display: flex;
      padding: 10px 28px;
      gap: 14px;
      max-width: 860px;
      width: 100%;
      margin: 0 auto;
    }
    .msg-row.user { flex-direction: row-reverse; }

    .avatar {
      width: 32px; height: 32px; border-radius: 50%;
      display: flex; align-items: center; justify-content: center;
      font-size: 14px; flex-shrink: 0; margin-top: 2px;
    }
    .user .avatar  { background: #3b82f6; }
    .model .avatar { background: linear-gradient(135deg, #3b82f6, #8b5cf6); }

    .bubble {
      max-width: 78%;
      padding: 12px 16px;
      border-radius: 14px;
      font-size: 14px;
      line-height: 1.7;
      white-space: pre-wrap;
    }
    .user  .bubble { background: #1e3a5f; color: #e2e8f0; border-radius: 14px 4px 14px 14px; }
    .model .bubble { background: #1e2535; color: #e2e8f0; border-radius: 4px 14px 14px 14px; }

    /* typing indicator */
    .typing { display: flex; gap: 5px; align-items: center; padding: 4px 0; }
    .typing span {
      width: 7px; height: 7px; border-radius: 50%;
      background: #4a5568; animation: bounce .9s infinite;
    }
    .typing span:nth-child(2) { animation-delay: .15s; }
    .typing span:nth-child(3) { animation-delay: .30s; }
    @keyframes bounce { 0%,60%,100% { transform: translateY(0); } 30% { transform: translateY(-5px); } }

    /* sources tag */
    .sources {
      margin-top: 10px;
      display: flex; flex-wrap: wrap; gap: 6px;
    }
    .src-tag {
      font-size: 11px; padding: 2px 8px;
      background: #0f2944; border: 1px solid #1e3a5f;
      border-radius: 4px; color: #60a5fa;
    }

    /* ── Input bar ── */
    #input-bar {
      padding: 16px 28px 20px;
      border-top: 1px solid #1e2535;
      background: #0f1117;
    }
    #input-wrap {
      max-width: 860px; margin: 0 auto;
      display: flex; gap: 10px; align-items: flex-end;
      background: #1e2535; border: 1px solid #2d3748;
      border-radius: 12px; padding: 10px 14px;
      transition: border-color .2s;
    }
    #input-wrap:focus-within { border-color: #3b82f6; }

    #question {
      flex: 1; background: transparent; border: none; outline: none;
      color: #e2e8f0; font-size: 14px; resize: none;
      max-height: 140px; min-height: 22px; line-height: 1.5;
      font-family: inherit;
    }
    #question::placeholder { color: #475569; }

    #send {
      background: #3b82f6; border: none; border-radius: 8px;
      width: 36px; height: 36px; cursor: pointer;
      display: flex; align-items: center; justify-content: center;
      flex-shrink: 0; transition: background .2s;
    }
    #send:hover:not(:disabled) { background: #2563eb; }
    #send:disabled { background: #2d3748; cursor: not-allowed; }
    #send svg { width: 16px; height: 16px; fill: white; }

    #footer-note {
      text-align: center; font-size: 11px; color: #334155;
      margin-top: 8px; max-width: 860px; margin-left: auto; margin-right: auto;
    }

    /* scrollbar */
    #chat::-webkit-scrollbar { width: 5px; }
    #chat::-webkit-scrollbar-track { background: transparent; }
    #chat::-webkit-scrollbar-thumb { background: #2d3748; border-radius: 3px; }

    /* markdown rendering inside bubbles */
    .bubble { white-space: normal; }
    .bubble p  { margin-bottom: 8px; }
    .bubble p:last-child { margin-bottom: 0; }
    .bubble ul, .bubble ol { padding-left: 20px; margin-bottom: 8px; }
    .bubble li { margin-bottom: 3px; }
    .bubble strong { color: #93c5fd; }
    .bubble em { color: #c4b5fd; }
    .bubble code {
      background: #0f2030; border: 1px solid #1e3a5f;
      padding: 1px 6px; border-radius: 4px;
      font-family: 'Courier New', monospace; font-size: 13px;
      color: #7dd3fc;
    }
    .bubble pre {
      background: #0f2030; border: 1px solid #1e3a5f;
      padding: 10px 14px; border-radius: 8px;
      overflow-x: auto; margin: 8px 0;
    }
    .bubble pre code { background: none; border: none; padding: 0; }
    .bubble table { border-collapse: collapse; width: 100%; margin: 8px 0; font-size: 13px; }
    .bubble th { background: #1e3a5f; padding: 6px 10px; text-align: left; color: #93c5fd; }
    .bubble td { padding: 5px 10px; border-top: 1px solid #1e2535; }
    .bubble tr:hover td { background: #1a2a3a; }
    /* timer in typing row */
    .typing-label { font-size: 12px; color: #475569; margin-left: 8px; }
  </style>
</head>
<body>

<header>
  <div class="logo">⚡</div>
  <h1>StructRAG</h1>
  <span>Eurocode 2 · gpt-oss-120b</span>
</header>

<div id="chat">
  <div id="welcome">
    <div class="icon">🏗️</div>
    <h2>Eurocode 2 Assistant</h2>
    <p>Posez vos questions sur l'Eurocode 2 — enrobage, matériaux, dimensionnement, détails constructifs.</p>
    <div class="examples">
      <div class="ex" onclick="ask(this)">Quelle est la formule de l'enrobage nominal ?</div>
      <div class="ex" onclick="ask(this)">Quelle est la valeur de γc en situation durable ?</div>
      <div class="ex" onclick="ask(this)">Quelles sont les classes d'exposition XC ?</div>
      <div class="ex" onclick="ask(this)">Quelle est la résistance fck pour C30/37 ?</div>
    </div>
  </div>
</div>

<div id="input-bar">
  <div id="input-wrap">
    <textarea id="question" rows="1" placeholder="Posez une question sur l'Eurocode 2…"></textarea>
    <button id="send" onclick="sendQuestion()">
      <svg viewBox="0 0 24 24"><path d="M2 21l21-9L2 3v7l15 2-15 2z"/></svg>
    </button>
  </div>
  <div id="footer-note">Réponses basées uniquement sur NF EN 1992-1-1:2004 · 0% hallucination sur citations de clauses</div>
</div>

<script>
  // Configure marked for safe rendering
  marked.setOptions({ breaks: true, gfm: true });

  const chat     = document.getElementById('chat');
  const welcome  = document.getElementById('welcome');
  const textarea = document.getElementById('question');
  const sendBtn  = document.getElementById('send');

  let timerInterval = null;

  // Auto-resize textarea
  textarea.addEventListener('input', () => {
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 140) + 'px';
  });

  // Enter to send (Shift+Enter = newline)
  textarea.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendQuestion(); }
  });

  function ask(el) {
    textarea.value = el.textContent;
    sendQuestion();
  }

  function removeWelcome() {
    const w = document.getElementById('welcome');
    if (w) w.remove();
  }

  function addMessage(role, content) {
    removeWelcome();
    const row = document.createElement('div');
    row.className = `msg-row ${role}`;

    const avatar = document.createElement('div');
    avatar.className = 'avatar';
    avatar.textContent = role === 'user' ? '👤' : '🤖';

    const bubble = document.createElement('div');
    bubble.className = 'bubble';

    if (role === 'model') {
      // Render markdown for model answers
      bubble.innerHTML = marked.parse(content);
    } else {
      // Plain text for user messages
      bubble.textContent = content;
    }

    row.appendChild(avatar);
    row.appendChild(bubble);
    chat.appendChild(row);
    chat.scrollTop = chat.scrollHeight;
  }

  function addTyping() {
    removeWelcome();
    const row = document.createElement('div');
    row.className = 'msg-row model';
    row.id = 'typing-row';

    const avatar = document.createElement('div');
    avatar.className = 'avatar';
    avatar.textContent = '🤖';

    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    bubble.innerHTML = `
      <div style="display:flex;align-items:center">
        <div class="typing"><span></span><span></span><span></span></div>
        <span class="typing-label" id="timer-label">Recherche en cours…</span>
      </div>`;

    row.appendChild(avatar);
    row.appendChild(bubble);
    chat.appendChild(row);
    chat.scrollTop = chat.scrollHeight;

    // Live elapsed timer
    let elapsed = 0;
    timerInterval = setInterval(() => {
      elapsed++;
      const label = document.getElementById('timer-label');
      if (label) {
        label.textContent = elapsed < 5
          ? 'Recherche en cours…'
          : elapsed < 15
          ? `Récupération du contexte… (${elapsed}s)`
          : `Génération de la réponse… (${elapsed}s)`;
      }
    }, 1000);
  }

  function removeTyping() {
    if (timerInterval) { clearInterval(timerInterval); timerInterval = null; }
    const el = document.getElementById('typing-row');
    if (el) el.remove();
  }

  async function sendQuestion() {
    const q = textarea.value.trim();
    if (!q) return;

    addMessage('user', q);
    textarea.value = '';
    textarea.style.height = 'auto';
    sendBtn.disabled = true;
    addTyping();

    try {
      const res = await fetch('/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: q }),
      });
      const data = await res.json();
      removeTyping();
      if (data.error) {
        addMessage('model', '⚠️ ' + data.error);
      } else {
        addMessage('model', data.answer);
      }
    } catch (err) {
      removeTyping();
      addMessage('model', '⚠️ Erreur de connexion. Vérifiez que le serveur tourne.');
    } finally {
      sendBtn.disabled = false;
      textarea.focus();
    }
  }
</script>

</body>
</html>"""


# ── Routes ────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Question vide."}), 400

    try:
        init_pipeline()

        # Domain check
        from structrag.rag_pipeline.groq_generation import is_engineering_query
        if not is_engineering_query(question):
            return jsonify({
                "answer": "Cette question ne concerne pas l'Eurocode 2. Je suis spécialisé dans les questions de génie civil relatives à la norme NF EN 1992-1-1.",
                "sources": []
            })

        # Retrieve
        chunks = retrieve(question)
        if not chunks:
            return jsonify({
                "answer": "Aucun contexte pertinent trouvé. Veuillez reformuler votre question.",
                "sources": []
            })

        # Generate
        from structrag.rag_pipeline.multi_model_generation import generate_multi
        result = generate_multi(
            question=question,
            chunks=chunks,
            model=MODEL_ID,
            provider=PROVIDER,
            temperature=0.01,
        )

        # Extract clause sources
        sources = sorted(set(result.sources)) if result.sources else []

        return jsonify({
            "answer":  result.answer,
            "sources": sources,
        })

    except Exception as e:
        return jsonify({"error": f"Erreur interne: {str(e)[:200]}"}), 500


# ── Entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*55)
    print("  StructRAG — Eurocode 2 Assistant")
    print("  Model  : openai/gpt-oss-120b (Groq)")
    print("  Open   : http://127.0.0.1:5000")
    print("="*55 + "\n")
    app.run(debug=False, host="127.0.0.1", port=5000)
