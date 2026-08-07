# StructRAG — Project Summary

**Research question:** Does a RAG system that understands a document's structure
retrieve better answers than one that doesn't?

---

## What is RAG?

RAG = Retrieval-Augmented Generation.
Instead of asking an LLM a question from memory, you first search a database of
document chunks for relevant passages, then hand those passages to the LLM as
context. The LLM only answers from what it was given — not from hallucination.

The quality of the answer depends entirely on the quality of the retrieved chunks.
That is what this project tests.

---

## The Problem with Standard RAG on Engineering Codes

Eurocode 2 and ACI 318 are not flat text. They are deeply nested:

```
Section 6 — ÉTATS-LIMITES ULTIMES
  └─ 6.2  Effort tranchant
      └─ 6.2.2  Éléments sans armature d'effort tranchant
          ├─ (1)P  La résistance de calcul VRd,c est...
          ├─ (2)P  La valeur de CRd,c est donnée par...
          └─ NOTE  La valeur nationale peut être fournie par...
```

A standard RAG pipeline chops the document into fixed 512-token windows.
The window might start in the middle of clause 6.2.1 and end in the middle of
clause 6.2.2 — losing the heading, losing the applicability conditions, losing
the notes. The LLM gets a fragment and hallucinates the rest.

This project tests whether respecting the document's native structure fixes that.

---

## What Was Built — Phase by Phase

### Phase 1 — Corpus Acquisition

**File: `structrag/data/source_manifest.json`**

Before writing any code, every source document is registered in this file:
- What it is (title, edition, language)
- How it was obtained and its license status
- Whether it is born-digital or scanned
- What parsing strategy it needs

Currently registered:
- **ec2_2004_nf** — NF EN 1992-1-1:2004 (French AFNOR edition, 211 pages) — ACTIVE
- **aci318_19**   — ACI 318-19 — PENDING (not yet acquired)

---

### Phase 2 — Structural Parsing

**File: `structrag/parsing/hierarchy_schema.py`**

Defines the `Node` dataclass — the universal unit shared by every parser.
A Node is one piece of the document. Types:
- `section`    — top-level heading (e.g. "Section 6")
- `clause`     — X.Y level (e.g. "6.2 Effort tranchant")
- `subclause`  — X.Y.Z or deeper (e.g. "6.2.2")
- `paragraph`  — body text, tagged (1)P, (2)P, etc.
- `note`       — NOTE blocks
- `table`      — image/table blocks

Every Node carries:
- `id`             — unique identifier (e.g. "ec2_2004_nf_subclause_6.2.2")
- `clause_number`  — the numeric code (e.g. "6.2.2")
- `ancestor_path`  — full breadcrumb (e.g. "Eurocode 2 > Section 6 > 6.2 > 6.2.2")
- `parent_id`      — link to the parent node
- `page_number`    — where it is in the PDF
- `cross_refs`     — list of other clauses mentioned in the text

Written before any parser so all parsers produce identical output.

---

**File: `structrag/parsing/standards_parser.py`**

Reads the EC2 PDF using PyMuPDF (fitz).
Detects headings by font size + bold flag (discovered by manual PDF inspection):

| Font size | Bold | Meaning |
|-----------|------|---------|
| 12        | yes  | Section heading |
| 11        | yes  | Clause (X.Y) |
| 10        | yes  | Subclause (X.Y.Z or deeper) |
| 10        | no   | Body paragraph |
| 9         | no   | NOTE block |

Uses a state machine to assemble multi-span headings, tracks a parent stack to
build ancestor_path, and extracts cross-references from paragraph text.

**Output: `structrag/data/parsed/ec2_2004_nf_nodes.json`**

2,132 nodes total:
- 12 sections
- 86 clauses
- 275 subclauses
- 1,326 paragraphs
- 266 notes
- 167 table/image blocks
- 569 nodes with cross-references extracted

---

### Phase 3 — Hierarchical Chunking

**File: `structrag/chunking/hierarchical_chunker.py`** ← MAIN CONTRIBUTION

Assembles nodes into retrievable chunks. Each chunk = one complete clause or
subclause with ALL its direct child paragraphs and notes.

The chunk text is structured:
```
[Eurocode 2 > Section 6 > 6.2 Effort tranchant > 6.2.2]
6.2.2  Éléments pour lesquels aucune armature d'effort tranchant n'est requise

(1)P  Pour les membres ne nécessitant pas d'armatures de cisaillement...
(2)P  La résistance au cisaillement de calcul VRd,c est donnée par...
NOTE  La valeur de CRd,c à utiliser dans un pays donné...
```

The ancestor path is prepended to the text so the embedding model captures
the full context — not just the local words.

If a clause exceeds 512 tokens, it splits at paragraph boundaries (never
mid-sentence), and repeats the heading at the top of each continuation chunk.

**Output: `structrag/data/chunks/ec2_hierarchical_chunks.json`**

467 chunks — token stats: min=31, avg=337, max=1739 (before splitting)

---

**File: `structrag/chunking/flat_chunker.py`** ← BASELINE

Naive sliding-window chunking. Takes the same PDF, extracts raw text, slides
a 512-token window with 64-token overlap. No awareness of clause boundaries,
headings, paragraph tags, or cross-references.

**Output: `structrag/data/chunks/ec2_flat_chunks.json`**

315 chunks — every chunk is exactly 512 tokens, many starting mid-sentence.
No clause_number, no ancestor_path metadata.

**Why both:** Same corpus, same embedding model, same token budget. The only
variable is the chunking strategy. Any difference in RAGAS scores is
attributable purely to hierarchy awareness.

---

### Phase 4 — Cross-Reference Linking (partial)

**What a cross-reference is:**
A mention of another clause inside the text of a clause. Example:
> "La valeur de CRd,c est définie en 6.2.3 et dans le Tableau 3.1"

The parser already extracts these automatically and stores them in each node's
`cross_refs` field (e.g. `["6.2.3", "3.1"]`).

**What remains for Phase 4:**
When ACI 318 and textbooks are added, the cross-reference linker will:
1. Detect when a textbook says "per EC2 §6.2.2"
2. Create a bidirectional link between the textbook chunk and the EC2 chunk
3. Store the link in `linked_node_ids` on both nodes

This means the retriever can fetch a normative clause AND its applied
illustration from a textbook together — which is exactly the cross-document
retrieval the project is testing.

File to be built: `structrag/linking/cross_reference_linker.py`

---

### Phase 5 — Indexing and Retrieval

**File: `structrag/retrieval/chroma_client.py`**

Sets up a local persistent ChromaDB vector database with two collections:
- `ec2_hierarchical` — 467 chunks with full metadata
- `ec2_flat`         — 315 chunks with minimal metadata

Both use the `all-MiniLM-L6-v2` embedding model (22M parameters, multilingual,
fast). The model was chosen because it handles French technical text well and
is small enough to run locally without a GPU.

Stored at: `structrag/data/chroma_db/`

---

**File: `structrag/retrieval/indexer.py`**

Reads both chunk JSON files and upserts them into ChromaDB in batches of 100.

Per hierarchical chunk, stored in metadata:
- `clause_number`, `ancestor_path`, `level`, `page_number`, `cross_refs`

Per flat chunk, stored in metadata:
- `start_page`, `end_page` only

This asymmetry is deliberate — the flat chunks have no structural metadata
because the flat chunker never computed it.

---

**File: `structrag/retrieval/retriever.py`**

Query side. Two functions with identical signatures:

```python
retrieve_hierarchical(query, n_results=5, source_id=None, level=None, clause_number=None)
retrieve_flat(query, n_results=5, source_id=None)
```

The hierarchical retriever supports metadata pre-filtering:
- `level=2` — return only subclause-level chunks
- `clause_number="6.2.2"` — exact clause lookup
- `source_id="ec2_2004_nf"` — restrict to one document

The flat retriever cannot filter by clause or level because it has no such metadata.

Each result carries: chunk text, similarity score, clause_number, ancestor_path,
page_number, cross_refs.

---

### Application Layer

**File: `structrag/rag_pipeline/generation.py`**

Builds the LLM prompt from retrieved chunks and calls GPT-4o-mini.

System prompt enforces three rules:
1. Answer ONLY from the provided context passages
2. Always cite the specific clause number and source
3. If context is insufficient, say so — do not guess

These rules make the RAGAS faithfulness score meaningful. An answer that
ignores the context and guesses will score near 0 on faithfulness.

---

**File: `structrag/rag_pipeline/app.py`**

The runnable application. Three modes:

```bash
# Interactive REPL (needs OPENAI_API_KEY)
python structrag/rag_pipeline/app.py

# Single query with LLM answer
python structrag/rag_pipeline/app.py --query "minimum cover for XC2?"

# Retrieval only — no API key needed
python structrag/rag_pipeline/app.py --no-llm --query "shear resistance formula"
```

On each query, runs both pipelines and prints:
- Hierarchical top-N chunks (with clause numbers + ancestor paths)
- Flat top-N chunks (with page ranges only)
- Generated answers from both (if LLM is enabled)

---

**File: `run_pipeline.py`**

Master runner. Executes all 5 steps from one command:

```bash
python run_pipeline.py          # full run
python run_pipeline.py --from 2 # skip parsing, re-use existing nodes
python run_pipeline.py --from 4 # re-index only
python run_pipeline.py --from 5 # smoke test only
```

---

## Current State

```
DONE:
  Phase 1 — source_manifest.json (EC2 registered, ACI 318 placeholder)
  Phase 2 — EC2 parsed into 2,132 nodes
  Phase 3 — 467 hierarchical + 315 flat chunks built
  Phase 5 — both collections indexed in ChromaDB, retrieval working

IN PROGRESS:
  Phase 4 — cross_refs extracted per node; linker not yet built (needs ACI 318)

NOT STARTED:
  Phase 7 — QA benchmark (compliance questions with ground-truth clause answers)
  Phase 8 — RAGAS evaluation (faithfulness, context precision, context recall)
```

---

## What the Smoke Test Showed

Three queries run against both collections. Example result:

**Query:** "What is the minimum concrete cover for a beam in exposure class XC2?"

| Pipeline     | Top result          | Score | What it retrieved |
|--------------|---------------------|-------|-------------------|
| Hierarchical | clause 4.2, page 48 | 0.686 | Conditions environnementales (correct section) |
| Flat         | pages 114–115       | 0.687 | Unknown — no clause label |

The hierarchical retriever already surfaces the right clause with its label.
The flat retriever returns a page range with no way to verify if it is correct.
The RAGAS evaluation will quantify this difference across 50+ benchmark questions.

---

## Next Steps (in order)

1. **Acquire ACI 318** — add to source_manifest.json, run standards_parser.py
2. **Build textbook parser** — `structrag/parsing/textbook_parser.py`
3. **Build cross-reference linker** — `structrag/linking/cross_reference_linker.py`
4. **Build QA benchmark** — 50+ compliance questions with ground-truth clause IDs
5. **Run RAGAS evaluation** — score both pipelines, write the research report
