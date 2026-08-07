"""
chroma_client.py
----------------
Manages the local persistent ChromaDB instance for StructRAG.

Two collections are created and kept separate:
  - ec2_hierarchical  : chunks from hierarchical_chunker.py
  - ec2_flat          : chunks from flat_chunker.py  (baseline)

This separation is intentional — it lets the RAGAS evaluation query each
collection independently using identical queries, so the only variable is
the chunking strategy.

Why ChromaDB?
- Local persistent storage: no server process needed, data survives restarts.
- Built-in embedding support: we plug in a SentenceTransformer and ChromaDB
  calls it automatically on every upsert and query.
- Metadata filtering: ChromaDB's `where` clause lets us filter by source_id,
  clause_number, level, etc. alongside semantic similarity.

Embedding model choice: "all-MiniLM-L6-v2"
- Fast (22M params), good multilingual quality for technical text.
- The French EC2 text encodes well because the model was trained on
  multilingual data.  For Phase 8 we can swap to a larger model if needed.
"""

from __future__ import annotations
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import chromadb
from chromadb.utils import embedding_functions

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path(r"C:\Users\manef\OneDrive\Desktop\stageesprit")
CHROMA_DB_PATH = str(BASE / "structrag" / "data" / "chroma_db")

# ---------------------------------------------------------------------------
# Collection names
# ---------------------------------------------------------------------------
COLLECTION_HIERARCHICAL = "ec2_hierarchical"
COLLECTION_FLAT         = "ec2_flat"

# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def get_embedding_function():
    """Return a ChromaDB-compatible SentenceTransformer embedding function."""
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )


def get_client() -> chromadb.PersistentClient:
    """Return (or create) the persistent ChromaDB client."""
    Path(CHROMA_DB_PATH).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=CHROMA_DB_PATH)


def get_collection(
    name: str,
    create_if_missing: bool = True,
) -> chromadb.Collection:
    """
    Return a ChromaDB collection by name.

    If create_if_missing=True the collection is created on first call.
    Subsequent calls return the existing collection without re-embedding.
    """
    client = get_client()
    ef = get_embedding_function()

    if create_if_missing:
        return client.get_or_create_collection(
            name=name,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},   # cosine similarity
        )
    else:
        return client.get_collection(name=name, embedding_function=ef)


def list_collections() -> list[str]:
    """Return names of all existing collections."""
    client = get_client()
    return list(client.list_collections())  # v0.6+: returns names directly


def delete_collection(name: str) -> None:
    """Delete a collection (use when re-indexing from scratch)."""
    client = get_client()
    client.delete_collection(name)
    print(f"Deleted collection: {name}")
