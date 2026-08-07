"""
hierarchy_schema.py
-------------------
Shared data model for every document type in the StructRAG corpus.

Both standards_parser.py and textbook_parser.py must output Node objects.
Everything downstream — chunker, linker, indexer — depends on this schema,
so it is defined here once and imported everywhere else.

Design decisions:
- dataclass over plain dict: gives you type hints, repr, and easy serialisation
  without the overhead of a full ORM or Pydantic model
- 'type' is a plain string rather than an Enum so the schema stays open to
  new node types (e.g. "annex", "figure") without touching this file
- 'ancestor_path' is stored as a pre-built string (not recomputed on the fly)
  because it will be embedded directly into every chunk's metadata for retrieval
- 'children' is NOT stored here — the tree is assembled by the parser and then
  flattened into a list of Nodes for the chunker.  Keeping Nodes flat makes
  serialisation to JSON trivial.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional
import json


@dataclass
class Node:
    # ------------------------------------------------------------------ #
    # Identity                                                             #
    # ------------------------------------------------------------------ #
    id: str
    """Unique identifier within the whole corpus.
    Convention: <source_id>_<type>_<clause_number_or_index>
    Example:    ec2_2004_en_subclause_6.2.2
    """

    source_id: str
    """Foreign key into source_manifest.json.
    Example: ec2_2004_en
    """

    # ------------------------------------------------------------------ #
    # Structure                                                            #
    # ------------------------------------------------------------------ #
    type: str
    """Node type.  Valid values for standards:
        section   — top-level section  (e.g. "Section 6")
        clause    — X.Y               (e.g. "6.2")
        subclause — X.Y.Z or deeper   (e.g. "6.2.2", "5.8.3.1")
        paragraph — numbered body paragraph  (e.g. "(1)P ...")
        note      — NOTE / NOTE 1 / NOTE 2 blocks
        table     — table node (image or parsed table)
        formula   — formula / equation block
    Valid values for textbooks (added by textbook_parser.py):
        chapter | example | step | given | solution | result
    """

    level: int
    """Nesting depth, 0-indexed.
    section=0, clause=1, subclause=2, sub-subclause=3, paragraph=4
    """

    title: str
    """Heading text without the clause number.
    Example: "Effort tranchant" (not "6.2 Effort tranchant")
    """

    clause_number: Optional[str]
    """The numeric identifier if present, else None.
    Example: "6.2.2"  |  None for a plain paragraph or note
    """

    text: str
    """Full text content of this node (body paragraphs, note text, etc.).
    For section/clause/subclause nodes this holds the *direct* text only,
    not the text of child nodes — children are separate Nodes.
    """

    parent_id: Optional[str]
    """id of the immediate parent Node, or None for root-level sections."""

    ancestor_path: str
    """Human-readable breadcrumb from root to this node.
    Example: "Eurocode 2 > Section 6 > 6.2 Effort tranchant > 6.2.2"
    This is stored in chunk metadata so the retriever can surface it.
    """

    # ------------------------------------------------------------------ #
    # Location                                                             #
    # ------------------------------------------------------------------ #
    page_number: int
    """1-indexed page number in the source PDF."""

    # ------------------------------------------------------------------ #
    # Optional extras                                                      #
    # ------------------------------------------------------------------ #
    cross_refs: list[str] = field(default_factory=list)
    """Clause numbers referenced inside this node's text.
    Example: ["6.2.2", "Table 3.1"]
    Populated by cross_reference_linker.py in Phase 4.
    """

    linked_node_ids: list[str] = field(default_factory=list)
    """IDs of Nodes in other documents that reference or are referenced by
    this node.  Populated by cross_reference_linker.py.
    """

    metadata: dict = field(default_factory=dict)
    """Catch-all for source-specific metadata that does not fit above.
    Example for EC2: {"national_annex": "NF", "paragraph_tag": "(1)P"}
    Example for textbooks: {"example_number": "4.3", "step": 2}
    """

    # ------------------------------------------------------------------ #
    # Serialisation helpers                                                #
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict."""
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Node":
        """Reconstruct a Node from a dict (e.g. loaded from JSON)."""
        return Node(**d)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


# ------------------------------------------------------------------ #
# Convenience: persist / load a list of Nodes                         #
# ------------------------------------------------------------------ #

def save_nodes(nodes: list[Node], path: str) -> None:
    """Write a list of Nodes to a JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump([n.to_dict() for n in nodes], f, ensure_ascii=False, indent=2)
    print(f"Saved {len(nodes)} nodes -> {path}")


def load_nodes(path: str) -> list[Node]:
    """Load a list of Nodes from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [Node.from_dict(d) for d in data]
