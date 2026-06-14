"""
D&D-aware document chunker.

Strategy:
- Use RecursiveCharacterTextSplitter with D&D-specific separators so that
  spell blocks, monster stat blocks, and rule sections stay intact.
- Carry forward page/section metadata into every child chunk.
- Assign a stable chunk_id (sha256 of content) for idempotent re-ingestion.
"""

from __future__ import annotations

import hashlib
import re
import uuid

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rich.console import Console

console = Console()

# Separators ordered from most-preferred to least-preferred split point.
# We try to split at natural D&D content boundaries first.
_DND_SEPARATORS = [
    # Chapter / section headings (ALL CAPS lines)
    r"\n(?=[A-Z][A-Z\s&']{4,}\n)",
    # Spell/monster name lines (Title Case followed by a blank line)
    r"\n\n(?=[A-Z][a-z])",
    # Double newline (paragraph break)
    "\n\n",
    # Table rows
    r"\n\|",
    # Single newline
    "\n",
    # Sentence boundary
    ". ",
    # Word boundary (last resort)
    " ",
    "",
]

# Blocks we try NOT to split mid-way
_SPELL_HEADER = re.compile(
    r"^(?P<name>[A-Z][a-z][\w\s]+)\n"
    r"(?P<level>\d+(?:st|nd|rd|th)-level|\bcantrip\b).*?\n",
    re.MULTILINE,
)


def _chunk_id(text: str) -> str:
    """Return a deterministic UUID derived from the chunk content."""
    hex32 = hashlib.sha256(text.encode()).hexdigest()[:32]
    return str(uuid.UUID(hex32))


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[Document]:
    """
    Split page-level documents into retrieval-sized chunks.

    Each chunk inherits the source, page, and section metadata from its
    parent document and gains a unique chunk_id.
    """
    splitter = RecursiveCharacterTextSplitter(
        separators=_DND_SEPARATORS,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        is_separator_regex=True,
        keep_separator=False,
        add_start_index=True,
    )

    chunks: list[Document] = []

    for doc in documents:
        splits = splitter.split_documents([doc])
        for split in splits:
            text = split.page_content.strip()
            if len(text) < 30:  # skip near-empty chunks
                continue
            split.metadata["chunk_id"] = _chunk_id(text)
            split.metadata["char_count"] = len(text)
            chunks.append(split)

    console.print(
        f"[green]Chunked {len(documents)} pages → {len(chunks)} chunks "
        f"(avg {sum(c.metadata['char_count'] for c in chunks) // max(len(chunks), 1)} chars)[/green]"
    )
    return chunks


def deduplicate(chunks: list[Document]) -> list[Document]:
    """Remove duplicate chunks by chunk_id (safe for re-runs)."""
    seen: set[str] = set()
    unique: list[Document] = []
    for chunk in chunks:
        cid = chunk.metadata.get("chunk_id", "")
        if cid not in seen:
            seen.add(cid)
            unique.append(chunk)
    removed = len(chunks) - len(unique)
    if removed:
        console.print(f"[yellow]Deduplication removed {removed} duplicate chunks.[/yellow]")
    return unique
