"""
Unit tests for the chunker module.
No external dependencies required — pure Python.
"""

from langchain_core.documents import Document

from src.ingestion.chunker import _chunk_id, chunk_documents, deduplicate

# ── Fixtures ───────────────────────────────────────────────────────────────────

def make_doc(text: str, page: int = 1, section: str = "Rules") -> Document:
    return Document(
        page_content=text,
        metadata={"source": "srd52.pdf", "page": page, "section": section},
    )


FIREBALL_TEXT = """\
Fireball
3rd-level evocation

Casting Time: 1 action
Range: 150 feet
Components: V, S, M (a tiny ball of bat guano and sulfur)
Duration: Instantaneous

A bright streak flashes from your pointing finger to a point you choose within range
and then blossoms with a low roar into an explosion of flame. Each creature in a
20-foot-radius sphere centered on that point must make a Dexterity saving throw.
A target takes 8d6 fire damage on a failed save, or half as much damage on a successful one.

The fire spreads around corners. It ignites flammable objects in the area that aren't
being worn or carried. At Higher Levels. When you cast this spell using a spell slot
of 4th level or higher, the damage increases by 1d6 for each slot level above 3rd.
"""


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestChunkId:
    def test_deterministic(self):
        assert _chunk_id("hello") == _chunk_id("hello")

    def test_unique_on_different_content(self):
        assert _chunk_id("foo") != _chunk_id("bar")

    def test_length(self):
        assert len(_chunk_id("any text")) == 16


class TestChunkDocuments:
    def test_returns_documents(self):
        doc = make_doc(FIREBALL_TEXT * 3)  # make it long enough to chunk
        chunks = chunk_documents([doc], chunk_size=256, chunk_overlap=32)
        assert len(chunks) > 0
        assert all(isinstance(c, Document) for c in chunks)

    def test_metadata_inherited(self):
        doc = make_doc(FIREBALL_TEXT * 3, page=42, section="Spells")
        chunks = chunk_documents([doc], chunk_size=256, chunk_overlap=32)
        for chunk in chunks:
            assert chunk.metadata["source"] == "srd52.pdf"
            assert chunk.metadata["page"] == 42
            assert chunk.metadata["section"] == "Spells"

    def test_chunk_id_assigned(self):
        doc = make_doc(FIREBALL_TEXT * 3)
        chunks = chunk_documents([doc], chunk_size=256, chunk_overlap=32)
        for chunk in chunks:
            assert "chunk_id" in chunk.metadata
            assert len(chunk.metadata["chunk_id"]) == 16

    def test_min_length_filter(self):
        """Very short text should produce no chunks."""
        doc = make_doc("Short.")
        chunks = chunk_documents([doc], chunk_size=256, chunk_overlap=32)
        assert len(chunks) == 0

    def test_char_count_metadata(self):
        doc = make_doc(FIREBALL_TEXT * 3)
        chunks = chunk_documents([doc], chunk_size=256, chunk_overlap=32)
        for chunk in chunks:
            assert chunk.metadata["char_count"] == len(chunk.page_content)

    def test_multiple_documents(self):
        docs = [make_doc(FIREBALL_TEXT * 2, page=i) for i in range(1, 4)]
        chunks = chunk_documents(docs, chunk_size=256, chunk_overlap=32)
        pages = {c.metadata["page"] for c in chunks}
        assert len(pages) >= 1  # at least one page represented


class TestDeduplicate:
    def test_removes_duplicates(self):
        doc = make_doc(FIREBALL_TEXT * 3)
        chunks = chunk_documents([doc, doc], chunk_size=256, chunk_overlap=32)
        unique = deduplicate(chunks)
        assert len(unique) < len(chunks)

    def test_idempotent(self):
        doc = make_doc(FIREBALL_TEXT * 3)
        chunks = chunk_documents([doc], chunk_size=256, chunk_overlap=32)
        once = deduplicate(chunks)
        twice = deduplicate(once)
        assert len(once) == len(twice)
