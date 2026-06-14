"""
Unit tests for the hybrid retriever and RRF fusion logic.
Qdrant is mocked so these tests run without any running services.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from langchain_core.documents import Document

from src.retrieval.retriever import HybridRetriever, _rrf_fuse
from src.retrieval.vectorstore import SearchResult

# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_result(id_: str, score: float, text: str = "content") -> SearchResult:
    doc = Document(page_content=text, metadata={"section": "Test", "page": 1, "score": score})
    return SearchResult(document=doc, score=score, id=id_)


# ── RRF fusion tests ───────────────────────────────────────────────────────────

class TestRRFFuse:
    def test_deduplication(self):
        """Same doc appearing in both lists should appear only once."""
        dense = [_make_result("a", 0.9), _make_result("b", 0.8)]
        sparse = [_make_result("a", 0.85), _make_result("c", 0.7)]
        fused = _rrf_fuse(dense, sparse, top_k=10)
        ids = [r.id for r in fused]
        assert len(ids) == len(set(ids))  # no duplicates
        assert "a" in ids

    def test_top_k_respected(self):
        dense = [_make_result(str(i), 1.0 - i * 0.1) for i in range(8)]
        sparse = [_make_result(str(i + 4), 1.0 - i * 0.1) for i in range(8)]
        fused = _rrf_fuse(dense, sparse, top_k=5)
        assert len(fused) <= 5

    def test_mutual_top_results_ranked_higher(self):
        """A doc that appears high in BOTH lists should beat one in only one list."""
        # "shared" is rank-1 in both dense and sparse
        dense = [_make_result("shared", 0.99), _make_result("dense_only", 0.98)]
        sparse = [_make_result("shared", 0.99), _make_result("sparse_only", 0.98)]
        fused = _rrf_fuse(dense, sparse, top_k=3)
        assert fused[0].id == "shared"

    def test_rrf_score_added_to_metadata(self):
        dense = [_make_result("a", 0.9)]
        sparse = [_make_result("a", 0.85)]
        fused = _rrf_fuse(dense, sparse, top_k=5)
        assert "rrf_score" in fused[0].document.metadata

    def test_empty_inputs(self):
        assert _rrf_fuse([], [], top_k=5) == []

    def test_one_empty_list(self):
        dense = [_make_result("a", 0.9)]
        fused = _rrf_fuse(dense, [], top_k=5)
        assert len(fused) == 1
        assert fused[0].id == "a"


# ── HybridRetriever tests ──────────────────────────────────────────────────────

class TestHybridRetriever:
    def _make_retriever(self, dense_results, sparse_results):
        store = MagicMock()
        store.search_dense.return_value = dense_results
        store.search_sparse.return_value = sparse_results

        embedder = MagicMock()
        embedder.embed_query.return_value = [0.1] * 384

        settings = MagicMock()
        settings.retrieval_top_k = 10

        return HybridRetriever(store, embedder, settings)

    def test_returns_documents(self):
        dense = [_make_result("a", 0.9, "Fireball does 8d6 damage.")]
        sparse = [_make_result("b", 0.8, "Fireball is a 3rd level spell.")]
        retriever = self._make_retriever(dense, sparse)
        docs = retriever.retrieve("What does Fireball do?")
        assert isinstance(docs, list)
        assert all(isinstance(d, Document) for d in docs)

    def test_calls_both_search_methods(self):
        dense = [_make_result("a", 0.9)]
        sparse = [_make_result("b", 0.8)]
        retriever = self._make_retriever(dense, sparse)
        retriever.retrieve("test query")
        retriever.store.search_dense.assert_called_once()
        retriever.store.search_sparse.assert_called_once()

    def test_deduplicates_results(self):
        shared = _make_result("shared", 0.9)
        retriever = self._make_retriever([shared], [shared])
        docs = retriever.retrieve("query")
        assert len(docs) == 1  # shared result deduplicated

    def test_as_langchain_retriever(self):
        dense = [_make_result("a", 0.9, "content")]
        retriever = self._make_retriever(dense, [])
        lc_retriever = retriever.as_langchain_retriever()
        # Should be callable with get_relevant_documents
        assert hasattr(lc_retriever, "get_relevant_documents") or hasattr(
            lc_retriever, "invoke"
        )
