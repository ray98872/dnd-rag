"""
Hybrid retriever: dense + sparse search fused with Reciprocal Rank Fusion (RRF).

RRF formula: score(d) = Σ 1 / (k + rank(d))
where k=60 is the standard constant that down-weights outlier ranks.
"""

from __future__ import annotations

from collections import defaultdict

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import Settings
from src.retrieval.vectorstore import QdrantStore, SearchResult

_RRF_K = 60


def _rrf_fuse(
    dense_results: list[SearchResult],
    sparse_results: list[SearchResult],
    top_k: int,
) -> list[SearchResult]:
    """
    Fuse two ranked lists with Reciprocal Rank Fusion.
    Returns a deduplicated, re-ranked list of up to top_k results.
    """
    scores: dict[str, float] = defaultdict(float)
    result_map: dict[str, SearchResult] = {}

    for rank, r in enumerate(dense_results):
        scores[r.id] += 1.0 / (_RRF_K + rank + 1)
        result_map[r.id] = r

    for rank, r in enumerate(sparse_results):
        scores[r.id] += 1.0 / (_RRF_K + rank + 1)
        if r.id not in result_map:
            result_map[r.id] = r

    ranked_ids = sorted(scores, key=scores.__getitem__, reverse=True)[:top_k]

    fused: list[SearchResult] = []
    for rid in ranked_ids:
        sr = result_map[rid]
        # Replace raw cosine score with RRF score for transparency
        sr.document.metadata["rrf_score"] = round(scores[rid], 6)
        sr.document.metadata["dense_rank"] = next(
            (i for i, r in enumerate(dense_results) if r.id == rid), -1
        )
        sr.document.metadata["sparse_rank"] = next(
            (i for i, r in enumerate(sparse_results) if r.id == rid), -1
        )
        fused.append(sr)

    return fused


class HybridRetriever:
    """
    Combines dense semantic search with sparse keyword search via RRF.

    Usage:
        retriever = HybridRetriever(store, embedder, settings)
        docs = retriever.retrieve("What does the Fireball spell do?")
    """

    def __init__(
        self,
        store: QdrantStore,
        embedder: HuggingFaceEmbeddings,
        settings: Settings,
    ):
        self.store = store
        self.embedder = embedder
        self.top_k = settings.retrieval_top_k

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict | None = None,
    ) -> list[Document]:
        k = top_k or self.top_k

        # Dense search
        query_vec = self.embedder.embed_query(query)
        dense = self.store.search_dense(query_vec, top_k=k, filters=filters)

        # Sparse search
        sparse = self.store.search_sparse(query, top_k=k, filters=filters)

        # Fuse
        fused = _rrf_fuse(dense, sparse, top_k=k)
        return [r.document for r in fused]

    def as_langchain_retriever(self, **kwargs):
        """Wrap as a LangChain BaseRetriever for use in LCEL chains."""
        from langchain_core.callbacks import CallbackManagerForRetrieverRun
        from langchain_core.retrievers import BaseRetriever

        outer = self

        class _Wrapped(BaseRetriever):
            def _get_relevant_documents(
                self, query: str, *, run_manager: CallbackManagerForRetrieverRun
            ) -> list[Document]:
                return outer.retrieve(query, **kwargs)

        return _Wrapped()
