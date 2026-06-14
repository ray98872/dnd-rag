"""
Qdrant client wrapper.

Provides typed search methods for dense and sparse retrieval.
All results are returned as LangChain Documents so the rest of the
pipeline stays framework-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
    NamedSparseVector,
    NamedVector,
    ScoredPoint,
    SparseVector,
)

from src.config import Settings
from src.ingestion.embedder import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME, _build_sparse_vector

_PAYLOAD_TEXT_KEY = "text"


@dataclass
class SearchResult:
    document: Document
    score: float
    id: str


def _point_to_result(point: ScoredPoint) -> SearchResult:
    payload = point.payload or {}
    doc = Document(
        page_content=payload.get(_PAYLOAD_TEXT_KEY, ""),
        metadata={
            "source": payload.get("source", ""),
            "page": payload.get("page", 0),
            "section": payload.get("section", ""),
            "score": point.score,
        },
    )
    return SearchResult(document=doc, score=point.score, id=str(point.id))


def build_client(settings: Settings) -> QdrantClient:
    kwargs: dict = {"url": settings.qdrant_url}
    if settings.qdrant_api_key:
        kwargs["api_key"] = settings.qdrant_api_key
    return QdrantClient(**kwargs)


class QdrantStore:
    def __init__(self, client: QdrantClient, collection: str):
        self.client = client
        self.collection = collection

    def search_dense(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        qdrant_filter = _build_filter(filters)
        response = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            using=DENSE_VECTOR_NAME,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        )
        return [_point_to_result(r) for r in response.points]

    def search_sparse(
        self,
        query_text: str,
        top_k: int = 10,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        sparse = _build_sparse_vector(query_text)
        if not sparse["indices"]:
            return []
        qdrant_filter = _build_filter(filters)
        response = self.client.query_points(
            collection_name=self.collection,
            query=SparseVector(indices=sparse["indices"], values=sparse["values"]),
            using=SPARSE_VECTOR_NAME,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        )
        return [_point_to_result(r) for r in response.points]

    def is_healthy(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False

    def count(self) -> int:
        try:
            info = self.client.get_collection(self.collection)
            return info.points_count or 0
        except Exception:
            return 0


def _build_filter(filters: dict | None) -> Filter | None:
    if not filters:
        return None
    conditions = [
        FieldCondition(key=k, match=MatchValue(value=v))
        for k, v in filters.items()
    ]
    return Filter(must=conditions)
