"""
Cross-encoder reranker.

Takes the top-K candidates from hybrid retrieval and reorders them by
fine-grained relevance to the original query. This is the most impactful
single quality improvement in a RAG pipeline.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (local, free, ~67 MB)
"""

from __future__ import annotations

import numpy as np
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from src.config import Settings


class CrossEncoderReranker:
    """
    Reranks documents using a cross-encoder relevance model.

    Unlike bi-encoders (used for retrieval), cross-encoders see both the
    query and the document simultaneously, giving much more accurate scores
    at the cost of being too slow for full-corpus search.
    """

    def __init__(self, settings: Settings):
        self._model_name = settings.reranker_model
        self._model: CrossEncoder | None = None  # lazy load

    @property
    def model(self) -> CrossEncoder:
        if self._model is None:
            self._model = CrossEncoder(self._model_name, max_length=512)
        return self._model

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_n: int | None = None,
    ) -> list[Document]:
        """
        Rerank documents and return top_n most relevant.

        Adds 'rerank_score' to each document's metadata.
        """
        if not documents:
            return []

        pairs = [[query, doc.page_content] for doc in documents]
        scores: np.ndarray = self.model.predict(pairs, show_progress_bar=False)

        scored = sorted(
            zip(scores.tolist(), documents),
            key=lambda x: x[0],
            reverse=True,
        )

        result: list[Document] = []
        for score, doc in scored[: top_n or len(scored)]:
            doc.metadata["rerank_score"] = round(float(score), 4)
            result.append(doc)

        return result
