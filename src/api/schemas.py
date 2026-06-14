"""
Pydantic request/response models for the FastAPI layer.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000, description="Your D&D rules question")
    use_hyde: bool = Field(True, description="Use Hypothetical Document Embeddings for retrieval")
    filters: dict[str, str] | None = Field(
        None,
        description="Optional metadata filters, e.g. {\"section\": \"Spells\"}",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "question": "What does the Fireball spell do and what is its damage?",
                    "use_hyde": True,
                    "filters": None,
                }
            ]
        }
    }


class SourceDocument(BaseModel):
    content: str
    section: str
    page: int
    source: str
    rerank_score: float | None = None
    rrf_score: float | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceDocument]
    question: str
    retrieval_query: str
    latency_ms: float
    hyde_used: bool


class StreamStartEvent(BaseModel):
    """Sent as the first SSE event so clients know retrieval is done."""
    type: str = "start"
    sources: list[SourceDocument]
    question: str
    hyde_used: bool


class StreamTokenEvent(BaseModel):
    type: str = "token"
    content: str


class StreamEndEvent(BaseModel):
    type: str = "end"
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    qdrant_connected: bool
    qdrant_collection: str
    document_count: int
    model: str
    embedding_model: str
