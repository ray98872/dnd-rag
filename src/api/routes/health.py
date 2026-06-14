from fastapi import APIRouter, Request

from src.api.schemas import HealthResponse
from src.config import get_settings

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, summary="Health check")
async def health(request: Request) -> HealthResponse:
    settings = get_settings()
    store = request.app.state.store

    connected = store.is_healthy()
    count = store.count() if connected else 0

    return HealthResponse(
        status="ok" if connected else "degraded",
        qdrant_connected=connected,
        qdrant_collection=settings.qdrant_collection,
        document_count=count,
        model=settings.groq_model,
        embedding_model=settings.embedding_model,
    )
