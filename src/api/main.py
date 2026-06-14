"""
FastAPI application entry point.

All heavy objects (Qdrant client, embedding model, reranker, chain) are
initialised once in the lifespan context and stored on app.state so routes
can access them via Request.app.state without global variables.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from rich.console import Console

from src.api.routes import health, query
from src.config import get_settings
from src.generation.chain import RAGChain
from src.ingestion.embedder import build_embedding_model
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.retriever import HybridRetriever
from src.retrieval.vectorstore import QdrantStore, build_client

console = Console()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise all components at startup; clean up on shutdown."""
    settings = get_settings()
    console.print("[bold cyan]🐉 Starting D&D RAG API…[/bold cyan]")

    # Vector store
    client = build_client(settings)
    store = QdrantStore(client, settings.qdrant_collection)
    app.state.store = store

    # Embedding model (downloads on first run, cached to ~/.cache/huggingface)
    embedder = build_embedding_model(settings)

    # Retriever
    retriever = HybridRetriever(store, embedder, settings)

    # Reranker (downloads ~67 MB model on first run)
    reranker = CrossEncoderReranker(settings)

    # Chain
    chain = RAGChain(retriever, reranker, settings)
    app.state.chain = chain

    console.print(
        f"[green]Ready.[/green] Qdrant: {settings.qdrant_url} | "
        f"Collection: {settings.qdrant_collection} | "
        f"LLM: {settings.groq_model}"
    )
    yield

    # Shutdown
    client.close()
    console.print("[yellow]Shutdown complete.[/yellow]")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.api_title,
        version=settings.api_version,
        description=(
            "Production-grade RAG API for the D&D 5e 2024 SRD. "
            "Hybrid retrieval (dense + sparse) · Cross-encoder reranking · HyDE · SSE streaming."
        ),
        lifespan=lifespan,
    )

    # CORS — open for demo; tighten in production
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(query.router)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    s = get_settings()
    uvicorn.run("src.api.main:app", host=s.api_host, port=s.api_port, reload=True)
