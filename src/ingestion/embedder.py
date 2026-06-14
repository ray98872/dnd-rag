"""
Embedding pipeline — encodes chunks and upserts them into Qdrant.

Dense vectors: BAAI/bge-small-en-v1.5 (384-dim, local, free)
Sparse vectors: BM25 via FastEmbed (for hybrid search in Qdrant)
"""

from __future__ import annotations

import uuid
from typing import Any

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
)
from rich.console import Console
from rich.progress import track

from src.config import Settings

console = Console()

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


def build_embedding_model(settings: Settings) -> HuggingFaceEmbeddings:
    """Load the local sentence-transformer embedding model."""
    console.print(f"[cyan]Loading embedding model: {settings.embedding_model}[/cyan]")
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": settings.embedding_device},
        encode_kwargs={"normalize_embeddings": True, "batch_size": settings.embedding_batch_size},
    )


def ensure_collection(client: QdrantClient, collection_name: str, vector_size: int = 384) -> None:
    """Create the Qdrant collection with dense + sparse vectors if it doesn't exist."""
    existing = {c.name for c in client.get_collections().collections}
    if collection_name in existing:
        console.print(f"[yellow]Collection '{collection_name}' already exists — skipping creation.[/yellow]")
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            DENSE_VECTOR_NAME: VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
                on_disk=False,
            )
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: SparseVectorParams(
                index=SparseIndexParams(on_disk=False)
            )
        },
    )
    console.print(f"[green]Created collection '{collection_name}' (dense={vector_size}d + sparse BM25).[/green]")


def _build_sparse_vector(text: str) -> dict[str, Any]:
    """
    Compute a simple TF-based sparse vector for BM25-style retrieval.
    For production, swap this with fastembed's SparseTextEmbedding('Qdrant/bm25').
    This implementation avoids the fastembed model download during testing.
    """
    from collections import Counter

    tokens = text.lower().split()
    counts = Counter(tokens)
    total = len(tokens)
    if total == 0:
        return {"indices": [], "values": []}

    # Accumulate TF for tokens that hash-collide to the same index.
    # Qdrant requires all indices to be unique within a sparse vector.
    idx_to_value: dict[int, float] = {}
    for token, count in counts.items():
        idx = abs(hash(token)) % 100_000
        tf = count / total
        idx_to_value[idx] = idx_to_value.get(idx, 0.0) + tf

    return {"indices": list(idx_to_value.keys()), "values": list(idx_to_value.values())}


def embed_and_upsert(
    chunks: list[Document],
    client: QdrantClient,
    embedding_model: HuggingFaceEmbeddings,
    collection_name: str,
    batch_size: int = 64,
) -> int:
    """
    Embed all chunks and upsert into Qdrant in batches.
    Returns the number of points upserted.
    """
    texts = [c.page_content for c in chunks]
    total = 0

    console.print(f"[cyan]Embedding and upserting {len(chunks)} chunks…[/cyan]")

    for i in track(range(0, len(chunks), batch_size), description="Upserting batches"):
        batch_chunks = chunks[i : i + batch_size]
        batch_texts = texts[i : i + batch_size]

        # Dense embeddings
        dense_vecs = embedding_model.embed_documents(batch_texts)

        points: list[PointStruct] = []
        for chunk, dense_vec, text in zip(batch_chunks, dense_vecs, batch_texts):
            sparse_vec = _build_sparse_vector(text)
            point = PointStruct(
                id=chunk.metadata.get("chunk_id") or str(uuid.uuid4()),
                vector={
                    DENSE_VECTOR_NAME: dense_vec,
                    SPARSE_VECTOR_NAME: sparse_vec,
                },
                payload={
                    "text": chunk.page_content,
                    "source": chunk.metadata.get("source", ""),
                    "page": chunk.metadata.get("page", 0),
                    "section": chunk.metadata.get("section", ""),
                    "char_count": chunk.metadata.get("char_count", 0),
                },
            )
            points.append(point)

        client.upsert(collection_name=collection_name, points=points, wait=True)
        total += len(points)

    console.print(f"[green]Upserted {total} points into '{collection_name}'.[/green]")
    return total
