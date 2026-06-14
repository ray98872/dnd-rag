"""
Query endpoints.

POST /query          — standard JSON response
POST /query/stream   — Server-Sent Events (SSE) streaming response
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from src.api.schemas import (
    QueryRequest,
    QueryResponse,
    SourceDocument,
    StreamEndEvent,
    StreamStartEvent,
    StreamTokenEvent,
)

router = APIRouter(tags=["query"])


def _doc_to_source(doc) -> SourceDocument:
    m = doc.metadata
    return SourceDocument(
        content=doc.page_content,
        section=m.get("section", ""),
        page=m.get("page", 0),
        source=m.get("source", ""),
        rerank_score=m.get("rerank_score"),
        rrf_score=m.get("rrf_score"),
    )


@router.post("/query", response_model=QueryResponse, summary="Ask the SRD a question")
async def query(body: QueryRequest, request: Request) -> QueryResponse:
    """
    Non-streaming RAG query. Returns the full answer + source documents.
    """
    chain = request.app.state.chain
    result = await chain.aquery(body.question, use_hyde=body.use_hyde)

    return QueryResponse(
        answer=result.answer,
        sources=[_doc_to_source(d) for d in result.sources],
        question=result.question,
        retrieval_query=result.retrieval_query,
        latency_ms=result.latency_ms,
        hyde_used=result.hyde_used,
    )


@router.post("/query/stream", summary="Streaming SSE query")
async def query_stream(body: QueryRequest, request: Request) -> StreamingResponse:
    """
    Streaming RAG query via Server-Sent Events.

    Event sequence:
        1. {type: "start", sources: [...], question: "...", hyde_used: bool}
        2. {type: "token", content: "..."} × N
        3. {type: "end", latency_ms: 123}

    In JavaScript: const es = new EventSource('/query/stream', {method: 'POST', ...})
    """
    chain = request.app.state.chain

    async def event_generator():
        t0 = time.perf_counter()
        use_hyde = body.use_hyde

        # Run retrieval upfront (non-streaming) so we can send sources immediately
        retrieval_query = (
            await chain._ahyde_query(body.question) if use_hyde else body.question
        )
        docs = chain._retrieve_and_rerank(retrieval_query, body.question)
        sources = [_doc_to_source(d) for d in docs]

        # Event 1: sources available
        start_event = StreamStartEvent(
            sources=sources,
            question=body.question,
            hyde_used=use_hyde,
        )
        yield f"data: {start_event.model_dump_json()}\n\n"

        # Event 2…N: stream tokens
        from langchain_core.output_parsers import StrOutputParser

        from src.generation.prompts import RAG_PROMPT

        context_text = "\n\n---\n\n".join(
            f"[{i+1}] {d.page_content}" for i, d in enumerate(docs)
        )
        stream_chain = RAG_PROMPT | chain.llm | StrOutputParser()

        async for token in stream_chain.astream(
            {"question": body.question, "context": context_text}
        ):
            tok_event = StreamTokenEvent(content=token)
            yield f"data: {tok_event.model_dump_json()}\n\n"

        # Event last: done
        end_event = StreamEndEvent(latency_ms=round((time.perf_counter() - t0) * 1000, 1))
        yield f"data: {end_event.model_dump_json()}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable Nginx buffering for SSE
        },
    )
