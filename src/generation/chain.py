"""
RAG chain: HyDE → Hybrid Retrieve → Cross-Encoder Rerank → Stream Generate.

The chain is intentionally class-based rather than pure LCEL pipes so
that each step is inspectable and testable in isolation.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from dataclasses import dataclass

import tenacity
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from rich.console import Console

from src.config import Settings
from src.generation.prompts import HYDE_PROMPT, RAG_PROMPT
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.retriever import HybridRetriever

console = Console()


@dataclass
class RAGResult:
    answer: str
    sources: list[Document]
    question: str
    retrieval_query: str       # may differ from question if HyDE was used
    latency_ms: float
    hyde_used: bool


def _format_context(docs: list[Document]) -> str:
    """Render retrieved docs into a numbered context block for the prompt."""
    parts = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        header = f"[{i}] Section: {meta.get('section', 'Unknown')} | Page {meta.get('page', '?')}"
        parts.append(f"{header}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


class RAGChain:
    """
    Full RAG pipeline.

    Steps:
        1. HyDE (optional) — generate a hypothetical SRD passage to improve retrieval
        2. Hybrid retrieval — dense + sparse + RRF
        3. Cross-encoder reranking — re-score top-K to get top-N
        4. Generation — stream answer from Groq/Llama

    All LLM calls include retry logic with exponential back-off to handle
    Groq's free-tier rate limits gracefully.
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        reranker: CrossEncoderReranker,
        settings: Settings,
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.settings = settings
        self.llm = self._build_llm(settings)
        self._hyde_chain = HYDE_PROMPT | self.llm | StrOutputParser()
        self._rag_chain = RAG_PROMPT | self.llm | StrOutputParser()

    # ── LLM factory ──────────────────────────────────────────────────────────

    @staticmethod
    def _build_llm(settings: Settings) -> ChatGroq:
        if settings.llm_backend.value == "groq":
            return ChatGroq(
                api_key=settings.groq_api_key,
                model=settings.groq_model,
                temperature=0.1,
                max_tokens=1024,
                max_retries=settings.groq_max_retries,
            )
        raise NotImplementedError(
            f"LLM backend '{settings.llm_backend}' not yet wired in chain.py. "
            "Add an elif block here."
        )

    # ── Retry wrapper ─────────────────────────────────────────────────────────

    def _retry(self, fn):
        """Apply tenacity retry to any callable to handle rate-limit 429s."""
        return tenacity.retry(
            reraise=True,
            stop=tenacity.stop_after_attempt(self.settings.groq_max_retries),
            wait=tenacity.wait_exponential(
                min=self.settings.groq_retry_min_wait,
                max=self.settings.groq_retry_max_wait,
            ),
            retry=tenacity.retry_if_exception_type(Exception),
            before_sleep=lambda rs: console.print(
                f"[yellow]Rate-limited — retrying in {rs.next_action.sleep:.1f}s "
                f"(attempt {rs.attempt_number})…[/yellow]"
            ),
        )(fn)

    # ── Core pipeline steps ───────────────────────────────────────────────────

    def _hyde_query(self, question: str) -> str:
        """Generate a hypothetical SRD passage and use it as the retrieval query."""
        return self._hyde_chain.invoke({"question": question})

    async def _ahyde_query(self, question: str) -> str:
        return await self._hyde_chain.ainvoke({"question": question})

    def _retrieve_and_rerank(self, retrieval_query: str, original_question: str) -> list[Document]:
        candidates = self.retriever.retrieve(retrieval_query, top_k=self.settings.retrieval_top_k)
        return self.reranker.rerank(
            original_question,  # rerank against the ORIGINAL question, not HyDE
            candidates,
            top_n=self.settings.rerank_top_n,
        )

    # ── Public interface ──────────────────────────────────────────────────────

    def query(self, question: str, use_hyde: bool | None = None) -> RAGResult:
        """Synchronous RAG query."""
        use_hyde = self.settings.use_hyde if use_hyde is None else use_hyde
        t0 = time.perf_counter()

        retrieval_query = self._hyde_query(question) if use_hyde else question
        docs = self._retrieve_and_rerank(retrieval_query, question)
        context = _format_context(docs)
        answer = self._rag_chain.invoke({"question": question, "context": context})

        return RAGResult(
            answer=answer,
            sources=docs,
            question=question,
            retrieval_query=retrieval_query,
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
            hyde_used=use_hyde,
        )

    async def aquery(self, question: str, use_hyde: bool | None = None) -> RAGResult:
        """Async RAG query (non-streaming)."""
        use_hyde = self.settings.use_hyde if use_hyde is None else use_hyde
        t0 = time.perf_counter()

        retrieval_query = await self._ahyde_query(question) if use_hyde else question
        docs = self._retrieve_and_rerank(retrieval_query, question)
        context = _format_context(docs)
        answer = await self._rag_chain.ainvoke({"question": question, "context": context})

        return RAGResult(
            answer=answer,
            sources=docs,
            question=question,
            retrieval_query=retrieval_query,
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
            hyde_used=use_hyde,
        )

    async def astream_query(
        self,
        question: str,
        use_hyde: bool | None = None,
    ) -> AsyncIterator[str]:
        """
        Async streaming RAG query.

        Yields text tokens as they arrive from the LLM.
        Retrieval and reranking happen synchronously first (they're fast),
        then generation streams.

        Usage in FastAPI (SSE):
            async for token in chain.astream_query(q):
                yield f"data: {token}\n\n"
        """
        use_hyde = self.settings.use_hyde if use_hyde is None else use_hyde

        retrieval_query = await self._ahyde_query(question) if use_hyde else question
        docs = self._retrieve_and_rerank(retrieval_query, question)
        context = _format_context(docs)

        stream_chain = RAG_PROMPT | self.llm | StrOutputParser()
        async for token in stream_chain.astream({"question": question, "context": context}):
            yield token
