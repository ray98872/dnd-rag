"""
Central configuration — all settings flow from here.
Swap the entire stack (Groq vs Azure OpenAI, Qdrant vs Azure AI Search)
by changing values in .env without touching any other file.
"""

import sys
from functools import lru_cache

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:  # pragma: no cover
    from enum import Enum as _Enum  # type: ignore[assignment]
    class StrEnum(str, _Enum):  # type: ignore[no-redef]
        pass

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMBackend(StrEnum):
    groq = "groq"
    azure_openai = "azure_openai"


class VectorBackend(StrEnum):
    qdrant = "qdrant"
    azure_ai_search = "azure_ai_search"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM backend
    llm_backend: LLMBackend = LLMBackend.groq

    # Groq
    groq_api_key: str = Field(default="", description="Groq API key")
    groq_model: str = "llama-3.3-70b-versatile"

    # Azure OpenAI (optional swap)
    azure_openai_api_key: str = Field(default="", description="Azure OpenAI API key")
    azure_openai_endpoint: str = ""
    azure_openai_chat_deployment: str = "gpt-4o"
    azure_openai_embedding_deployment: str = "text-embedding-3-small"
    azure_openai_api_version: str = "2024-02-01"

    # Embeddings
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_device: str = "cpu"
    embedding_batch_size: int = 64

    # Vector store
    vector_backend: VectorBackend = VectorBackend.qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "dnd_srd"

    # RAG parameters
    chunk_size: int = 512
    chunk_overlap: int = 64
    retrieval_top_k: int = 10
    rerank_top_n: int = 4
    use_hyde: bool = True

    # Reranker
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_title: str = "D&D 5e 2024 RAG API"
    api_version: str = "0.1.0"

    # LangSmith (optional tracing)
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "dnd-rag"

    # Groq rate-limit retry
    groq_max_retries: int = 5
    groq_retry_min_wait: float = 1.0
    groq_retry_max_wait: float = 30.0

    @model_validator(mode="after")
    def _validate_backends(self) -> "Settings":
        if self.llm_backend == LLMBackend.groq and not self.groq_api_key:
            raise ValueError("GROQ_API_KEY is required when llm_backend=groq")
        if self.llm_backend == LLMBackend.azure_openai and not self.azure_openai_api_key:
            raise ValueError("AZURE_OPENAI_API_KEY is required when llm_backend=azure_openai")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
