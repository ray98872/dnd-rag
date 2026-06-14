"""
Evaluation CLI

Usage:
  python scripts/evaluate.py                        # generate dataset + run eval
  python scripts/evaluate.py --n-samples 20         # smaller/faster run
  python scripts/evaluate.py --skip-generation      # reuse existing dataset
  python scripts/evaluate.py --output-dir my_eval   # custom output directory
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

# ── Compatibility shim ────────────────────────────────────────────────────────
# ragas 0.4.x imports ChatVertexAI from langchain_community.chat_models.vertexai,
# which was removed in langchain-community 0.4.x. Stub it out so the import
# doesn't fail — we never use VertexAI in this project.
if "langchain_community.chat_models.vertexai" not in sys.modules:
    _stub = types.ModuleType("langchain_community.chat_models.vertexai")
    _stub.ChatVertexAI = type("ChatVertexAI", (), {})  # type: ignore[attr-defined]
    sys.modules["langchain_community.chat_models.vertexai"] = _stub
# ─────────────────────────────────────────────────────────────────────────────

import typer
from langchain_groq import ChatGroq
from rich.console import Console

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_settings
from src.eval.dataset import generate_eval_dataset, load_eval_dataset
from src.eval.report import save_report
from src.eval.runner import evaluate_pipeline
from src.generation.chain import RAGChain
from src.ingestion.chunker import chunk_documents
from src.ingestion.embedder import build_embedding_model
from src.ingestion.loader import DEFAULT_PDF_PATH, load_srd
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.retriever import HybridRetriever
from src.retrieval.vectorstore import QdrantStore, build_client

app = typer.Typer(name="evaluate", help="RAGAS evaluation pipeline", add_completion=False)
console = Console()

DATASET_PATH = Path("data/processed/eval_dataset.json")


@app.command()
def main(
    n_samples: int = typer.Option(30, help="Number of eval Q&A pairs to generate"),
    skip_generation: bool = typer.Option(
        False, "--skip-generation", help="Reuse existing eval_dataset.json"
    ),
    pdf: Path = typer.Option(DEFAULT_PDF_PATH, help="SRD PDF path (for dataset generation)"),
    output_dir: Path = typer.Option(Path("eval_results"), help="Directory for report output"),
    throttle: float = typer.Option(2.5, help="Seconds to wait between LLM calls"),
) -> None:
    settings = get_settings()

    console.rule("[bold cyan]D&D RAG — RAGAS Evaluation[/bold cyan]")

    # ── Build shared components ────────────────────────────────────────────────
    embedder = build_embedding_model(settings)
    client = build_client(settings)
    store = QdrantStore(client, settings.qdrant_collection)
    retriever = HybridRetriever(store, embedder, settings)
    reranker = CrossEncoderReranker(settings)
    chain = RAGChain(retriever, reranker, settings)

    llm = ChatGroq(
        api_key=settings.groq_api_key,
        model=settings.groq_model,
        temperature=0,
        max_retries=settings.groq_max_retries,
    )

    # ── 1. Dataset ─────────────────────────────────────────────────────────────
    console.rule("Step 1/3 — Eval Dataset")

    if skip_generation and DATASET_PATH.exists():
        console.print(f"[yellow]Loading existing dataset from {DATASET_PATH}[/yellow]")
        samples = load_eval_dataset(DATASET_PATH)
        console.print(f"[green]Loaded {len(samples)} samples.[/green]")
    else:
        console.print("[cyan]Generating synthetic eval dataset from corpus…[/cyan]")
        pages = load_srd(pdf, auto_download=False)
        chunks = chunk_documents(pages, settings.chunk_size, settings.chunk_overlap)
        samples = generate_eval_dataset(
            chunks, llm, n_samples=n_samples, throttle_seconds=throttle
        )

    # ── 2. Evaluate ────────────────────────────────────────────────────────────
    console.rule("Step 2/3 — Run Evaluation")
    results = evaluate_pipeline(samples, chain, embedder, settings, output_dir)

    # ── 3. Report ──────────────────────────────────────────────────────────────
    console.rule("Step 3/3 — Save Report")
    save_report(results, model=settings.groq_model, output_dir=output_dir)

    console.rule("[bold green]Evaluation Complete[/bold green]")
    scores = results["scores"]
    console.print(
        f"\n  Faithfulness:      [bold]{scores['faithfulness']:.4f}[/bold]\n"
        f"  Answer Relevancy:  [bold]{scores['answer_relevancy']:.4f}[/bold]\n"
        f"  Context Recall:    [bold]{scores['context_recall']:.4f}[/bold]\n"
    )
    console.print(f"Full report → {output_dir}/report.html")


if __name__ == "__main__":
    app()
