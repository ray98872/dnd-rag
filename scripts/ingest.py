"""
Ingestion CLI

Usage:
  python scripts/ingest.py                          # auto-download + ingest
  python scripts/ingest.py --pdf data/raw/srd52.pdf # use local PDF
  python scripts/ingest.py --download-only           # just download, no ingest
  python scripts/ingest.py --reset                   # drop collection and re-ingest
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer
from rich.console import Console

# Ensure project root is on the path when run as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_settings
from src.ingestion.chunker import chunk_documents, deduplicate
from src.ingestion.embedder import build_embedding_model, embed_and_upsert, ensure_collection
from src.ingestion.loader import DEFAULT_PDF_PATH, download_srd, load_srd
from src.retrieval.vectorstore import build_client

app = typer.Typer(name="ingest", help="D&D SRD ingestion pipeline", add_completion=False)
console = Console()


@app.command()
def main(
    pdf: Path = typer.Option(DEFAULT_PDF_PATH, help="Path to the SRD PDF"),
    download_only: bool = typer.Option(False, "--download-only", help="Just download, don't ingest"),
    reset: bool = typer.Option(False, "--reset", help="Drop and recreate the Qdrant collection"),
    chunk_size: int = typer.Option(None, help="Override chunk size from config"),
    chunk_overlap: int = typer.Option(None, help="Override chunk overlap from config"),
    batch_size: int = typer.Option(64, help="Embedding batch size"),
) -> None:
    settings = get_settings()

    # Override config if CLI args provided
    if chunk_size:
        settings.__dict__["chunk_size"] = chunk_size
    if chunk_overlap:
        settings.__dict__["chunk_overlap"] = chunk_overlap

    console.rule("[bold cyan]D&D SRD Ingestion Pipeline[/bold cyan]")

    # ── 1. Download ────────────────────────────────────────────────────────────
    if not Path(pdf).exists():
        console.print(f"[yellow]PDF not found at {pdf} — attempting download…[/yellow]")
        try:
            download_srd(Path(pdf))
        except Exception as e:
            console.print(
                f"[red]Download failed: {e}\n\n"
                "Please download the SRD manually:\n"
                "  1. Go to https://www.dndbeyond.com/srd\n"
                "  2. Click 'Download PDF'\n"
                f"  3. Save as {pdf}[/red]"
            )
            raise typer.Exit(code=1)

    if download_only:
        console.print("[green]Download complete. Exiting (--download-only).[/green]")
        raise typer.Exit()

    # ── 2. Load pages ──────────────────────────────────────────────────────────
    console.rule("Step 1/4 — Load PDF")
    pages = load_srd(pdf, auto_download=False)

    # ── 3. Chunk ───────────────────────────────────────────────────────────────
    console.rule("Step 2/4 — Chunk")
    chunks = chunk_documents(pages, settings.chunk_size, settings.chunk_overlap)
    chunks = deduplicate(chunks)

    # ── 4. Connect to Qdrant ───────────────────────────────────────────────────
    console.rule("Step 3/4 — Prepare Qdrant")
    client = build_client(settings)

    if reset:
        console.print(f"[yellow]--reset: deleting collection '{settings.qdrant_collection}'…[/yellow]")
        try:
            client.delete_collection(settings.qdrant_collection)
            console.print("[green]Collection deleted.[/green]")
        except Exception:
            pass

    ensure_collection(client, settings.qdrant_collection, vector_size=384)

    # ── 5. Embed + upsert ──────────────────────────────────────────────────────
    console.rule("Step 4/4 — Embed & Upsert")
    embedder = build_embedding_model(settings)
    n = embed_and_upsert(chunks, client, embedder, settings.qdrant_collection, batch_size)

    console.rule("[bold green]Done[/bold green]")
    console.print(
        f"[bold green]✓ Ingestion complete:[/bold green] "
        f"{len(pages)} pages → {len(chunks)} chunks → {n} vectors in Qdrant"
    )


if __name__ == "__main__":
    app()
