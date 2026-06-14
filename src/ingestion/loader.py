"""
PDF loader for the D&D 5e 2024 SRD.

Downloads the SRD PDF from D&D Beyond (CC BY 4.0) if not already present,
then extracts text page-by-page with PyMuPDF, preserving section metadata.
"""

from __future__ import annotations

import re
from pathlib import Path

import fitz  # PyMuPDF
import httpx
from langchain_core.documents import Document
from rich.console import Console
from rich.progress import track

console = Console()

# Official SRD 5.2 PDF — CC BY 4.0
SRD_URL = "https://media.dndbeyond.com/compendium-images/free-rules/srd52.pdf"
DEFAULT_PDF_PATH = Path("data/raw/srd52.pdf")

# Regex patterns for D&D section headers
_H1 = re.compile(r"^[A-Z][A-Z\s&']+$")          # ALL CAPS → chapter title
_H2 = re.compile(r"^\*\*(.+?)\*\*$")              # **Bold** → section heading
_SPELL_BLOCK = re.compile(r"^(\w[\w\s]+)\n_(.+?)_", re.MULTILINE)


def download_srd(dest: Path = DEFAULT_PDF_PATH, force: bool = False) -> Path:
    """Download the SRD 5.2 PDF if not already present."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        console.print(f"[green]SRD already at {dest}[/green]")
        return dest

    console.print("[cyan]Downloading SRD 5.2 from D&D Beyond…[/cyan]")
    try:
        with httpx.stream("GET", SRD_URL, follow_redirects=True, timeout=60) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                downloaded = 0
                for chunk in r.iter_bytes(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
            console.print(f"[green]Downloaded {downloaded / 1_048_576:.1f} MB → {dest}[/green]")
    except httpx.HTTPError as e:
        console.print(
            f"[yellow]Auto-download failed ({e}).\n"
            f"Please download the SRD manually from https://www.dndbeyond.com/srd\n"
            f"and place it at: {dest}[/yellow]"
        )
        raise

    return dest


def load_pdf(pdf_path: Path) -> list[Document]:
    """
    Extract text from the SRD PDF page by page.

    Returns a list of Documents — one per page — with metadata:
        - source: filename
        - page: 1-indexed page number
        - section: best-guess section heading for that page
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found at {pdf_path}. "
            "Run `python scripts/ingest.py --download` to fetch it."
        )

    docs: list[Document] = []
    current_section = "Introduction"

    with fitz.open(str(pdf_path)) as pdf:
        console.print(f"[cyan]Loading {len(pdf)} pages from {pdf_path.name}…[/cyan]")
        for page_num in track(range(len(pdf)), description="Extracting pages"):
            page = pdf[page_num]
            text = page.get_text("text").strip()

            if not text:
                continue

            # Update running section tracker from bold/heading lines
            first_lines = text.split("\n")[:5]
            for line in first_lines:
                line = line.strip()
                if _H1.match(line) and len(line) > 3:
                    current_section = line.title()
                    break

            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": pdf_path.name,
                        "page": page_num + 1,
                        "section": current_section,
                    },
                )
            )

    console.print(f"[green]Loaded {len(docs)} pages.[/green]")
    return docs


def load_srd(
    pdf_path: Path = DEFAULT_PDF_PATH,
    auto_download: bool = True,
) -> list[Document]:
    """High-level entry point: download if needed, then load."""
    if not Path(pdf_path).exists() and auto_download:
        download_srd(Path(pdf_path))
    return load_pdf(pdf_path)
