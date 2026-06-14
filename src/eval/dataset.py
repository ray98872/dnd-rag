"""
Synthetic evaluation dataset generator.

Samples chunks from the corpus and uses the LLM to generate realistic
D&D rules questions + ground-truth answers. This gives us a labelled
dataset to pass to RAGAS without needing manually annotated data.

Rate limiting: Groq free tier allows ~500K tokens/day at 6K tokens/min.
We throttle generation to stay within limits.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import TypedDict

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from rich.console import Console
from rich.progress import track

console = Console()

_QA_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a D&D 5e 2024 rules expert. Given an excerpt from the SRD, "
        "generate ONE specific, answerable question and its concise ground-truth answer. "
        "The question should be something a player or DM would genuinely ask. "
        "Respond in JSON only, with keys: 'question' and 'ground_truth'.",
    ),
    ("human", "SRD excerpt:\n\n{context}"),
])


class EvalSample(TypedDict):
    question: str
    ground_truth: str
    context_chunk: str
    source: str
    page: int


def generate_eval_dataset(
    chunks: list[Document],
    llm: ChatGroq,
    n_samples: int = 50,
    seed: int = 42,
    throttle_seconds: float = 2.0,
) -> list[EvalSample]:
    """
    Generate n_samples question/ground-truth pairs from sampled corpus chunks.

    Args:
        chunks:           All ingested document chunks.
        llm:              Groq LLM (shared with the chain to save memory).
        n_samples:        How many eval pairs to generate.
        seed:             For reproducibility.
        throttle_seconds: Pause between LLM calls to avoid rate limits.

    Returns:
        List of EvalSample dicts, also written to data/processed/eval_dataset.json.
    """
    random.seed(seed)

    # Filter out very short chunks that won't produce good questions
    valid_chunks = [c for c in chunks if len(c.page_content) >= 200]
    if len(valid_chunks) < n_samples:
        console.print(
            f"[yellow]Only {len(valid_chunks)} valid chunks — "
            f"reducing n_samples to {len(valid_chunks)}.[/yellow]"
        )
        n_samples = len(valid_chunks)

    sampled = random.sample(valid_chunks, n_samples)
    chain = _QA_PROMPT | llm | StrOutputParser()

    samples: list[EvalSample] = []
    failed = 0

    for chunk in track(sampled, description="Generating eval Q&A pairs"):
        try:
            raw = chain.invoke({"context": chunk.page_content[:800]})
            # Extract JSON even if wrapped in markdown fences
            raw = raw.strip().lstrip("```json").rstrip("```").strip()
            parsed = json.loads(raw)
            samples.append(
                EvalSample(
                    question=parsed["question"],
                    ground_truth=parsed["ground_truth"],
                    context_chunk=chunk.page_content,
                    source=chunk.metadata.get("source", ""),
                    page=chunk.metadata.get("page", 0),
                )
            )
        except Exception as e:
            console.print(f"[red]Failed to parse Q&A for chunk: {e}[/red]")
            failed += 1

        time.sleep(throttle_seconds)

    console.print(
        f"[green]Generated {len(samples)} eval samples "
        f"({failed} failed).[/green]"
    )

    # Persist
    out_path = Path("data/processed/eval_dataset.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    console.print(f"[cyan]Saved eval dataset → {out_path}[/cyan]")

    return samples


def load_eval_dataset(path: Path = Path("data/processed/eval_dataset.json")) -> list[EvalSample]:
    """Load a previously generated eval dataset from disk."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)
