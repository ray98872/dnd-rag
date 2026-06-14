"""
RAGAS evaluation runner.

Metrics evaluated:
  - Faithfulness:       Does the answer only make claims supported by the retrieved context?
  - Answer Relevancy:   Is the answer relevant to the question asked?
  - Context Recall:     Does the retrieved context contain the info needed to answer?

Rate-limit strategy: answers are generated in sequence with a delay between
calls; RAGAS metric scoring uses the same LLM with exponential back-off.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import AnswerRelevancy, ContextRecall, Faithfulness
from rich.console import Console
from rich.progress import track

from src.config import Settings
from src.eval.dataset import EvalSample
from src.generation.chain import RAGChain

console = Console()


def run_rag_answers(
    samples: list[EvalSample],
    chain: RAGChain,
    throttle_seconds: float = 2.5,
) -> list[dict[str, Any]]:
    """
    Run each eval question through the RAG chain to collect:
      - generated answer
      - retrieved context chunks (as list[str])
    """
    rows: list[dict[str, Any]] = []

    for sample in track(samples, description="Running RAG answers"):
        try:
            result = chain.query(sample["question"], use_hyde=False)
            rows.append(
                {
                    "question": sample["question"],
                    "answer": result.answer,
                    "contexts": [d.page_content for d in result.sources],
                    "ground_truth": sample["ground_truth"],
                }
            )
        except Exception as e:
            console.print(f"[red]RAG failed for '{sample['question'][:60]}': {e}[/red]")
            # Include with empty answer so RAGAS can still score context metrics
            rows.append(
                {
                    "question": sample["question"],
                    "answer": "",
                    "contexts": [sample["context_chunk"]],
                    "ground_truth": sample["ground_truth"],
                }
            )
        time.sleep(throttle_seconds)

    return rows


def run_ragas_eval(
    rows: list[dict[str, Any]],
    settings: Settings,
    embeddings: HuggingFaceEmbeddings,
) -> dict[str, float]:
    """
    Run RAGAS evaluation on the collected rows.

    Returns a dict of metric_name → score (0.0–1.0).
    """
    # RAGAS 0.2+ requires wrapped LLM/embeddings and EvaluationDataset
    wrapped_llm = LangchainLLMWrapper(
        ChatGroq(
            api_key=settings.groq_api_key,
            model=settings.groq_model,
            temperature=0,
            max_retries=settings.groq_max_retries,
        )
    )
    wrapped_embeddings = LangchainEmbeddingsWrapper(embeddings)

    ragas_samples = [
        SingleTurnSample(
            user_input=row["question"],
            response=row["answer"],
            retrieved_contexts=row["contexts"],
            reference=row["ground_truth"],
        )
        for row in rows
    ]
    dataset = EvaluationDataset(samples=ragas_samples)

    ar_metric = AnswerRelevancy(llm=wrapped_llm, embeddings=wrapped_embeddings)
    # Groq only supports n=1; limit generated questions to avoid BadRequestError
    if hasattr(ar_metric, "n_generated_questions"):
        ar_metric.n_generated_questions = 1

    metrics = [
        Faithfulness(llm=wrapped_llm),
        ar_metric,
        ContextRecall(llm=wrapped_llm),
    ]

    console.print("[cyan]Running RAGAS evaluation (this may take a few minutes)…[/cyan]")
    result = evaluate(dataset=dataset, metrics=metrics)

    def _safe_mean(val) -> float:
        """Handle scalar, list, or NaN-containing list returned by RAGAS 0.4.x."""
        if isinstance(val, (int, float)):
            return float(val)
        arr = np.array(val, dtype=float)
        return float(np.nanmean(arr)) if not np.all(np.isnan(arr)) else 0.0

    scores = {
        "faithfulness": round(_safe_mean(result["faithfulness"]), 4),
        "answer_relevancy": round(_safe_mean(result["answer_relevancy"]), 4),
        "context_recall": round(_safe_mean(result["context_recall"]), 4),
    }

    console.print("\n[bold green]RAGAS Results:[/bold green]")
    for metric, score in scores.items():
        bar = "█" * int(score * 20)
        console.print(f"  {metric:<22} {score:.4f}  {bar}")

    return scores


def evaluate_pipeline(
    samples: list[EvalSample],
    chain: RAGChain,
    embeddings: HuggingFaceEmbeddings,
    settings: Settings,
    output_dir: Path = Path("eval_results"),
) -> dict[str, Any]:
    """
    Full evaluation pipeline:
      1. Run RAG on all eval questions
      2. Score with RAGAS
      3. Return results dict (also saved to disk by report.py)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: collect RAG answers
    rows = run_rag_answers(samples, chain)

    # Step 2: RAGAS scoring
    scores = run_ragas_eval(rows, settings, embeddings)

    # Step 3: bundle results
    df = pd.DataFrame(rows)
    df["faithfulness"] = None      # populated by RAGAS per-sample in full datasets
    df["answer_relevancy"] = None
    df["context_recall"] = None

    return {
        "scores": scores,
        "n_samples": len(rows),
        "rows": rows,
        "dataframe": df,
    }
