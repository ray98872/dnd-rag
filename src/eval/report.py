"""
Evaluation report generator.

Produces:
  - eval_results/scores.json          — machine-readable summary
  - eval_results/report.html          — human-readable report with a score dashboard
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console

console = Console()

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>D&D RAG — RAGAS Evaluation Report</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: system-ui, sans-serif; background: #0f1117; color: #e2e8f0; padding: 2rem; }}
  h1 {{ font-size: 1.8rem; margin-bottom: 0.25rem; color: #f8fafc; }}
  .subtitle {{ color: #94a3b8; margin-bottom: 2rem; font-size: 0.9rem; }}
  .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1rem; margin-bottom: 2rem; }}
  .card {{ background: #1e2130; border-radius: 12px; padding: 1.5rem; border: 1px solid #2d3748; }}
  .card h3 {{ font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; color: #94a3b8; margin-bottom: 0.5rem; }}
  .card .score {{ font-size: 2.5rem; font-weight: 700; color: {score_color}; }}
  .bar-bg {{ height: 6px; background: #2d3748; border-radius: 3px; margin-top: 0.75rem; }}
  .bar-fill {{ height: 6px; border-radius: 3px; background: linear-gradient(90deg, #6366f1, #8b5cf6); }}
  table {{ width: 100%; border-collapse: collapse; background: #1e2130; border-radius: 12px; overflow: hidden; }}
  th {{ background: #2d3748; padding: 0.75rem 1rem; text-align: left; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.06em; color: #94a3b8; }}
  td {{ padding: 0.75rem 1rem; border-bottom: 1px solid #2d3748; font-size: 0.85rem; vertical-align: top; }}
  tr:last-child td {{ border-bottom: none; }}
  .q {{ color: #a5b4fc; font-weight: 500; }}
  .a {{ color: #e2e8f0; }}
  .gt {{ color: #86efac; }}
  .tag {{ display: inline-block; background: #2d3748; padding: 0.15rem 0.5rem; border-radius: 4px; font-size: 0.7rem; margin-right: 0.25rem; }}
</style>
</head>
<body>
<h1>🐉 D&D 5e 2024 SRD — RAG Evaluation Report</h1>
<p class="subtitle">Generated {timestamp} &nbsp;·&nbsp; {n_samples} samples &nbsp;·&nbsp; Model: {model}</p>

<div class="metrics">
  <div class="card">
    <h3>Faithfulness</h3>
    <div class="score">{faithfulness:.2f}</div>
    <div class="bar-bg"><div class="bar-fill" style="width:{faithfulness_pct}%"></div></div>
  </div>
  <div class="card">
    <h3>Answer Relevancy</h3>
    <div class="score">{answer_relevancy:.2f}</div>
    <div class="bar-bg"><div class="bar-fill" style="width:{answer_relevancy_pct}%"></div></div>
  </div>
  <div class="card">
    <h3>Context Recall</h3>
    <div class="score">{context_recall:.2f}</div>
    <div class="bar-bg"><div class="bar-fill" style="width:{context_recall_pct}%"></div></div>
  </div>
</div>

<table>
<thead>
<tr>
  <th>#</th>
  <th>Question</th>
  <th>Generated Answer</th>
  <th>Ground Truth</th>
  <th>Context chunks</th>
</tr>
</thead>
<tbody>
{rows}
</tbody>
</table>
</body>
</html>
"""

_ROW_TEMPLATE = """\
<tr>
  <td>{i}</td>
  <td class="q">{question}</td>
  <td class="a">{answer}</td>
  <td class="gt">{ground_truth}</td>
  <td><span class="tag">{n_contexts} chunks</span></td>
</tr>
"""


def _avg_score(scores: dict[str, float]) -> float:
    vals = [v for v in scores.values() if v is not None]
    return sum(vals) / len(vals) if vals else 0.0


def save_report(
    results: dict[str, Any],
    model: str,
    output_dir: Path = Path("eval_results"),
) -> tuple[Path, Path]:
    """
    Save scores.json and report.html.

    Returns (json_path, html_path).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    scores = results["scores"]
    rows_data = results["rows"]
    n_samples = results["n_samples"]
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    # JSON
    json_payload = {
        "timestamp": ts,
        "model": model,
        "n_samples": n_samples,
        "scores": scores,
    }
    json_path = output_dir / "scores.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2)

    # HTML
    avg = _avg_score(scores)
    score_color = "#22c55e" if avg >= 0.7 else "#f59e0b" if avg >= 0.5 else "#ef4444"
    html_rows = "".join(
        _ROW_TEMPLATE.format(
            i=i + 1,
            question=row["question"].replace("<", "&lt;"),
            answer=(row["answer"] or "")[:300].replace("<", "&lt;"),
            ground_truth=row["ground_truth"].replace("<", "&lt;"),
            n_contexts=len(row.get("contexts", [])),
        )
        for i, row in enumerate(rows_data)
    )

    html = _HTML_TEMPLATE.format(
        timestamp=ts,
        n_samples=n_samples,
        model=model,
        faithfulness=scores["faithfulness"],
        faithfulness_pct=round(scores["faithfulness"] * 100),
        answer_relevancy=scores["answer_relevancy"],
        answer_relevancy_pct=round(scores["answer_relevancy"] * 100),
        context_recall=scores["context_recall"],
        context_recall_pct=round(scores["context_recall"] * 100),
        score_color=score_color,
        rows=html_rows,
    )
    html_path = output_dir / "report.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    console.print("[green]Report saved:[/green]")
    console.print(f"  JSON → {json_path}")
    console.print(f"  HTML → {html_path}")
    return json_path, html_path
