"""Reproducible JSON reporting for QA model comparisons."""

from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Sequence

from question_answering.evaluation import ModelEvaluation


def build_comparison(
    evaluations: Sequence[ModelEvaluation],
    *,
    sample_size: int,
    generated_at: str | None = None,
) -> dict[str, object]:
    """Create a comparable report and select the best model by F1/latency."""
    if not evaluations:
        raise ValueError("at least one model evaluation is required")
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if any(evaluation.count != sample_size for evaluation in evaluations):
        raise ValueError("all evaluations must use the same sample count")

    best = max(
        evaluations,
        key=lambda evaluation: (evaluation.f1, -evaluation.mean_latency_ms),
    )
    timestamp = generated_at or datetime.now(UTC).isoformat().replace("+00:00", "Z")
    return {
        "dataset": "SQuAD v1.1 validation",
        "sample_size": sample_size,
        "generated_at": timestamp,
        "ranking_rule": "highest F1, then lowest mean latency",
        "best_model": best.model_name,
        "models": [evaluation.to_dict() for evaluation in evaluations],
    }


def save_comparison_report(report: dict[str, object], path: Path) -> None:
    """Write one formatted UTF-8 comparison report, creating parents."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
