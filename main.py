"""Main orchestration script for the unsupervised topic modeling pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.preprocessing import preprocess_dataset
from src.models import run_topic_modeling


def main() -> None:
    project_root = Path(__file__).resolve().parent
    data_path = project_root / "data" / "BBC news dataset.csv"
    reports_dir = project_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    print("Loading and cleaning dataset")
    df = preprocess_dataset(data_path=data_path)

    print("Training topic models")
    result = run_topic_modeling(df, text_column="cleaned_tokens")

    print("\n=== LDA Topic Keywords ===")
    for idx, keywords in enumerate(result["lda_keywords"], start=1):
        print(f"Topic {idx}: {', '.join(keywords)}")

    print("\n=== NMF Topic Keywords ===")
    for idx, keywords in enumerate(result["nmf_keywords"], start=1):
        print(f"Topic {idx}: {', '.join(keywords)}")

    print(f"\nBest topic count selected: {result['best_topics']}")
    print(f"Reports exported to: {reports_dir}")


if __name__ == "__main__":
    main()
