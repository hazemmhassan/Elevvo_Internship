"""Run the checked-in retrieval benchmark against a persisted FAISS index."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from talent_search.embeddings import (  # noqa: E402
    DEFAULT_EMBEDDING_MODEL,
    create_local_embeddings,
)
from talent_search.evaluation import (  # noqa: E402
    evaluate_retrieval_case,
    load_retrieval_cases,
    summarize_retrieval_results,
)
from talent_search.retrieval import retrieve_candidates_hybrid  # noqa: E402
from talent_search.vector_store import load_faiss_index  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate candidate retrieval without exposing resume text."
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=PROJECT_ROOT / ".artifacts" / "faiss_index",
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "retrieval_cases.jsonl",
    )
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("-k", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    embeddings = create_local_embeddings(args.model, device=args.device)
    vector_store = load_faiss_index(
        args.index,
        embeddings,
        expected_embedding_model_name=args.model,
    )
    cases = load_retrieval_cases(args.cases)
    results = [
        evaluate_retrieval_case(
            case,
            retrieve_candidates_hybrid(
                vector_store,
                case.query,
                candidate_k=args.k,
            ),
            k=args.k,
        )
        for case in cases
    ]
    summary = summarize_retrieval_results(results)
    print(json.dumps(asdict(summary), indent=2))


if __name__ == "__main__":
    main()
