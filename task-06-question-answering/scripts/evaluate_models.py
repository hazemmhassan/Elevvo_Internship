"""Compare local extractive QA models on a fixed SQuAD validation subset."""

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from question_answering.dataset import load_squad_validation
from question_answering.evaluation import evaluate_answerer
from question_answering.model import create_question_answerer
from question_answering.reporting import build_comparison, save_comparison_report


DEFAULT_MODELS = (
    "distilbert/distilbert-base-cased-distilled-squad",
    "deepset/minilm-uncased-squad2",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate local Hugging Face QA models on SQuAD v1.1."
    )
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "model_comparison.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = tuple(args.models) if args.models else DEFAULT_MODELS
    examples = load_squad_validation(args.limit)
    evaluations = []

    for model_name in models:
        print(f"Loading {model_name}...")
        answerer = create_question_answerer(model_name, device=args.device)
        result = evaluate_answerer(answerer, examples)
        evaluations.append(result)
        print(
            f"{model_name}: EM={result.exact_match:.2f}, "
            f"F1={result.f1:.2f}, latency={result.mean_latency_ms:.1f} ms"
        )

    report = build_comparison(evaluations, sample_size=len(examples))
    save_comparison_report(report, args.output)
    print(f"Best model: {report['best_model']}")
    print(f"Report written to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
