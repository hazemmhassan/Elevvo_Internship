"""Model evaluation over validated SQuAD examples."""

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Callable, Protocol, Sequence

from question_answering.dataset import SquadExample
from question_answering.metrics import evaluate_predictions, score_prediction
from question_answering.model import QAPrediction


class Answerer(Protocol):
    """Small interface implemented by local QA model adapters."""

    model_name: str

    def answer(self, question: str, context: str) -> QAPrediction: ...


@dataclass(frozen=True)
class EvaluatedExample:
    """Auditable prediction and per-example scores."""

    id: str
    question: str
    prediction: str
    references: tuple[str, ...]
    confidence: float
    exact_match: float
    f1: float
    latency_ms: float


@dataclass(frozen=True)
class ModelEvaluation:
    """Aggregate benchmark results plus auditable example details."""

    model_name: str
    count: int
    exact_match: float
    f1: float
    mean_latency_ms: float
    examples: tuple[EvaluatedExample, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return asdict(self)


def evaluate_answerer(
    answerer: Answerer,
    examples: Sequence[SquadExample],
    *,
    clock: Callable[[], float] = perf_counter,
) -> ModelEvaluation:
    """Run a model over examples and calculate SQuAD EM/F1 plus latency."""
    if not examples:
        raise ValueError("evaluation requires at least one example")

    predictions: dict[str, str] = {}
    references: dict[str, tuple[str, ...]] = {}
    evaluated: list[EvaluatedExample] = []

    for example in examples:
        started = clock()
        prediction = answerer.answer(example.question, example.context)
        latency_ms = (clock() - started) * 1000
        score = score_prediction(prediction.answer, example.answers)
        predictions[example.id] = prediction.answer
        references[example.id] = example.answers
        evaluated.append(
            EvaluatedExample(
                id=example.id,
                question=example.question,
                prediction=prediction.answer,
                references=example.answers,
                confidence=prediction.score,
                exact_match=score.exact_match,
                f1=score.f1,
                latency_ms=latency_ms,
            )
        )

    metrics = evaluate_predictions(predictions, references)
    return ModelEvaluation(
        model_name=answerer.model_name,
        count=metrics.count,
        exact_match=metrics.exact_match,
        f1=metrics.f1,
        mean_latency_ms=sum(item.latency_ms for item in evaluated) / len(evaluated),
        examples=tuple(evaluated),
    )
