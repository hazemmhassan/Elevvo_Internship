"""Transparent Exact Match and token-level F1 metrics for SQuAD answers."""

from collections import Counter
from dataclasses import dataclass
import re
import string
from typing import Mapping, Sequence


@dataclass(frozen=True)
class AnswerScore:
    """Scores for one prediction against one or more references."""

    exact_match: float
    f1: float


@dataclass(frozen=True)
class EvaluationMetrics:
    """Aggregate SQuAD metrics expressed as percentages."""

    exact_match: float
    f1: float
    count: int


def normalize_answer(text: str) -> str:
    """Apply the official SQuAD-style answer normalization steps."""
    lowered = text.lower()
    no_punctuation = "".join(char for char in lowered if char not in string.punctuation)
    no_articles = re.sub(r"\b(a|an|the)\b", " ", no_punctuation)
    return " ".join(no_articles.split())


def exact_match_score(prediction: str, reference: str) -> float:
    """Return 1 when normalized answers match, otherwise 0."""
    return float(normalize_answer(prediction) == normalize_answer(reference))


def token_f1_score(prediction: str, reference: str) -> float:
    """Calculate token overlap F1 after SQuAD normalization."""
    prediction_tokens = normalize_answer(prediction).split()
    reference_tokens = normalize_answer(reference).split()

    if not prediction_tokens or not reference_tokens:
        return float(prediction_tokens == reference_tokens)

    overlap = Counter(prediction_tokens) & Counter(reference_tokens)
    matching_tokens = sum(overlap.values())
    if matching_tokens == 0:
        return 0.0

    precision = matching_tokens / len(prediction_tokens)
    recall = matching_tokens / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def score_prediction(prediction: str, references: Sequence[str]) -> AnswerScore:
    """Use the best score across all accepted SQuAD reference answers."""
    if not references:
        raise ValueError("references must contain at least one answer")

    return AnswerScore(
        exact_match=max(exact_match_score(prediction, answer) for answer in references),
        f1=max(token_f1_score(prediction, answer) for answer in references),
    )


def evaluate_predictions(
    predictions: Mapping[str, str],
    references: Mapping[str, Sequence[str]],
) -> EvaluationMetrics:
    """Aggregate predictions into percentage Exact Match and F1 scores."""
    if set(predictions) != set(references):
        raise ValueError("predictions and references must contain the same question IDs")
    if not predictions:
        raise ValueError("predictions must not be empty")

    scores = [
        score_prediction(predictions[question_id], references[question_id])
        for question_id in predictions
    ]
    count = len(scores)
    return EvaluationMetrics(
        exact_match=100 * sum(score.exact_match for score in scores) / count,
        f1=100 * sum(score.f1 for score in scores) / count,
        count=count,
    )
