"""SQuAD loading and structural validation."""

from dataclasses import dataclass
from itertools import islice
from typing import Callable, Iterable, Mapping


@dataclass(frozen=True)
class SquadExample:
    """One validated extractive question-answering example."""

    id: str
    question: str
    context: str
    answers: tuple[str, ...]
    answer_starts: tuple[int, ...]


def parse_squad_record(record: Mapping[str, object]) -> SquadExample:
    """Validate a Hugging Face SQuAD record and return a typed example."""
    required_text: dict[str, str] = {}
    for field in ("id", "question", "context"):
        value = record.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} must be a non-blank string")
        required_text[field] = value.strip() if field != "context" else value

    raw_answers = record.get("answers")
    if not isinstance(raw_answers, Mapping):
        raise ValueError("answers must contain text and answer_start lists")
    answer_texts = raw_answers.get("text")
    answer_starts = raw_answers.get("answer_start")
    if (
        not isinstance(answer_texts, (list, tuple))
        or not isinstance(answer_starts, (list, tuple))
        or len(answer_texts) == 0
        or len(answer_texts) != len(answer_starts)
    ):
        raise ValueError("answer text and start lists must have the same non-zero length")

    context = required_text["context"]
    validated_answers: list[str] = []
    validated_starts: list[int] = []
    for answer, start in zip(answer_texts, answer_starts, strict=True):
        if not isinstance(answer, str) or not answer:
            raise ValueError("answer text must be non-empty")
        if not isinstance(start, int) or start < 0:
            raise ValueError("answer_start must be a non-negative integer")
        if context[start : start + len(answer)] != answer:
            raise ValueError("answer span does not match context")
        validated_answers.append(answer)
        validated_starts.append(start)

    return SquadExample(
        id=required_text["id"],
        question=required_text["question"],
        context=context,
        answers=tuple(validated_answers),
        answer_starts=tuple(validated_starts),
    )


def load_squad_validation(
    limit: int,
    *,
    dataset_loader: Callable[..., Iterable[Mapping[str, object]]] | None = None,
) -> tuple[SquadExample, ...]:
    """Load and validate the first ``limit`` SQuAD validation examples."""
    if limit <= 0:
        raise ValueError("limit must be positive")

    if dataset_loader is None:
        from datasets import load_dataset

        dataset_loader = load_dataset

    records = dataset_loader("rajpurkar/squad", split="validation")
    return tuple(parse_squad_record(record) for record in islice(records, limit))
