"""Local Hugging Face extractive question-answering model adapter."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal


DEFAULT_MODEL = "deepset/minilm-uncased-squad2"
DevicePreference = Literal["auto", "cpu", "cuda"]


@dataclass(frozen=True)
class QAPrediction:
    """One answer extracted directly from the supplied context."""

    answer: str
    score: float
    start: int
    end: int
    model_name: str


@dataclass(frozen=True)
class TokenSpan:
    """Best valid context-token span for one encoded feature."""

    start: int
    end: int
    start_token: int
    end_token: int
    score: float


class QuestionAnswerer:
    """Validated boundary around a Transformers question-answering pipeline."""

    def __init__(
        self,
        backend: Callable[..., Mapping[str, Any]],
        *,
        model_name: str,
    ) -> None:
        self._backend = backend
        self.model_name = model_name

    def answer(self, question: str, context: str) -> QAPrediction:
        """Extract an answer span for one question and context."""
        if not question.strip():
            raise ValueError("question must not be blank")
        if not context.strip():
            raise ValueError("context must not be blank")

        raw = self._backend(question=question.strip(), context=context)
        try:
            start = int(raw["start"])
            end = int(raw["end"])
            score = float(raw["score"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError("model returned an invalid prediction") from error

        if start < 0 or end <= start or end > len(context):
            raise RuntimeError("model returned an invalid answer span")

        return QAPrediction(
            answer=context[start:end],
            score=score,
            start=start,
            end=end,
            model_name=self.model_name,
        )


def resolve_pipeline_device(
    preference: DevicePreference,
    *,
    cuda_available: bool,
) -> int:
    """Translate a user-friendly device choice into a pipeline device index."""
    if preference not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")
    if preference == "cuda" and not cuda_available:
        raise RuntimeError("CUDA was requested but is not available")
    if preference == "cpu":
        return -1
    return 0 if cuda_available else -1


def select_best_span(
    *,
    start_logits: Sequence[float],
    end_logits: Sequence[float],
    offsets: Sequence[Sequence[int]],
    sequence_ids: Sequence[int | None],
    max_answer_tokens: int,
) -> TokenSpan:
    """Select the highest-logit span composed only of context tokens."""
    if max_answer_tokens <= 0:
        raise ValueError("max_answer_tokens must be positive")
    lengths = {len(start_logits), len(end_logits), len(offsets), len(sequence_ids)}
    if len(lengths) != 1:
        raise ValueError("logits, offsets, and sequence IDs must have equal length")

    best: TokenSpan | None = None
    for start_token, start_logit in enumerate(start_logits):
        if sequence_ids[start_token] != 1:
            continue
        last_token = min(len(end_logits), start_token + max_answer_tokens)
        for end_token in range(start_token, last_token):
            if sequence_ids[end_token] != 1:
                break
            start_char = int(offsets[start_token][0])
            end_char = int(offsets[end_token][1])
            if end_char <= start_char:
                continue
            score = float(start_logit) + float(end_logits[end_token])
            if best is None or score > best.score:
                best = TokenSpan(
                    start=start_char,
                    end=end_char,
                    start_token=start_token,
                    end_token=end_token,
                    score=score,
                )

    if best is None:
        raise RuntimeError("model feature contained no valid context span")
    return best


class TransformersQABackend:
    """Direct tokenizer/model inference for extractive QA in Transformers 5."""

    def __init__(
        self,
        tokenizer: Any,
        model: Any,
        *,
        device: Any,
        max_length: int = 384,
        stride: int = 128,
        max_answer_tokens: int = 30,
    ) -> None:
        self._tokenizer = tokenizer
        self._model = model
        self._device = device
        self._max_length = max_length
        self._stride = stride
        self._max_answer_tokens = max_answer_tokens

    def __call__(self, *, question: str, context: str) -> Mapping[str, object]:
        import torch

        encoded = self._tokenizer(
            question,
            context,
            max_length=self._max_length,
            truncation="only_second",
            stride=self._stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding=True,
            return_tensors="pt",
        )
        offsets = encoded.pop("offset_mapping")
        encoded.pop("overflow_to_sample_mapping", None)
        sequence_ids = [encoded.sequence_ids(index) for index in range(len(offsets))]
        model_inputs = {name: value.to(self._device) for name, value in encoded.items()}

        with torch.inference_mode():
            output = self._model(**model_inputs)

        best_span: TokenSpan | None = None
        best_feature = -1
        for feature_index in range(len(offsets)):
            candidate = select_best_span(
                start_logits=output.start_logits[feature_index].tolist(),
                end_logits=output.end_logits[feature_index].tolist(),
                offsets=offsets[feature_index].tolist(),
                sequence_ids=sequence_ids[feature_index],
                max_answer_tokens=self._max_answer_tokens,
            )
            if best_span is None or candidate.score > best_span.score:
                best_span = candidate
                best_feature = feature_index

        if best_span is None:
            raise RuntimeError("model produced no valid context answer")

        start_probability = torch.softmax(output.start_logits[best_feature], dim=-1)[
            best_span.start_token
        ].item()
        end_probability = torch.softmax(output.end_logits[best_feature], dim=-1)[
            best_span.end_token
        ].item()
        return {
            "answer": context[best_span.start : best_span.end],
            "score": start_probability * end_probability,
            "start": best_span.start,
            "end": best_span.end,
        }


def create_question_answerer(
    model_name: str = DEFAULT_MODEL,
    *,
    device: DevicePreference = "auto",
) -> QuestionAnswerer:
    """Download/cache a Hugging Face model and build the local QA adapter."""
    import torch
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer

    pipeline_device = resolve_pipeline_device(
        device,
        cuda_available=torch.cuda.is_available(),
    )
    torch_device = torch.device("cuda:0" if pipeline_device == 0 else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    model.to(torch_device)
    model.eval()
    backend = TransformersQABackend(tokenizer, model, device=torch_device)
    return QuestionAnswerer(backend, model_name=model_name)
