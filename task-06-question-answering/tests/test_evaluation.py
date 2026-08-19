from dataclasses import dataclass

import pytest


@dataclass(frozen=True)
class FakePrediction:
    answer: str
    score: float
    start: int
    end: int
    model_name: str


class FakeAnswerer:
    model_name = "fake-model"

    def answer(self, question, context):
        answers = {
            "Capital of France?": "Paris",
            "Largest city?": "New York",
        }
        answer = answers[question]
        start = context.index(answer)
        return FakePrediction(answer, 0.9, start, start + len(answer), self.model_name)


def test_evaluate_answerer_reports_metrics_latency_and_examples():
    from question_answering.dataset import SquadExample
    from question_answering.evaluation import evaluate_answerer

    examples = (
        SquadExample("q1", "Capital of France?", "Paris is in France.", ("Paris",), (0,)),
        SquadExample(
            "q2",
            "Largest city?",
            "New York is a large city.",
            ("New York City",),
            (0,),
        ),
    )
    times = iter((10.0, 10.2, 20.0, 20.4))

    result = evaluate_answerer(FakeAnswerer(), examples, clock=lambda: next(times))

    assert result.model_name == "fake-model"
    assert result.count == 2
    assert result.exact_match == 50.0
    assert result.f1 == pytest.approx(90.0)
    assert result.mean_latency_ms == pytest.approx(300.0)
    assert result.examples[0].prediction == "Paris"


def test_evaluate_answerer_rejects_empty_dataset():
    from question_answering.evaluation import evaluate_answerer

    with pytest.raises(ValueError, match="at least one example"):
        evaluate_answerer(FakeAnswerer(), ())
