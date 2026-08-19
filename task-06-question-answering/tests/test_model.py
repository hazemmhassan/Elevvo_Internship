import pytest


def test_question_answerer_returns_context_span_not_backend_rewrite():
    from question_answering.model import QuestionAnswerer

    def backend(*, question, context):
        assert question == "Where?"
        assert context == "It happened in Cairo."
        return {"answer": "cairo", "score": 0.91, "start": 15, "end": 20}

    prediction = QuestionAnswerer(backend, model_name="fake-model").answer(
        "Where?", "It happened in Cairo."
    )

    assert prediction.answer == "Cairo"
    assert prediction.score == pytest.approx(0.91)
    assert prediction.start == 15
    assert prediction.end == 20
    assert prediction.model_name == "fake-model"


@pytest.mark.parametrize(
    ("question", "context", "message"),
    [("", "Context", "question"), ("Question", "  ", "context")],
)
def test_question_answerer_rejects_blank_inputs(question, context, message):
    from question_answering.model import QuestionAnswerer

    answerer = QuestionAnswerer(lambda **kwargs: {}, model_name="fake-model")

    with pytest.raises(ValueError, match=message):
        answerer.answer(question, context)


def test_question_answerer_rejects_invalid_backend_span():
    from question_answering.model import QuestionAnswerer

    answerer = QuestionAnswerer(
        lambda **kwargs: {"answer": "bad", "score": 0.2, "start": 50, "end": 60},
        model_name="fake-model",
    )

    with pytest.raises(RuntimeError, match="invalid answer span"):
        answerer.answer("Question", "Short context")


@pytest.mark.parametrize(
    ("preference", "cuda_available", "expected"),
    [("auto", True, 0), ("auto", False, -1), ("cpu", True, -1), ("cuda", True, 0)],
)
def test_resolve_pipeline_device(preference, cuda_available, expected):
    from question_answering.model import resolve_pipeline_device

    assert resolve_pipeline_device(preference, cuda_available=cuda_available) == expected


def test_resolve_pipeline_device_rejects_unavailable_cuda():
    from question_answering.model import resolve_pipeline_device

    with pytest.raises(RuntimeError, match="CUDA was requested"):
        resolve_pipeline_device("cuda", cuda_available=False)


def test_select_best_span_ignores_question_and_special_tokens():
    from question_answering.model import select_best_span

    span = select_best_span(
        start_logits=[100.0, 50.0, 2.0, 8.0, 100.0],
        end_logits=[100.0, 50.0, 1.0, 7.0, 100.0],
        offsets=[(0, 0), (0, 5), (0, 2), (3, 8), (0, 0)],
        sequence_ids=[None, 0, 1, 1, None],
        max_answer_tokens=3,
    )

    assert span.start == 3
    assert span.end == 8
    assert span.score == 15.0


def test_select_best_span_honors_maximum_answer_length():
    from question_answering.model import select_best_span

    span = select_best_span(
        start_logits=[9.0, 1.0, 1.0],
        end_logits=[1.0, 1.0, 9.0],
        offsets=[(0, 3), (4, 7), (8, 13)],
        sequence_ids=[1, 1, 1],
        max_answer_tokens=2,
    )

    assert span.end_token - span.start_token + 1 <= 2


def test_select_best_span_rejects_features_without_context_tokens():
    from question_answering.model import select_best_span

    with pytest.raises(RuntimeError, match="no valid context span"):
        select_best_span(
            start_logits=[1.0, 2.0],
            end_logits=[1.0, 2.0],
            offsets=[(0, 0), (0, 4)],
            sequence_ids=[None, 0],
            max_answer_tokens=2,
        )
