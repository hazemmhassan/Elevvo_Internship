import pytest


VALID_RECORD = {
    "id": "q1",
    "question": "Where is the Eiffel Tower?",
    "context": "The Eiffel Tower is in Paris, France.",
    "answers": {"text": ["Paris"], "answer_start": [23]},
}


def test_parse_squad_record_returns_validated_example():
    from question_answering.dataset import parse_squad_record

    example = parse_squad_record(VALID_RECORD)

    assert example.id == "q1"
    assert example.question == "Where is the Eiffel Tower?"
    assert example.context == "The Eiffel Tower is in Paris, France."
    assert example.answers == ("Paris",)
    assert example.answer_starts == (23,)


@pytest.mark.parametrize("field", ["id", "question", "context"])
def test_parse_squad_record_rejects_blank_required_text(field):
    from question_answering.dataset import parse_squad_record

    record = dict(VALID_RECORD)
    record[field] = "  "

    with pytest.raises(ValueError, match=field):
        parse_squad_record(record)


def test_parse_squad_record_rejects_mismatched_answer_lists():
    from question_answering.dataset import parse_squad_record

    record = {**VALID_RECORD, "answers": {"text": ["Paris"], "answer_start": []}}

    with pytest.raises(ValueError, match="same non-zero length"):
        parse_squad_record(record)


def test_parse_squad_record_rejects_answer_span_that_does_not_match_context():
    from question_answering.dataset import parse_squad_record

    record = {**VALID_RECORD, "answers": {"text": ["Paris"], "answer_start": [0]}}

    with pytest.raises(ValueError, match="does not match context"):
        parse_squad_record(record)


def test_load_squad_validation_limits_and_parses_records():
    from question_answering.dataset import load_squad_validation

    def fake_loader(name, *, split):
        assert name == "rajpurkar/squad"
        assert split == "validation"
        return [VALID_RECORD, {**VALID_RECORD, "id": "q2"}]

    examples = load_squad_validation(limit=1, dataset_loader=fake_loader)

    assert tuple(example.id for example in examples) == ("q1",)


def test_load_squad_validation_rejects_non_positive_limit():
    from question_answering.dataset import load_squad_validation

    with pytest.raises(ValueError, match="limit must be positive"):
        load_squad_validation(limit=0, dataset_loader=lambda *args, **kwargs: [])
