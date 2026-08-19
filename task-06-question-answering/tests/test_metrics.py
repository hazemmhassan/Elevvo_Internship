import pytest


def test_normalize_answer_removes_articles_punctuation_and_extra_space():
    from question_answering.metrics import normalize_answer

    assert normalize_answer("  The, QUICK!  fox ") == "quick fox"


def test_exact_match_accepts_equivalent_normalized_answers():
    from question_answering.metrics import exact_match_score

    assert exact_match_score("The Nile.", "nile") == 1.0
    assert exact_match_score("Nile River", "Nile") == 0.0


def test_token_f1_gives_partial_credit_for_overlapping_answer_tokens():
    from question_answering.metrics import token_f1_score

    assert token_f1_score("New York", "New York City") == pytest.approx(0.8)


def test_token_f1_handles_empty_answers_without_dividing_by_zero():
    from question_answering.metrics import token_f1_score

    assert token_f1_score("", "") == 1.0
    assert token_f1_score("", "answer") == 0.0


def test_score_prediction_uses_best_valid_reference_answer():
    from question_answering.metrics import score_prediction

    score = score_prediction("NYC", ("New York City", "NYC"))

    assert score.exact_match == 1.0
    assert score.f1 == 1.0


def test_evaluate_predictions_returns_percentage_metrics():
    from question_answering.metrics import evaluate_predictions

    result = evaluate_predictions(
        predictions={"q1": "Paris", "q2": "New York"},
        references={"q1": ("Paris",), "q2": ("New York City",)},
    )

    assert result.count == 2
    assert result.exact_match == 50.0
    assert result.f1 == pytest.approx(90.0)


def test_evaluate_predictions_rejects_missing_or_extra_ids():
    from question_answering.metrics import evaluate_predictions

    with pytest.raises(ValueError, match="same question IDs"):
        evaluate_predictions(
            predictions={"q1": "Paris"},
            references={"q2": ("Paris",)},
        )
