import json

import pytest


def make_evaluation(model_name, exact_match, f1, latency):
    from question_answering.evaluation import ModelEvaluation

    return ModelEvaluation(
        model_name=model_name,
        count=2,
        exact_match=exact_match,
        f1=f1,
        mean_latency_ms=latency,
        examples=(),
    )


def test_build_comparison_selects_highest_f1_then_lowest_latency():
    from question_answering.reporting import build_comparison

    report = build_comparison(
        (
            make_evaluation("slower", 80.0, 90.0, 20.0),
            make_evaluation("faster", 79.0, 90.0, 10.0),
        ),
        sample_size=2,
        generated_at="2026-08-19T00:00:00Z",
    )

    assert report["dataset"] == "SQuAD v1.1 validation"
    assert report["sample_size"] == 2
    assert report["best_model"] == "faster"
    assert report["models"][0]["model_name"] == "slower"


def test_build_comparison_rejects_inconsistent_sample_counts():
    from question_answering.evaluation import ModelEvaluation
    from question_answering.reporting import build_comparison

    inconsistent = ModelEvaluation("other", 3, 80.0, 80.0, 10.0, ())

    with pytest.raises(ValueError, match="same sample count"):
        build_comparison(
            (make_evaluation("first", 80.0, 80.0, 10.0), inconsistent),
            sample_size=2,
        )


def test_save_comparison_report_creates_readable_json(tmp_path):
    from question_answering.reporting import build_comparison, save_comparison_report

    report = build_comparison(
        (make_evaluation("model", 75.0, 85.0, 12.5),),
        sample_size=2,
        generated_at="2026-08-19T00:00:00Z",
    )
    output = tmp_path / "nested" / "comparison.json"

    save_comparison_report(report, output)

    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["best_model"] == "model"
    assert saved["models"][0]["f1"] == 85.0
