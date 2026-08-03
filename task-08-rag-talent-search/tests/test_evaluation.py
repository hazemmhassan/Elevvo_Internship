from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
import pytest

from talent_search.evaluation import (
    RelevantCandidate,
    RetrievalEvaluationCase,
    evaluate_retrieval_case,
    load_retrieval_cases,
    summarize_retrieval_results,
)
from talent_search.retrieval import CandidateMatch, ChunkEvidence


def candidate_match(
    candidate_id: str,
    score: float,
    *sections: str,
) -> CandidateMatch:
    evidence = tuple(
        ChunkEvidence(
            document=Document(
                page_content=f"{section}\nPrivacy-safe evidence",
                metadata={
                    "candidate_id": candidate_id,
                    "section": section,
                    "privacy_masked": True,
                },
            ),
            score=score - (index * 0.01),
        )
        for index, section in enumerate(sections)
    )
    return CandidateMatch(candidate_id=candidate_id, score=score, evidence=evidence)


def test_evaluation_measures_candidate_rank_and_expected_evidence_sections() -> None:
    case = RetrievalEvaluationCase(
        case_id="case-1",
        query="SQL analyst with dashboard experience",
        relevant_candidates=(
            RelevantCandidate("candidate-a", ("WORK EXPERIENCE",)),
            RelevantCandidate("candidate-c", ("SKILLS",)),
        ),
        labeling_method="Manually verified from privacy-safe chunks.",
    )
    matches = [
        candidate_match("candidate-b", 0.91, "SKILLS"),
        candidate_match("candidate-a", 0.88, "WORK EXPERIENCE", "PROFILE"),
        candidate_match("candidate-c", 0.84, "SKILLS"),
    ]

    result = evaluate_retrieval_case(case, matches, k=2)

    assert result.recall_at_k == 0.5
    assert result.precision_at_k == 0.5
    assert result.first_relevant_rank == 2
    assert result.reciprocal_rank == 0.5
    assert result.evidence_section_recall_at_k == 0.5
    assert result.missing_candidate_ids == ("candidate-c",)


def test_evaluation_reports_zero_when_no_relevant_candidate_is_retrieved() -> None:
    case = RetrievalEvaluationCase(
        case_id="case-1",
        query="Machine learning candidate with Flask experience",
        relevant_candidates=(
            RelevantCandidate("candidate-a", ("WORK EXPERIENCE",)),
        ),
        labeling_method="Manually verified from privacy-safe chunks.",
    )

    result = evaluate_retrieval_case(
        case,
        [candidate_match("candidate-b", 0.91, "SKILLS")],
        k=1,
    )

    assert result.recall_at_k == 0.0
    assert result.precision_at_k == 0.0
    assert result.first_relevant_rank is None
    assert result.reciprocal_rank == 0.0
    assert result.evidence_section_recall_at_k == 0.0
    assert result.missing_candidate_ids == ("candidate-a",)


def test_example_evaluation_case_loads_from_the_project_dataset() -> None:
    cases_path = Path(__file__).parents[1] / "evaluation" / "retrieval_cases.jsonl"

    cases = load_retrieval_cases(cases_path)

    assert len(cases) == 8
    assert cases[0].case_id == "machine-learning-flask-001"
    assert cases[0].query == "machine learning candidate with Flask experience"
    assert cases[0].query_type == "exact_skills"
    assert cases[0].relevant_candidates == (
        RelevantCandidate(
            candidate_id="candidate-a17b9ffa1b72",
            expected_sections=("WORK EXPERIENCE",),
        ),
    )
    assert {case.query_type for case in cases} == {
        "exact_skills",
        "broad_exact_skills",
        "multi_constraint",
        "framework_skills",
        "domain_experience",
        "cross_section_experience",
        "seniority_multi_constraint",
        "leadership_methodology",
    }


@pytest.mark.parametrize(
    "candidate_id, expected_sections, expected_message",
    [
        ("", ("WORK EXPERIENCE",), "candidate"),
        ("candidate-a", (), "section"),
    ],
)
def test_relevant_candidate_rejects_incomplete_ground_truth(
    candidate_id: str,
    expected_sections: tuple[str, ...],
    expected_message: str,
) -> None:
    with pytest.raises(ValueError, match=expected_message):
        RelevantCandidate(candidate_id, expected_sections)


def test_evaluation_case_requires_at_least_one_relevant_candidate() -> None:
    with pytest.raises(ValueError, match="relevant candidate"):
        RetrievalEvaluationCase(
            case_id="case-1",
            query="SQL analyst",
            relevant_candidates=(),
            labeling_method="Manual review.",
        )


def test_evaluation_rejects_non_positive_k() -> None:
    case = RetrievalEvaluationCase(
        case_id="case-1",
        query="SQL analyst",
        relevant_candidates=(
            RelevantCandidate("candidate-a", ("WORK EXPERIENCE",)),
        ),
        labeling_method="Manual review.",
    )

    with pytest.raises(ValueError, match="positive"):
        evaluate_retrieval_case(case, [], k=0)


def test_summary_averages_results_across_cases_at_the_same_k() -> None:
    successful_case = RetrievalEvaluationCase(
        case_id="success",
        query="SQL analyst",
        relevant_candidates=(
            RelevantCandidate("candidate-a", ("WORK EXPERIENCE",)),
        ),
        labeling_method="Manual review.",
    )
    missed_case = RetrievalEvaluationCase(
        case_id="miss",
        query="Python developer",
        relevant_candidates=(
            RelevantCandidate("candidate-c", ("SKILLS",)),
        ),
        labeling_method="Manual review.",
    )
    successful_result = evaluate_retrieval_case(
        successful_case,
        [
            candidate_match("candidate-a", 0.91, "WORK EXPERIENCE"),
            candidate_match("candidate-b", 0.88, "SKILLS"),
        ],
        k=2,
    )
    missed_result = evaluate_retrieval_case(
        missed_case,
        [
            candidate_match("candidate-a", 0.91, "WORK EXPERIENCE"),
            candidate_match("candidate-b", 0.88, "SKILLS"),
        ],
        k=2,
    )

    summary = summarize_retrieval_results([successful_result, missed_result])

    assert summary.case_count == 2
    assert summary.k == 2
    assert summary.hit_rate_at_k == 0.5
    assert summary.mean_recall_at_k == 0.5
    assert summary.mean_precision_at_k == 0.25
    assert summary.mean_reciprocal_rank == 0.5
    assert summary.mean_evidence_section_recall_at_k == 0.5


def test_summary_rejects_results_with_different_k_values() -> None:
    case = RetrievalEvaluationCase(
        case_id="case-1",
        query="SQL analyst",
        relevant_candidates=(
            RelevantCandidate("candidate-a", ("WORK EXPERIENCE",)),
        ),
        labeling_method="Manual review.",
    )
    result_at_one = evaluate_retrieval_case(
        case,
        [candidate_match("candidate-a", 0.91, "WORK EXPERIENCE")],
        k=1,
    )
    result_at_two = evaluate_retrieval_case(
        case,
        [candidate_match("candidate-a", 0.91, "WORK EXPERIENCE")],
        k=2,
    )

    with pytest.raises(ValueError, match="same k"):
        summarize_retrieval_results([result_at_one, result_at_two])
