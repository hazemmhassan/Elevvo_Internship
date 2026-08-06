from __future__ import annotations

import json

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
import pytest

from talent_search.llm_evaluation import (
    CandidateAssessment,
    CandidateEvaluationBatch,
    assemble_candidate_evidence,
    evaluate_candidates_with_llm,
)
from talent_search.retrieval import CandidateMatch, ChunkEvidence


def candidate_match(
    candidate_id: str,
    *section_and_text: tuple[str, str],
) -> CandidateMatch:
    evidence = tuple(
        ChunkEvidence(
            document=Document(
                page_content=f"{section}\n{text}",
                metadata={
                    "candidate_id": candidate_id,
                    "section": section,
                    "privacy_masked": True,
                },
            ),
            score=0.8 - index * 0.1,
        )
        for index, (section, text) in enumerate(section_and_text)
    )
    return CandidateMatch(candidate_id=candidate_id, score=0.9, evidence=evidence)


def valid_batch(query: str) -> CandidateEvaluationBatch:
    return CandidateEvaluationBatch(
        query=query,
        assessments=(
            CandidateAssessment(
                candidate_id="candidate-a",
                summary="Strong SQL reporting evidence with an unverified Tableau gap.",
                strengths=("SQL reporting experience",),
                gaps=("Tableau is not shown in the supplied evidence",),
                evidence_sections=("WORK EXPERIENCE",),
                uncertainty="Only the supplied sections were evaluated.",
            ),
            CandidateAssessment(
                candidate_id="candidate-b",
                summary="Python is supported, while dashboard experience is unknown.",
                strengths=("Python skill evidence",),
                gaps=("Dashboard experience is not shown",),
                evidence_sections=("SKILLS",),
                uncertainty="The evidence is limited to one section.",
            ),
        ),
    )


def test_evidence_assembly_is_bounded_and_contains_only_safe_sections() -> None:
    matches = [
        candidate_match(
            "candidate-a",
            ("WORK EXPERIENCE", "SQL reporting " * 50),
            ("SKILLS", "Tableau"),
        )
    ]

    bundles = assemble_candidate_evidence(
        matches,
        max_chars_per_chunk=80,
        max_chars_per_candidate=120,
    )

    assert len(bundles) == 1
    assert bundles[0].candidate_id == "candidate-a"
    assert len(bundles[0].evidence[0].text) <= 80
    assert sum(len(item.text) for item in bundles[0].evidence) <= 120
    assert {item.section for item in bundles[0].evidence} == {
        "WORK EXPERIENCE",
        "SKILLS",
    }


def test_evidence_assembly_rejects_documents_that_skipped_privacy_masking() -> None:
    unsafe = Document(
        page_content="A Person person@example.com",
        metadata={"candidate_id": "candidate-a", "section": "PROFILE"},
    )
    match = CandidateMatch(
        candidate_id="candidate-a",
        score=0.9,
        evidence=(ChunkEvidence(document=unsafe, score=0.8),),
    )

    with pytest.raises(ValueError, match="privacy"):
        assemble_candidate_evidence([match])


def test_llm_evaluation_receives_json_evidence_and_returns_rank_order() -> None:
    query = "data analyst with SQL and Tableau"
    matches = [
        candidate_match("candidate-a", ("WORK EXPERIENCE", "SQL reporting")),
        candidate_match("candidate-b", ("SKILLS", "Python")),
    ]

    def evaluate_payload(payload: dict[str, str]) -> CandidateEvaluationBatch:
        evidence = json.loads(payload["candidate_evidence_json"])
        assert payload["query"] == query
        assert [item["candidate_id"] for item in evidence] == [
            "candidate-a",
            "candidate-b",
        ]
        assert "score" not in evidence[0]
        return valid_batch(query)

    batch = evaluate_candidates_with_llm(
        query,
        matches,
        RunnableLambda(evaluate_payload),
    )

    assert [assessment.candidate_id for assessment in batch.assessments] == [
        "candidate-a",
        "candidate-b",
    ]


@pytest.mark.parametrize(
    "assessment, expected_message",
    [
        (
            CandidateAssessment(
                candidate_id="candidate-unknown",
                summary="Unsupported candidate.",
                strengths=("Unknown",),
                gaps=("Unknown",),
                evidence_sections=("WORK EXPERIENCE",),
                uncertainty="Unknown",
            ),
            "candidate IDs",
        ),
        (
            CandidateAssessment(
                candidate_id="candidate-a",
                summary="Unsupported citation.",
                strengths=("Unknown",),
                gaps=("Unknown",),
                evidence_sections=("EDUCATION",),
                uncertainty="Unknown",
            ),
            "evidence section",
        ),
    ],
)
def test_llm_evaluation_rejects_unknown_candidates_or_sections(
    assessment: CandidateAssessment,
    expected_message: str,
) -> None:
    query = "SQL analyst"
    matches = [
        candidate_match("candidate-a", ("WORK EXPERIENCE", "SQL reporting"))
    ]
    invalid_batch = CandidateEvaluationBatch(
        query=query,
        assessments=(assessment,),
    )

    with pytest.raises(ValueError, match=expected_message):
        evaluate_candidates_with_llm(
            query,
            matches,
            RunnableLambda(lambda _payload: invalid_batch),
        )
