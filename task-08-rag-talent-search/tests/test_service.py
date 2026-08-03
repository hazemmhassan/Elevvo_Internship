from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from talent_search.llm_evaluation import (
    CandidateAssessment,
    CandidateEvaluationBatch,
)
from talent_search.retrieval import CandidateMatch, ChunkEvidence
from talent_search.service import search_talent


def safe_match() -> CandidateMatch:
    document = Document(
        page_content="SKILLS\nSQL Tableau",
        metadata={
            "candidate_id": "candidate-a",
            "section": "SKILLS",
            "privacy_masked": True,
        },
    )
    return CandidateMatch(
        candidate_id="candidate-a",
        score=1.1,
        evidence=(ChunkEvidence(document=document, score=0.8),),
    )


def test_search_workflow_sanitizes_retrieval_and_optionally_evaluates(
    monkeypatch,
) -> None:
    captured: dict[str, str] = {}

    def retrieve(_store, query: str, **_kwargs):
        captured["query"] = query
        return [safe_match()]

    monkeypatch.setattr("talent_search.service.retrieve_candidates_hybrid", retrieve)
    chain = RunnableLambda(
        lambda payload: CandidateEvaluationBatch(
            query=payload["query"],
            assessments=(
                CandidateAssessment(
                    candidate_id="candidate-a",
                    summary="SQL and Tableau are supported by the evidence.",
                    strengths=("SQL and Tableau",),
                    gaps=("Seniority is not shown",),
                    evidence_sections=("SKILLS",),
                    uncertainty="Only a skills section was supplied.",
                ),
            ),
        )
    )

    result = search_talent(
        object(),
        "Find a young female analyst with SQL and Tableau",
        evaluator_chain=chain,
    )

    assert captured["query"] == "Find an analyst with SQL and Tableau"
    assert result.query_safety.protected_terms == ("young", "female")
    assert result.bias_audit.passed is True
    assert result.evaluation is not None
    assert result.evaluation.assessments[0].candidate_id == "candidate-a"


def test_search_workflow_still_returns_retrieval_without_an_llm(monkeypatch) -> None:
    monkeypatch.setattr(
        "talent_search.service.retrieve_candidates_hybrid",
        lambda _store, _query, **_kwargs: [safe_match()],
    )

    result = search_talent(object(), "SQL analyst")

    assert [match.candidate_id for match in result.matches] == ["candidate-a"]
    assert result.evaluation is None
