"""Application workflow shared by the Streamlit UI and other interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from talent_search.bias import (
    BiasAuditReport,
    QuerySafetyResult,
    audit_candidate_matches,
    sanitize_recruiter_query,
)
from talent_search.llm_evaluation import (
    CandidateEvaluationBatch,
    evaluate_candidates_with_llm,
)
from talent_search.retrieval import CandidateMatch, retrieve_candidates_hybrid
from talent_search.vector_store import FaissVectorStore


@dataclass(frozen=True, slots=True)
class TalentSearchResult:
    """One complete, UI-neutral talent-search response."""

    query_safety: QuerySafetyResult
    matches: tuple[CandidateMatch, ...]
    bias_audit: BiasAuditReport
    evaluation: CandidateEvaluationBatch | None


def search_talent(
    vector_store: FaissVectorStore | Any,
    query: str,
    *,
    candidate_k: int = 3,
    evidence_per_candidate: int = 3,
    evaluator_chain: Any | None = None,
) -> TalentSearchResult:
    """Sanitize, retrieve, audit, and optionally explain top candidates."""

    query_safety = sanitize_recruiter_query(query)
    matches = tuple(
        retrieve_candidates_hybrid(
            vector_store,
            query_safety.cleaned_query,
            candidate_k=candidate_k,
            evidence_per_candidate=evidence_per_candidate,
        )
    )
    bias_audit = audit_candidate_matches(matches)

    evaluation = None
    if evaluator_chain is not None:
        if not bias_audit.passed:
            raise ValueError(
                "LLM evaluation was blocked because the evidence failed its safety audit."
            )
        evaluation = evaluate_candidates_with_llm(
            query_safety.cleaned_query,
            matches,
            evaluator_chain,
        )

    return TalentSearchResult(
        query_safety=query_safety,
        matches=matches,
        bias_audit=bias_audit,
        evaluation=evaluation,
    )
