"""Small, explicit measurements for candidate retrieval quality."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from talent_search.retrieval import CandidateMatch


@dataclass(frozen=True, slots=True)
class RelevantCandidate:
    """One anonymous ground-truth candidate and the evidence expected for them."""

    candidate_id: str
    expected_sections: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.candidate_id.strip():
            raise ValueError("relevant candidate ID must not be blank.")
        if not self.expected_sections:
            raise ValueError("relevant candidate needs an expected section.")


@dataclass(frozen=True, slots=True)
class RetrievalEvaluationCase:
    """One recruiter query paired with manually verified ground truth."""

    case_id: str
    query: str
    relevant_candidates: tuple[RelevantCandidate, ...]
    labeling_method: str
    query_type: str = "unspecified"

    def __post_init__(self) -> None:
        if not self.relevant_candidates:
            raise ValueError("evaluation case needs a relevant candidate.")


@dataclass(frozen=True, slots=True)
class RetrievalEvaluationResult:
    """Candidate and evidence metrics for one retrieval case at a chosen k."""

    case_id: str
    k: int
    recall_at_k: float
    precision_at_k: float
    first_relevant_rank: int | None
    reciprocal_rank: float
    evidence_section_recall_at_k: float
    missing_candidate_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RetrievalEvaluationSummary:
    """Mean retrieval quality across cases evaluated at the same k."""

    case_count: int
    k: int
    hit_rate_at_k: float
    mean_recall_at_k: float
    mean_precision_at_k: float
    mean_reciprocal_rank: float
    mean_evidence_section_recall_at_k: float


def load_retrieval_cases(path: str | Path) -> list[RetrievalEvaluationCase]:
    """Load JSONL retrieval cases without loading or exposing resume text."""

    cases: list[RetrievalEvaluationCase] = []
    with Path(path).open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            record: dict[str, Any] = json.loads(line)
            candidates = tuple(
                RelevantCandidate(
                    candidate_id=candidate["candidate_id"],
                    expected_sections=tuple(candidate["expected_sections"]),
                )
                for candidate in record["relevant_candidates"]
            )
            cases.append(
                RetrievalEvaluationCase(
                    case_id=record["case_id"],
                    query=record["query"],
                    relevant_candidates=candidates,
                    labeling_method=record["labeling_method"],
                    query_type=record["query_type"],
                )
            )
    return cases


def evaluate_retrieval_case(
    case: RetrievalEvaluationCase,
    matches: list[CandidateMatch],
    *,
    k: int,
) -> RetrievalEvaluationResult:
    """Measure whether relevant candidates and evidence appear in the top k."""

    if k <= 0:
        raise ValueError("evaluation k must be positive.")

    top_matches = matches[:k]
    expected_ids = tuple(
        candidate.candidate_id for candidate in case.relevant_candidates
    )
    retrieved_ids = {match.candidate_id for match in top_matches}
    retrieved_relevant_ids = retrieved_ids.intersection(expected_ids)

    first_relevant_rank = next(
        (
            rank
            for rank, match in enumerate(top_matches, start=1)
            if match.candidate_id in expected_ids
        ),
        None,
    )

    expected_section_pairs = {
        (candidate.candidate_id, section)
        for candidate in case.relevant_candidates
        for section in candidate.expected_sections
    }
    retrieved_section_pairs = {
        (match.candidate_id, str(evidence.document.metadata.get("section", "")))
        for match in top_matches
        for evidence in match.evidence
    }

    return RetrievalEvaluationResult(
        case_id=case.case_id,
        k=k,
        recall_at_k=len(retrieved_relevant_ids) / len(expected_ids),
        precision_at_k=len(retrieved_relevant_ids) / k,
        first_relevant_rank=first_relevant_rank,
        reciprocal_rank=(
            1.0 / first_relevant_rank if first_relevant_rank is not None else 0.0
        ),
        evidence_section_recall_at_k=(
            len(expected_section_pairs.intersection(retrieved_section_pairs))
            / len(expected_section_pairs)
        ),
        missing_candidate_ids=tuple(
            candidate_id
            for candidate_id in expected_ids
            if candidate_id not in retrieved_ids
        ),
    )


def summarize_retrieval_results(
    results: list[RetrievalEvaluationResult],
) -> RetrievalEvaluationSummary:
    """Average retrieval metrics across cases measured at one shared k."""

    k_values = {result.k for result in results}
    if len(k_values) != 1:
        raise ValueError("retrieval results must use the same k.")

    count = len(results)
    return RetrievalEvaluationSummary(
        case_count=count,
        k=results[0].k,
        hit_rate_at_k=(
            sum(result.first_relevant_rank is not None for result in results) / count
        ),
        mean_recall_at_k=sum(result.recall_at_k for result in results) / count,
        mean_precision_at_k=sum(result.precision_at_k for result in results) / count,
        mean_reciprocal_rank=(
            sum(result.reciprocal_rank for result in results) / count
        ),
        mean_evidence_section_recall_at_k=(
            sum(result.evidence_section_recall_at_k for result in results) / count
        ),
    )
