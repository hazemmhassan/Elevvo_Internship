"""Grounded LLM evaluation for retrieved, privacy-safe candidate evidence."""

from __future__ import annotations

from collections.abc import Sequence
import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from talent_search.retrieval import CandidateMatch


class EvidenceItem(BaseModel):
    """One bounded resume section supplied to the evaluator."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    section: str = Field(min_length=1)
    text: str = Field(min_length=1)


class CandidateEvidenceBundle(BaseModel):
    """Score-neutral evidence for one ranked candidate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: str = Field(min_length=1)
    rank: int = Field(ge=1)
    evidence: tuple[EvidenceItem, ...] = Field(min_length=1)


class CandidateAssessment(BaseModel):
    """A grounded explanation of one candidate's evidence and gaps."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    strengths: tuple[str, ...]
    gaps: tuple[str, ...]
    evidence_sections: tuple[str, ...]
    uncertainty: str = Field(min_length=1)


class CandidateEvaluationBatch(BaseModel):
    """Structured assessments for one recruiter query."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str = Field(min_length=1)
    assessments: tuple[CandidateAssessment, ...] = Field(min_length=1)


def assemble_candidate_evidence(
    matches: Sequence[CandidateMatch],
    *,
    max_chars_per_chunk: int = 1_500,
    max_chars_per_candidate: int = 4_000,
) -> tuple[CandidateEvidenceBundle, ...]:
    """Build bounded, anonymized evidence bundles without ranking scores."""

    if max_chars_per_chunk <= 0 or max_chars_per_candidate <= 0:
        raise ValueError("Evidence character limits must be positive.")

    bundles: list[CandidateEvidenceBundle] = []
    seen_candidate_ids: set[str] = set()
    for rank, match in enumerate(matches, start=1):
        if not match.candidate_id or match.candidate_id in seen_candidate_ids:
            raise ValueError("Candidate IDs must be non-empty and unique.")
        seen_candidate_ids.add(match.candidate_id)

        remaining = max_chars_per_candidate
        items: list[EvidenceItem] = []
        for chunk in match.evidence:
            metadata = chunk.document.metadata
            if metadata.get("privacy_masked") is not True:
                raise ValueError(
                    "Every LLM evidence document must pass privacy masking."
                )
            if metadata.get("candidate_id") != match.candidate_id:
                raise ValueError("Evidence candidate ID does not match its result.")
            section = metadata.get("section")
            if not isinstance(section, str) or not section.strip():
                raise ValueError("Every LLM evidence document needs a section.")
            if remaining <= 0:
                break
            text = chunk.document.page_content.strip()
            bounded_text = text[: min(max_chars_per_chunk, remaining)].strip()
            if not bounded_text:
                continue
            items.append(EvidenceItem(section=section.strip(), text=bounded_text))
            remaining -= len(bounded_text)

        if not items:
            raise ValueError(
                f"Candidate {match.candidate_id!r} has no usable safe evidence."
            )
        bundles.append(
            CandidateEvidenceBundle(
                candidate_id=match.candidate_id,
                rank=rank,
                evidence=tuple(items),
            )
        )
    return tuple(bundles)


def evaluate_candidates_with_llm(
    query: str,
    matches: Sequence[CandidateMatch],
    chain: Any,
) -> CandidateEvaluationBatch:
    """Evaluate candidate evidence and reject ungrounded structured output."""

    cleaned_query = query.strip()
    if not cleaned_query:
        raise ValueError("The evaluation query cannot be blank.")
    if not matches:
        raise ValueError("At least one candidate match is required.")

    bundles = assemble_candidate_evidence(matches)
    payload = {
        "query": cleaned_query,
        "candidate_evidence_json": json.dumps(
            [bundle.model_dump(mode="json") for bundle in bundles],
            ensure_ascii=False,
        ),
    }
    response = chain.invoke(payload)
    batch = CandidateEvaluationBatch.model_validate(response)

    if batch.query != cleaned_query:
        raise ValueError("The evaluator returned a different query.")

    expected_ids = [bundle.candidate_id for bundle in bundles]
    returned_ids = [item.candidate_id for item in batch.assessments]
    if len(set(returned_ids)) != len(returned_ids) or set(returned_ids) != set(
        expected_ids
    ):
        raise ValueError("The evaluator returned unexpected candidate IDs.")

    allowed_sections = {
        bundle.candidate_id: {item.section for item in bundle.evidence}
        for bundle in bundles
    }
    by_candidate = {item.candidate_id: item for item in batch.assessments}
    for assessment in batch.assessments:
        unknown_sections = set(assessment.evidence_sections).difference(
            allowed_sections[assessment.candidate_id]
        )
        if unknown_sections:
            raise ValueError(
                "The evaluator cited an unknown evidence section: "
                + ", ".join(sorted(unknown_sections))
            )

    return batch.model_copy(
        update={
            "assessments": tuple(by_candidate[candidate_id] for candidate_id in expected_ids)
        }
    )


def create_candidate_evaluation_chain(chat_model: Any) -> Any:
    """Create a LangChain prompt plus strict structured-output evaluator."""

    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You explain how anonymized resume evidence relates to a recruiter query.
Use only the supplied evidence. Never invent qualifications, identities, or facts.
For every candidate, state supported strengths, missing or unverified requirements,
and uncertainty. Cite only supplied section names. Do not use or infer demographic
traits. Do not make a hiring decision. Ranking and similarity are not probabilities
of job success. Return one assessment for every supplied candidate.""",
            ),
            (
                "human",
                "Recruiter query:\n{query}\n\nCandidate evidence JSON:\n"
                "{candidate_evidence_json}",
            ),
        ]
    )
    structured_model = chat_model.with_structured_output(
        CandidateEvaluationBatch,
        method="json_schema",
        strict=True,
    )
    return prompt | structured_model
