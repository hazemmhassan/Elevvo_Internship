"""Candidate-level aggregation for FAISS chunk search results."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
import math

from langchain_core.documents import Document

from talent_search.vector_store import FaissVectorStore, search_faiss_index


@dataclass(frozen=True, slots=True)
class ChunkEvidence:
    """One retrieved resume chunk supporting a candidate match."""

    document: Document
    score: float


@dataclass(frozen=True, slots=True)
class CandidateMatch:
    """A distinct candidate with score-neutral supporting evidence."""

    candidate_id: str
    score: float
    evidence: tuple[ChunkEvidence, ...]


def rank_candidate_matches(
    chunk_results: Sequence[tuple[Document, float]],
    *,
    candidate_k: int = 3,
    evidence_per_candidate: int = 3,
) -> list[CandidateMatch]:
    """Group chunks by candidate and rank candidates by their best chunk."""

    if candidate_k <= 0:
        raise ValueError("candidate result count must be positive.")
    if evidence_per_candidate <= 0:
        raise ValueError("evidence count must be positive.")

    grouped: dict[str, list[ChunkEvidence]] = defaultdict(list)
    for document, score in chunk_results:
        candidate_id = document.metadata.get("candidate_id")
        section = document.metadata.get("section")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError("Every retrieved chunk needs a candidate_id.")
        if not isinstance(section, str) or not section:
            raise ValueError("Every retrieved chunk needs a section.")
        if not math.isfinite(score):
            raise ValueError("Every retrieved chunk needs a finite score.")
        grouped[candidate_id].append(ChunkEvidence(document=document, score=score))

    matches: list[CandidateMatch] = []
    for candidate_id, chunks in grouped.items():
        ordered_chunks = sorted(chunks, key=lambda evidence: -evidence.score)
        evidence = _strongest_distinct_sections(
            ordered_chunks,
            limit=evidence_per_candidate,
        )
        matches.append(
            CandidateMatch(
                candidate_id=candidate_id,
                score=ordered_chunks[0].score,
                evidence=tuple(evidence),
            )
        )

    matches.sort(key=lambda match: (-match.score, match.candidate_id))
    return matches[:candidate_k]


def retrieve_candidates(
    vector_store: FaissVectorStore,
    query: str,
    *,
    candidate_k: int = 3,
    fetch_k: int = 30,
    evidence_per_candidate: int = 3,
) -> list[CandidateMatch]:
    """Search more chunks than candidates, then return distinct candidates."""

    if fetch_k < candidate_k:
        raise ValueError("FAISS fetch count must be at least the candidate count.")
    chunk_results = search_faiss_index(vector_store, query, k=fetch_k)
    return rank_candidate_matches(
        chunk_results,
        candidate_k=candidate_k,
        evidence_per_candidate=evidence_per_candidate,
    )


def _strongest_distinct_sections(
    chunks: Sequence[ChunkEvidence], *, limit: int
) -> list[ChunkEvidence]:
    selected: list[ChunkEvidence] = []
    seen_sections: set[str] = set()
    for evidence in chunks:
        section = str(evidence.document.metadata["section"])
        if section in seen_sections:
            continue
        seen_sections.add(section)
        selected.append(evidence)
        if len(selected) == limit:
            break
    return selected
