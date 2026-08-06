"""Candidate-level aggregation for FAISS chunk search results."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
import math
import re

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
    dense_score: float | None = None
    lexical_score: float | None = None
    term_coverage: float | None = None
    phrase_coverage: float | None = None
    ranking_method: str = "dense_best_chunk"


_TOKEN_PATTERN = re.compile(
    r"(?:\.[a-z0-9]+|[a-z0-9]+(?:[.+-][a-z0-9]+)*#?)",
    re.IGNORECASE,
)
_QUERY_STOPWORDS = {
    "a",
    "an",
    "and",
    "candidate",
    "candidates",
    "experience",
    "experienced",
    "find",
    "for",
    "has",
    "have",
    "in",
    "knows",
    "looking",
    "of",
    "or",
    "required",
    "skill",
    "skills",
    "the",
    "to",
    "who",
    "with",
}
_PHRASE_BOUNDARY_PATTERN = re.compile(
    r"[,;:/]|\b(?:and|candidate|candidates|experience|experienced|find|for|has|"
    r"have|looking|required|skills?|who|with)\b",
    re.IGNORECASE,
)


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
                dense_score=ordered_chunks[0].score,
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


def retrieve_candidates_hybrid(
    vector_store: FaissVectorStore,
    query: str,
    *,
    candidate_k: int = 3,
    evidence_per_candidate: int = 3,
    rrf_constant: int = 60,
) -> list[CandidateMatch]:
    """Combine dense similarity, BM25, and explicit query-term coverage."""

    if candidate_k <= 0:
        raise ValueError("candidate result count must be positive.")
    if evidence_per_candidate <= 0:
        raise ValueError("evidence count must be positive.")
    if rrf_constant <= 0:
        raise ValueError("RRF constant must be positive.")

    documents = list(vector_store.documents_by_id.values())
    dense_chunk_results = search_faiss_index(
        vector_store,
        query,
        k=len(documents),
    )
    chunks_by_candidate: dict[str, list[Document]] = defaultdict(list)
    dense_score_by_chunk: dict[tuple[str, int, int], float] = {}
    for document, score in dense_chunk_results:
        candidate_id = str(document.metadata["candidate_id"])
        chunks_by_candidate[candidate_id].append(document)
        dense_score_by_chunk[_chunk_key(document)] = score

    dense_matches = rank_candidate_matches(
        dense_chunk_results,
        candidate_k=len(chunks_by_candidate),
        evidence_per_candidate=evidence_per_candidate,
    )
    dense_rank = {
        match.candidate_id: rank
        for rank, match in enumerate(dense_matches, start=1)
    }
    dense_score = {match.candidate_id: match.score for match in dense_matches}

    query_terms = _meaningful_query_terms(query)
    query_phrases = _meaningful_query_phrases(query)
    candidate_tokens = {
        candidate_id: _tokenize(
            "\n".join(document.page_content for document in candidate_documents)
        )
        for candidate_id, candidate_documents in chunks_by_candidate.items()
    }
    lexical_scores = _bm25_scores(candidate_tokens, query_terms)
    lexical_order = sorted(
        lexical_scores,
        key=lambda candidate_id: (-lexical_scores[candidate_id], candidate_id),
    )
    lexical_rank = {
        candidate_id: rank
        for rank, candidate_id in enumerate(lexical_order, start=1)
    }

    query_term_set = set(query_terms)
    matches: list[CandidateMatch] = []
    for candidate_id, candidate_documents in chunks_by_candidate.items():
        candidate_term_set = set(candidate_tokens[candidate_id])
        coverage = len(query_term_set.intersection(candidate_term_set)) / len(
            query_term_set
        )
        candidate_normalized_text = " ".join(candidate_tokens[candidate_id])
        matched_phrases = sum(
            phrase in candidate_normalized_text for phrase in query_phrases
        )
        phrase_coverage = (
            matched_phrases / len(query_phrases) if query_phrases else 0.0
        )
        reciprocal_rank_fusion = (
            1.0 / (rrf_constant + dense_rank[candidate_id])
            + 1.0 / (rrf_constant + lexical_rank[candidate_id])
        )
        hybrid_score = coverage + phrase_coverage + reciprocal_rank_fusion

        evidence = [
            ChunkEvidence(
                document=document,
                score=dense_score_by_chunk[_chunk_key(document)],
            )
            for document in candidate_documents
        ]
        evidence.sort(
            key=lambda item: (
                -_phrase_coverage_count(item.document.page_content, query_phrases),
                -_term_coverage_count(item.document.page_content, query_term_set),
                -item.score,
            )
        )
        selected_evidence = _strongest_distinct_sections(
            evidence,
            limit=evidence_per_candidate,
        )
        matches.append(
            CandidateMatch(
                candidate_id=candidate_id,
                score=hybrid_score,
                evidence=tuple(selected_evidence),
                dense_score=dense_score[candidate_id],
                lexical_score=lexical_scores[candidate_id],
                term_coverage=coverage,
                phrase_coverage=phrase_coverage,
                ranking_method="hybrid_constraint_aware",
            )
        )

    matches.sort(key=lambda match: (-match.score, match.candidate_id))
    return matches[:candidate_k]


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


def _tokenize(text: str) -> list[str]:
    return [match.group(0).casefold() for match in _TOKEN_PATTERN.finditer(text)]


def _meaningful_query_terms(query: str) -> list[str]:
    tokens = _tokenize(query)
    meaningful = [token for token in tokens if token not in _QUERY_STOPWORDS]
    return meaningful or tokens


def _meaningful_query_phrases(query: str) -> list[str]:
    phrases: list[str] = []
    for segment in _PHRASE_BOUNDARY_PATTERN.split(query):
        tokens = [
            token
            for token in _tokenize(segment)
            if token not in _QUERY_STOPWORDS
        ]
        if len(tokens) >= 2:
            phrases.append(" ".join(tokens))
    return phrases


def _bm25_scores(
    candidate_tokens: dict[str, list[str]],
    query_terms: Sequence[str],
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> dict[str, float]:
    candidate_count = len(candidate_tokens)
    average_length = sum(map(len, candidate_tokens.values())) / candidate_count
    unique_query_terms = set(query_terms)
    document_frequency = {
        term: sum(term in tokens for tokens in candidate_tokens.values())
        for term in unique_query_terms
    }

    scores: dict[str, float] = {}
    for candidate_id, tokens in candidate_tokens.items():
        frequencies = Counter(tokens)
        length_ratio = len(tokens) / average_length
        score = 0.0
        for term in unique_query_terms:
            frequency = frequencies[term]
            if frequency == 0:
                continue
            inverse_document_frequency = math.log(
                1.0
                + (candidate_count - document_frequency[term] + 0.5)
                / (document_frequency[term] + 0.5)
            )
            score += inverse_document_frequency * (
                frequency * (k1 + 1.0)
                / (frequency + k1 * (1.0 - b + b * length_ratio))
            )
        scores[candidate_id] = score
    return scores


def _term_coverage_count(text: str, query_terms: set[str]) -> int:
    return len(query_terms.intersection(_tokenize(text)))


def _phrase_coverage_count(text: str, query_phrases: Sequence[str]) -> int:
    normalized_text = " ".join(_tokenize(text))
    return sum(phrase in normalized_text for phrase in query_phrases)


def _chunk_key(document: Document) -> tuple[str, int, int]:
    return (
        str(document.metadata["candidate_id"]),
        int(document.metadata["source_line"]),
        int(document.metadata["chunk_index"]),
    )
