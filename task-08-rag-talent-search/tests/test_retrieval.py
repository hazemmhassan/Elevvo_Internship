from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
import pytest

from talent_search.retrieval import rank_candidate_matches, retrieve_candidates
from talent_search.vector_store import build_faiss_index


def scored_chunk(
    candidate_id: str,
    section: str,
    score: float,
    *,
    chunk_index: int,
) -> tuple[Document, float]:
    return (
        Document(
            page_content=f"{section}\nEvidence {chunk_index}",
            metadata={
                "candidate_id": candidate_id,
                "section": section,
                "source_line": chunk_index + 1,
                "chunk_index": chunk_index,
                "privacy_masked": True,
            },
        ),
        score,
    )


def test_candidate_ranking_uses_each_candidates_best_chunk_score() -> None:
    chunk_results = [
        scored_chunk("candidate-a", "SKILLS", 0.90, chunk_index=0),
        scored_chunk("candidate-b", "WORK EXPERIENCE", 0.88, chunk_index=0),
        scored_chunk("candidate-a", "WORK EXPERIENCE", 0.80, chunk_index=1),
        scored_chunk("candidate-a", "EDUCATION", 0.79, chunk_index=2),
        scored_chunk("candidate-c", "SKILLS", 0.70, chunk_index=0),
    ]

    matches = rank_candidate_matches(chunk_results, candidate_k=3)

    assert [match.candidate_id for match in matches] == [
        "candidate-a",
        "candidate-b",
        "candidate-c",
    ]
    assert [match.score for match in matches] == [0.90, 0.88, 0.70]


def test_candidate_evidence_keeps_only_the_strongest_chunk_per_section() -> None:
    chunk_results = [
        scored_chunk("candidate-a", "WORK EXPERIENCE", 0.91, chunk_index=0),
        scored_chunk("candidate-a", "WORK EXPERIENCE", 0.87, chunk_index=1),
        scored_chunk("candidate-a", "SKILLS", 0.82, chunk_index=2),
        scored_chunk("candidate-a", "EDUCATION", 0.75, chunk_index=3),
    ]

    matches = rank_candidate_matches(
        chunk_results,
        candidate_k=1,
        evidence_per_candidate=2,
    )

    assert [
        evidence.document.metadata["section"] for evidence in matches[0].evidence
    ] == ["WORK EXPERIENCE", "SKILLS"]
    assert [evidence.score for evidence in matches[0].evidence] == [0.91, 0.82]


@pytest.mark.parametrize(
    "candidate_k, evidence_per_candidate, expected_message",
    [
        (0, 3, "candidate"),
        (3, 0, "evidence"),
    ],
)
def test_candidate_ranking_rejects_non_positive_limits(
    candidate_k: int,
    evidence_per_candidate: int,
    expected_message: str,
) -> None:
    with pytest.raises(ValueError, match=expected_message):
        rank_candidate_matches(
            [],
            candidate_k=candidate_k,
            evidence_per_candidate=evidence_per_candidate,
        )


class CandidateKeywordEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    @staticmethod
    def _embed(text: str) -> list[float]:
        lowered = text.casefold()
        if "sql" in lowered:
            return [1.0, 0.0, 0.0]
        if "python" in lowered:
            return [0.8, 0.6, 0.0]
        return [0.0, 0.0, 1.0]


def test_retrieve_candidates_returns_distinct_candidates_from_real_faiss_search() -> None:
    documents = [
        scored_chunk("candidate-a", "SKILLS", 0.0, chunk_index=0)[0],
        scored_chunk("candidate-a", "WORK EXPERIENCE", 0.0, chunk_index=1)[0],
        scored_chunk("candidate-b", "SKILLS", 0.0, chunk_index=0)[0],
    ]
    documents[0].page_content = "SKILLS\nSQL"
    documents[1].page_content = "WORK EXPERIENCE\nSQL reporting"
    documents[2].page_content = "SKILLS\nPython"
    vector_store = build_faiss_index(documents, CandidateKeywordEmbeddings())

    matches = retrieve_candidates(
        vector_store,
        "SQL analyst",
        candidate_k=2,
        fetch_k=3,
    )

    assert [match.candidate_id for match in matches] == [
        "candidate-a",
        "candidate-b",
    ]
    assert len(matches[0].evidence) == 2


def test_retrieve_candidates_requires_fetching_at_least_as_many_chunks_as_candidates() -> None:
    document = scored_chunk("candidate-a", "SKILLS", 0.0, chunk_index=0)[0]
    vector_store = build_faiss_index([document], CandidateKeywordEmbeddings())

    with pytest.raises(ValueError, match="fetch"):
        retrieve_candidates(
            vector_store,
            "SQL analyst",
            candidate_k=3,
            fetch_k=2,
        )
