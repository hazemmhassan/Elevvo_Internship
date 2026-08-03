from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
import pytest

from talent_search.retrieval import retrieve_candidates_hybrid
from talent_search.vector_store import build_faiss_index


class MisleadingSemanticEmbeddings(Embeddings):
    """Ranks a generic AI profile above the exact Flask evidence."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_document(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        del text
        return [1.0, 0.0]

    @staticmethod
    def _embed_document(text: str) -> list[float]:
        if "artificial intelligence" in text.casefold():
            return [1.0, 0.0]
        return [0.8, 0.6]


def safe_chunk(
    candidate_id: str,
    section: str,
    content: str,
    *,
    source_line: int,
    chunk_index: int,
) -> Document:
    return Document(
        page_content=f"{section}\n{content}",
        metadata={
            "candidate_id": candidate_id,
            "section": section,
            "source_line": source_line,
            "chunk_index": chunk_index,
            "section_chunk_index": 0,
            "privacy_masked": True,
        },
    )


def test_hybrid_retrieval_prioritizes_complete_term_coverage_over_dense_rank() -> None:
    documents = [
        safe_chunk(
            "candidate-generic",
            "PROFILE",
            "Artificial intelligence engineer",
            source_line=1,
            chunk_index=0,
        ),
        safe_chunk(
            "candidate-exact",
            "WORK EXPERIENCE",
            "Built machine learning services with Flask",
            source_line=2,
            chunk_index=0,
        ),
    ]
    store = build_faiss_index(documents, MisleadingSemanticEmbeddings())

    matches = retrieve_candidates_hybrid(
        store,
        "machine learning candidate with Flask experience",
        candidate_k=2,
    )

    assert [match.candidate_id for match in matches] == [
        "candidate-exact",
        "candidate-generic",
    ]
    assert matches[0].ranking_method == "hybrid_constraint_aware"
    assert matches[0].term_coverage == 1.0
    assert matches[0].dense_score == pytest.approx(0.8)
    assert matches[0].lexical_score > matches[1].lexical_score


def test_hybrid_evidence_collects_query_terms_from_distinct_sections() -> None:
    documents = [
        safe_chunk(
            "candidate-a",
            "WORK EXPERIENCE",
            "Created forecasting models",
            source_line=1,
            chunk_index=0,
        ),
        safe_chunk(
            "candidate-a",
            "SKILLS",
            "Python",
            source_line=1,
            chunk_index=1,
        ),
        safe_chunk(
            "candidate-a",
            "PROFILE",
            "Analytics professional",
            source_line=1,
            chunk_index=2,
        ),
        safe_chunk(
            "candidate-b",
            "PROFILE",
            "Artificial intelligence engineer",
            source_line=2,
            chunk_index=0,
        ),
    ]
    store = build_faiss_index(documents, MisleadingSemanticEmbeddings())

    matches = retrieve_candidates_hybrid(
        store,
        "forecasting and Python experience",
        candidate_k=1,
        evidence_per_candidate=2,
    )

    assert matches[0].candidate_id == "candidate-a"
    assert {
        evidence.document.metadata["section"] for evidence in matches[0].evidence
    } == {"WORK EXPERIENCE", "SKILLS"}


def test_hybrid_retrieval_preserves_multiword_role_phrases() -> None:
    documents = [
        safe_chunk(
            "candidate-a-scattered",
            "WORK EXPERIENCE",
            "Artificial intelligence team Agile lead",
            source_line=1,
            chunk_index=0,
        ),
        safe_chunk(
            "candidate-z-exact",
            "WORK EXPERIENCE",
            "Worked as a team lead using Agile",
            source_line=2,
            chunk_index=0,
        ),
    ]
    store = build_faiss_index(documents, MisleadingSemanticEmbeddings())

    matches = retrieve_candidates_hybrid(
        store,
        "team lead with Agile experience",
        candidate_k=2,
    )

    assert matches[0].candidate_id == "candidate-z-exact"
    assert matches[0].phrase_coverage == 1.0
    assert matches[1].phrase_coverage == 0.0


@pytest.mark.parametrize(
    "candidate_k, evidence_per_candidate, expected_message",
    [
        (0, 3, "candidate"),
        (3, 0, "evidence"),
    ],
)
def test_hybrid_retrieval_rejects_non_positive_limits(
    candidate_k: int,
    evidence_per_candidate: int,
    expected_message: str,
) -> None:
    document = safe_chunk(
        "candidate-a",
        "SKILLS",
        "Python",
        source_line=1,
        chunk_index=0,
    )
    store = build_faiss_index([document], MisleadingSemanticEmbeddings())

    with pytest.raises(ValueError, match=expected_message):
        retrieve_candidates_hybrid(
            store,
            "Python developer",
            candidate_k=candidate_k,
            evidence_per_candidate=evidence_per_candidate,
        )
