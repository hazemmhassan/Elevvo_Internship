from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
import pytest

from talent_search.vector_store import build_faiss_index, search_faiss_index


class KeywordEmbeddings(Embeddings):
    """Small deterministic embedding model for observable search behavior."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    @staticmethod
    def _embed(text: str) -> list[float]:
        lowered = text.casefold()
        if "sql" in lowered:
            return [1.0, 0.0]
        if "retail" in lowered:
            return [0.0, 1.0]
        return [0.70710678, 0.70710678]


def privacy_safe_document(
    content: str,
    *,
    candidate_id: str,
    source_line: int,
    chunk_index: int,
) -> Document:
    return Document(
        page_content=content,
        metadata={
            "candidate_id": candidate_id,
            "source_line": source_line,
            "chunk_index": chunk_index,
            "section": "SKILLS",
            "privacy_masked": True,
        },
    )


def test_build_faiss_index_records_every_privacy_safe_chunk() -> None:
    documents = [
        privacy_safe_document(
            "SKILLS\nSQL and Tableau",
            candidate_id="candidate-1",
            source_line=1,
            chunk_index=0,
        ),
        privacy_safe_document(
            "WORK EXPERIENCE\nRetail sales",
            candidate_id="candidate-2",
            source_line=2,
            chunk_index=0,
        ),
    ]

    vector_store = build_faiss_index(documents, KeywordEmbeddings())

    assert vector_store.index.ntotal == 2
    assert vector_store.index.d == 2


def test_search_faiss_index_returns_the_closest_chunk_with_its_metadata() -> None:
    documents = [
        privacy_safe_document(
            "SKILLS\nSQL and Tableau",
            candidate_id="candidate-1",
            source_line=1,
            chunk_index=0,
        ),
        privacy_safe_document(
            "WORK EXPERIENCE\nRetail sales",
            candidate_id="candidate-2",
            source_line=2,
            chunk_index=0,
        ),
    ]
    vector_store = build_faiss_index(documents, KeywordEmbeddings())

    results = search_faiss_index(vector_store, "SQL analyst", k=1)

    document, score = results[0]
    assert document.page_content == "SKILLS\nSQL and Tableau"
    assert document.metadata["candidate_id"] == "candidate-1"
    assert score == pytest.approx(1.0)


def test_build_faiss_index_rejects_raw_resume_content() -> None:
    raw_document = Document(page_content="A Person\nSKILLS\nSQL", metadata={})

    with pytest.raises(ValueError, match="privacy"):
        build_faiss_index([raw_document], KeywordEmbeddings())


def test_build_faiss_index_rejects_an_empty_document_collection() -> None:
    with pytest.raises(ValueError, match="at least one"):
        build_faiss_index([], KeywordEmbeddings())


def test_build_faiss_index_rejects_duplicate_chunk_identities() -> None:
    first = privacy_safe_document(
        "SKILLS\nSQL",
        candidate_id="candidate-1",
        source_line=1,
        chunk_index=0,
    )
    duplicate_identity = privacy_safe_document(
        "WORK EXPERIENCE\nSQL reporting",
        candidate_id="candidate-1",
        source_line=1,
        chunk_index=0,
    )

    with pytest.raises(ValueError, match="unique"):
        build_faiss_index([first, duplicate_identity], KeywordEmbeddings())


@pytest.mark.parametrize(
    "query, k, expected_message",
    [
        ("   ", 1, "query"),
        ("SQL analyst", 0, "positive"),
    ],
)
def test_search_faiss_index_rejects_invalid_search_requests(
    query: str, k: int, expected_message: str
) -> None:
    document = privacy_safe_document(
        "SKILLS\nSQL",
        candidate_id="candidate-1",
        source_line=1,
        chunk_index=0,
    )
    vector_store = build_faiss_index([document], KeywordEmbeddings())

    with pytest.raises(ValueError, match=expected_message):
        search_faiss_index(vector_store, query, k=k)
