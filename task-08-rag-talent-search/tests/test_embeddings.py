from __future__ import annotations

from langchain_core.documents import Document
import pytest

from talent_search.embeddings import embed_documents


class MeaningfulFakeEmbeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [
            [float(len(text)), float(text.casefold().count("sql"))] for text in texts
        ]


def test_embed_documents_preserves_document_order() -> None:
    documents = [
        Document(
            page_content="SKILLS\nSQL and Tableau",
            metadata={"privacy_masked": True},
        ),
        Document(
            page_content="EDUCATION\nInformation Systems",
            metadata={"privacy_masked": True},
        ),
    ]

    vectors = embed_documents(documents, MeaningfulFakeEmbeddings())

    assert vectors == [[22.0, 1.0], [29.0, 0.0]]


def test_embed_documents_refuses_content_that_did_not_pass_privacy_masking() -> None:
    raw_document = Document(page_content="A Person", metadata={})

    with pytest.raises(ValueError, match="privacy"):
        embed_documents([raw_document], MeaningfulFakeEmbeddings())


class BrokenEmbeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 2.0], [3.0]]


def test_embed_documents_rejects_inconsistent_vector_dimensions() -> None:
    documents = [
        Document(page_content="one", metadata={"privacy_masked": True}),
        Document(page_content="two", metadata={"privacy_masked": True}),
    ]

    with pytest.raises(ValueError, match="dimension"):
        embed_documents(documents, BrokenEmbeddings())
