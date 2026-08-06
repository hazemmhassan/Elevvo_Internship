from __future__ import annotations

import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
import pytest

from talent_search.vector_store import (
    build_faiss_index,
    load_faiss_index,
    save_faiss_index,
    search_faiss_index,
)


class PersistenceKeywordEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    @staticmethod
    def _embed(text: str) -> list[float]:
        lowered = text.casefold()
        if "sql" in lowered:
            return [1.0, 0.0]
        return [0.0, 1.0]


def safe_document(content: str, candidate_id: str, source_line: int) -> Document:
    return Document(
        page_content=f"SKILLS\n{content}",
        metadata={
            "candidate_id": candidate_id,
            "source_line": source_line,
            "chunk_index": 0,
            "section_chunk_index": 0,
            "section": "SKILLS",
            "privacy_masked": True,
        },
    )


def build_test_store():
    documents = [
        safe_document("SQL and Tableau", "candidate-a", 1),
        safe_document("Retail operations", "candidate-b", 2),
    ]
    return build_faiss_index(documents, PersistenceKeywordEmbeddings())


def test_faiss_index_round_trip_preserves_search_and_safe_metadata(
    tmp_path: Path,
) -> None:
    store = build_test_store()
    index_directory = tmp_path / "resume-index"

    manifest = save_faiss_index(
        store,
        index_directory,
        embedding_model_name="test/keyword-embeddings",
    )
    loaded = load_faiss_index(
        index_directory,
        PersistenceKeywordEmbeddings(),
        expected_embedding_model_name="test/keyword-embeddings",
    )
    results = search_faiss_index(loaded, "SQL analyst", k=1)

    assert manifest.vector_count == 2
    assert manifest.dimensions == 2
    assert (index_directory / "index.faiss").is_file()
    assert (index_directory / "documents.jsonl").is_file()
    assert (index_directory / "manifest.json").is_file()
    assert results[0][0].metadata["candidate_id"] == "candidate-a"
    assert results[0][0].metadata["privacy_masked"] is True
    assert results[0][1] == pytest.approx(1.0)


def test_load_rejects_tampered_persisted_documents(tmp_path: Path) -> None:
    index_directory = tmp_path / "resume-index"
    save_faiss_index(
        build_test_store(),
        index_directory,
        embedding_model_name="test/keyword-embeddings",
    )
    documents_path = index_directory / "documents.jsonl"
    documents_path.write_text(
        documents_path.read_text(encoding="utf-8").replace("SQL", "Python"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="fingerprint"):
        load_faiss_index(
            index_directory,
            PersistenceKeywordEmbeddings(),
            expected_embedding_model_name="test/keyword-embeddings",
        )


def test_load_rejects_a_different_embedding_model(tmp_path: Path) -> None:
    index_directory = tmp_path / "resume-index"
    save_faiss_index(
        build_test_store(),
        index_directory,
        embedding_model_name="test/keyword-embeddings",
    )

    with pytest.raises(ValueError, match="embedding model"):
        load_faiss_index(
            index_directory,
            PersistenceKeywordEmbeddings(),
            expected_embedding_model_name="test/different-model",
        )


@pytest.mark.parametrize(
    "field, value, expected_message",
    [
        ("vector_count", 99, "count"),
        ("dimensions", 99, "dimension"),
    ],
)
def test_load_rejects_an_incompatible_manifest(
    tmp_path: Path,
    field: str,
    value: int,
    expected_message: str,
) -> None:
    index_directory = tmp_path / "resume-index"
    save_faiss_index(
        build_test_store(),
        index_directory,
        embedding_model_name="test/keyword-embeddings",
    )
    manifest_path = index_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=expected_message):
        load_faiss_index(
            index_directory,
            PersistenceKeywordEmbeddings(),
            expected_embedding_model_name="test/keyword-embeddings",
        )


def test_save_refuses_to_overwrite_an_existing_directory(tmp_path: Path) -> None:
    index_directory = tmp_path / "resume-index"
    index_directory.mkdir()
    (index_directory / "keep.txt").write_text("user data", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        save_faiss_index(
            build_test_store(),
            index_directory,
            embedding_model_name="test/keyword-embeddings",
        )

    assert (index_directory / "keep.txt").read_text(encoding="utf-8") == "user data"
