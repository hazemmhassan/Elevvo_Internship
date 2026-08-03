"""A transparent LangChain-compatible vector store backed by FAISS."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any
from uuid import uuid4

import faiss
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
import numpy as np


INDEX_FORMAT_VERSION = 1


@dataclass(frozen=True, slots=True)
class IndexManifest:
    """Compatibility and integrity metadata for one persisted FAISS index."""

    format_version: int
    embedding_model_name: str
    vector_count: int
    dimensions: int
    documents_sha256: str


class FaissVectorStore(VectorStore):
    """Exact cosine-similarity search over LangChain Documents."""

    def __init__(
        self,
        embedding: Embeddings,
        index: faiss.IndexFlatIP,
        documents_by_id: dict[str, Document],
        index_to_document_id: dict[int, str],
    ) -> None:
        self._embedding = embedding
        self.index = index
        self.documents_by_id = documents_by_id
        self.index_to_document_id = index_to_document_id

    @property
    def embeddings(self) -> Embeddings:
        return self._embedding

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict[str, Any]] | None = None,
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> FaissVectorStore:
        del kwargs
        if not texts:
            raise ValueError("FAISS needs at least one document.")
        if metadatas is None:
            metadatas = [{} for _ in texts]
        if len(metadatas) != len(texts):
            raise ValueError("Every text needs matching metadata.")
        if ids is None:
            ids = [str(uuid4()) for _ in texts]
        if len(ids) != len(texts) or len(set(ids)) != len(ids):
            raise ValueError("FAISS document IDs must be unique and match the texts.")

        matrix = _as_normalized_matrix(embedding.embed_documents(texts))
        if matrix.shape[0] != len(texts):
            raise ValueError("Embedding model returned a different number of vectors.")

        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        documents_by_id = {
            document_id: Document(page_content=text, metadata=metadata)
            for document_id, text, metadata in zip(ids, texts, metadatas, strict=True)
        }
        index_to_document_id = dict(enumerate(ids))
        return cls(embedding, index, documents_by_id, index_to_document_id)

    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> list[Document]:
        return [
            document
            for document, _score in self.similarity_search_with_score(
                query, k=k, **kwargs
            )
        ]

    def similarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[tuple[Document, float]]:
        del kwargs
        if not query.strip():
            raise ValueError("Search query must not be blank.")
        if k <= 0:
            raise ValueError("Search result count must be positive.")

        query_matrix = _as_normalized_matrix([self._embedding.embed_query(query)])
        if query_matrix.shape[1] != self.index.d:
            raise ValueError("Query and index embedding dimensions must match.")

        scores, positions = self.index.search(
            query_matrix, min(k, self.index.ntotal)
        )
        results: list[tuple[Document, float]] = []
        for position, score in zip(positions[0], scores[0], strict=True):
            document_id = self.index_to_document_id[int(position)]
            results.append((self.documents_by_id[document_id], float(score)))
        return results


def build_faiss_index(
    documents: Sequence[Document], embedding: Embeddings
) -> FaissVectorStore:
    """Build an exact in-memory index from privacy-safe resume chunks."""

    documents = list(documents)
    if not documents:
        raise ValueError("FAISS needs at least one document.")
    if any(document.metadata.get("privacy_masked") is not True for document in documents):
        raise ValueError("Every document must pass privacy masking before FAISS indexing.")

    ids = [_document_id(document) for document in documents]
    if len(set(ids)) != len(ids):
        raise ValueError("Every FAISS chunk identity must be unique.")
    return FaissVectorStore.from_documents(documents, embedding, ids=ids)


def search_faiss_index(
    vector_store: FaissVectorStore, query: str, k: int = 5
) -> list[tuple[Document, float]]:
    """Return the closest chunks and their cosine-similarity scores."""

    return vector_store.similarity_search_with_score(query, k=k)


def save_faiss_index(
    vector_store: FaissVectorStore,
    directory: str | Path,
    *,
    embedding_model_name: str,
) -> IndexManifest:
    """Atomically persist an index and its privacy-safe LangChain documents."""

    if not embedding_model_name.strip():
        raise ValueError("embedding model name must not be blank.")
    target = Path(directory)
    if target.exists():
        raise FileExistsError(f"Index directory already exists: {target}")

    records: list[dict[str, Any]] = []
    for position in range(vector_store.index.ntotal):
        document_id = vector_store.index_to_document_id[position]
        document = vector_store.documents_by_id[document_id]
        if document.metadata.get("privacy_masked") is not True:
            raise ValueError("Persisted documents must pass privacy masking.")
        records.append(
            {
                "id": document_id,
                "page_content": document.page_content,
                "metadata": document.metadata,
            }
        )

    document_lines = [
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for record in records
    ]
    documents_text = "\n".join(document_lines) + "\n"
    manifest = IndexManifest(
        format_version=INDEX_FORMAT_VERSION,
        embedding_model_name=embedding_model_name,
        vector_count=vector_store.index.ntotal,
        dimensions=vector_store.index.d,
        documents_sha256=sha256(documents_text.encode("utf-8")).hexdigest(),
    )

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_directory = Path(
        tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent)
    )
    try:
        faiss.write_index(vector_store.index, str(temporary_directory / "index.faiss"))
        (temporary_directory / "documents.jsonl").write_text(
            documents_text,
            encoding="utf-8",
        )
        (temporary_directory / "manifest.json").write_text(
            json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_directory.replace(target)
    except Exception:
        shutil.rmtree(temporary_directory, ignore_errors=True)
        raise
    return manifest


def load_faiss_index(
    directory: str | Path,
    embedding: Embeddings,
    *,
    expected_embedding_model_name: str,
) -> FaissVectorStore:
    """Load a persisted index only after compatibility and integrity checks."""

    source = Path(directory)
    manifest_path = source / "manifest.json"
    documents_path = source / "documents.jsonl"
    index_path = source / "index.faiss"
    missing = [
        path.name
        for path in (manifest_path, documents_path, index_path)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Persisted index is missing required files: {', '.join(missing)}"
        )

    try:
        manifest = IndexManifest(**json.loads(manifest_path.read_text(encoding="utf-8")))
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Persisted index manifest is invalid.") from exc
    if manifest.format_version != INDEX_FORMAT_VERSION:
        raise ValueError("Persisted index format version is incompatible.")
    if manifest.embedding_model_name != expected_embedding_model_name:
        raise ValueError("Persisted index uses a different embedding model.")

    documents_text = documents_path.read_text(encoding="utf-8")
    actual_fingerprint = sha256(documents_text.encode("utf-8")).hexdigest()
    if actual_fingerprint != manifest.documents_sha256:
        raise ValueError("Persisted document fingerprint does not match the manifest.")

    records = [
        json.loads(line)
        for line in documents_text.splitlines()
        if line.strip()
    ]
    if len(records) != manifest.vector_count:
        raise ValueError("Persisted document count does not match the manifest.")

    documents_by_id: dict[str, Document] = {}
    index_to_document_id: dict[int, str] = {}
    for position, record in enumerate(records):
        document_id = str(record["id"])
        if document_id in documents_by_id:
            raise ValueError("Persisted document IDs must be unique.")
        document = Document(
            page_content=str(record["page_content"]),
            metadata=dict(record["metadata"]),
        )
        if document.metadata.get("privacy_masked") is not True:
            raise ValueError("Persisted documents must pass privacy masking.")
        documents_by_id[document_id] = document
        index_to_document_id[position] = document_id

    index = faiss.read_index(str(index_path))
    if index.ntotal != manifest.vector_count:
        raise ValueError("FAISS vector count does not match the manifest.")
    if index.d != manifest.dimensions:
        raise ValueError("FAISS vector dimensions do not match the manifest.")
    return FaissVectorStore(
        embedding,
        index,
        documents_by_id,
        index_to_document_id,
    )


def _document_id(document: Document) -> str:
    required_metadata = ("candidate_id", "source_line", "chunk_index")
    missing = [key for key in required_metadata if key not in document.metadata]
    if missing:
        raise ValueError(f"FAISS chunk metadata is missing: {', '.join(missing)}")
    return (
        f"{document.metadata['candidate_id']}:"
        f"line-{document.metadata['source_line']}:"
        f"chunk-{document.metadata['chunk_index']}"
    )


def _as_normalized_matrix(vectors: Sequence[Sequence[float]]) -> np.ndarray:
    try:
        matrix = np.asarray(vectors, dtype=np.float32)
    except ValueError as exc:
        raise ValueError("Embedding vectors must have consistent dimensions.") from exc
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("Embedding vectors must form a non-empty two-dimensional matrix.")
    if not np.isfinite(matrix).all():
        raise ValueError("Embedding vectors must contain only finite values.")
    faiss.normalize_L2(matrix)
    return matrix
