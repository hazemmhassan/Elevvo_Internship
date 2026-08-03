"""Local embedding configuration and safety checks."""

from __future__ import annotations

from typing import Protocol, Sequence

from langchain_core.documents import Document


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class DocumentEmbeddingModel(Protocol):
    def embed_documents(self, texts: list[str]) -> Sequence[Sequence[float]]:
        """Convert texts into equally sized numeric vectors."""


def create_local_embeddings(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    device: str = "cpu",
):
    """Create a normalized Hugging Face embedding model that runs locally."""

    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
        query_encode_kwargs={"normalize_embeddings": True},
        show_progress=False,
    )


def embed_documents(
    documents: Sequence[Document], embedding_model: DocumentEmbeddingModel
) -> list[list[float]]:
    """Embed privacy-safe Documents and validate the returned matrix shape."""

    if any(document.metadata.get("privacy_masked") is not True for document in documents):
        raise ValueError("Every document must pass privacy masking before embedding.")
    if not documents:
        return []

    raw_vectors = embedding_model.embed_documents(
        [document.page_content for document in documents]
    )
    vectors = [[float(value) for value in vector] for vector in raw_vectors]

    if len(vectors) != len(documents):
        raise ValueError("Embedding model returned a different number of vectors.")
    dimensions = {len(vector) for vector in vectors}
    if len(dimensions) != 1 or dimensions == {0}:
        raise ValueError("Every embedding vector must have the same non-zero dimension.")
    return vectors
