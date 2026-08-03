"""Build a privacy-safe persisted FAISS index from the resume JSONL dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from talent_search.embeddings import (  # noqa: E402
    DEFAULT_EMBEDDING_MODEL,
    create_local_embeddings,
)
from talent_search.loaders import ResumeJsonlLoader  # noqa: E402
from talent_search.privacy import sanitize_resume_document  # noqa: E402
from talent_search.splitting import ResumeSectionSplitter  # noqa: E402
from talent_search.vector_store import build_faiss_index, save_faiss_index  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the anonymized local FAISS resume index."
    )
    parser.add_argument("dataset", type=Path, help="Resume JSONL dataset path.")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / ".artifacts" / "faiss_index",
        help="New index directory; existing directories are never overwritten.",
    )
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_documents = ResumeJsonlLoader(args.dataset).load()
    safe_documents = [sanitize_resume_document(document) for document in raw_documents]
    chunks = ResumeSectionSplitter().split_documents(safe_documents)
    embeddings = create_local_embeddings(args.model, device=args.device)
    vector_store = build_faiss_index(chunks, embeddings)
    manifest = save_faiss_index(
        vector_store,
        args.output,
        embedding_model_name=args.model,
    )
    print(f"Candidates processed: {len(safe_documents)}")
    print(f"Privacy-safe chunks indexed: {manifest.vector_count}")
    print(f"Embedding dimensions: {manifest.dimensions}")
    print(f"Saved index: {args.output.resolve()}")


if __name__ == "__main__":
    main()
