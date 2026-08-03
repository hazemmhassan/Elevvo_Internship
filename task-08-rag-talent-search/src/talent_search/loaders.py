"""LangChain document loaders for resume datasets."""

from __future__ import annotations

from collections.abc import Iterator
import json
from pathlib import Path

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document


class ResumeDatasetError(ValueError):
    """Raised when a resume dataset record is malformed."""


class ResumeJsonlLoader(BaseLoader):
    """Load resume records from a JSON Lines file."""

    def __init__(self, file_path: str | Path) -> None:
        self.file_path = Path(file_path)

    def lazy_load(self) -> Iterator[Document]:
        with self.file_path.open("r", encoding="utf-8") as dataset:
            for source_line, line in enumerate(dataset, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ResumeDatasetError(
                        f"Invalid JSON in {self.file_path} at line {source_line}: {exc.msg}"
                    ) from exc
                content = record.get("content")
                if not isinstance(content, str) or not content.strip():
                    raise ResumeDatasetError(
                        f"Record at line {source_line} needs non-empty string content."
                    )
                annotations = record.get("annotation")
                if not isinstance(annotations, list):
                    raise ResumeDatasetError(
                        f"Record at line {source_line} needs an annotation list."
                    )
                yield Document(
                    page_content=content,
                    metadata={
                        "source": str(self.file_path),
                        "source_line": source_line,
                        "annotations": annotations,
                        "extras": record.get("extras"),
                    },
                )
