"""Section-aware splitting for sanitized resume documents."""

from __future__ import annotations

import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


SECTION_HEADINGS = {
    "SUMMARY": "SUMMARY",
    "PROFESSIONAL SUMMARY": "SUMMARY",
    "OBJECTIVE": "SUMMARY",
    "WORK EXPERIENCE": "WORK EXPERIENCE",
    "PROFESSIONAL EXPERIENCE": "WORK EXPERIENCE",
    "EXPERIENCE": "WORK EXPERIENCE",
    "EDUCATION": "EDUCATION",
    "SKILLS": "SKILLS",
    "TECHNICAL SKILLS": "SKILLS",
    "PROJECTS": "PROJECTS",
    "CERTIFICATIONS": "CERTIFICATIONS",
    "ACCOMPLISHMENTS": "ACCOMPLISHMENTS",
    "ADDITIONAL INFORMATION": "ADDITIONAL INFORMATION",
    "LANGUAGES": "LANGUAGES",
    "INTERESTS": "INTERESTS",
}

ENTITY_DISPLAY_ORDER = (
    "Designation",
    "Years of Experience",
    "Skills",
    "Degree",
    "College Name",
    "Graduation Year",
    "Companies worked at",
)


class ResumeSectionSplitter:
    """Turn one privacy-safe resume into small, meaningful Documents."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be non-negative and below chunk_size.")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_document(self, document: Document) -> list[Document]:
        if document.metadata.get("privacy_masked") is not True:
            raise ValueError("Resume must pass privacy masking before it can be split.")

        sections = self._extract_sections(document.page_content)
        safe_entities = document.metadata.get("safe_entities", {})
        summary = self._candidate_summary(safe_entities)
        if summary:
            sections.insert(0, ("CANDIDATE SUMMARY", summary))

        base_metadata = {
            key: document.metadata[key]
            for key in ("source", "source_line", "candidate_id", "privacy_masked")
            if key in document.metadata
        }
        results: list[Document] = []
        for section_name, section_text in sections:
            for section_chunk_index, text in enumerate(
                self._split_section(section_name, section_text)
            ):
                metadata = {
                    **base_metadata,
                    "section": section_name,
                    "section_chunk_index": section_chunk_index,
                    "chunk_index": len(results),
                }
                results.append(Document(page_content=text, metadata=metadata))
        return results

    def split_documents(self, documents: list[Document]) -> list[Document]:
        return [chunk for document in documents for chunk in self.split_document(document)]

    def _extract_sections(self, text: str) -> list[tuple[str, str]]:
        sections: list[tuple[str, str]] = []
        current_name = "PROFILE"
        current_lines: list[str] = []

        def save_current() -> None:
            body = "\n".join(current_lines).strip()
            if body and _has_searchable_text(body):
                sections.append((current_name, body))

        for line in text.splitlines():
            normalized = " ".join(line.strip().upper().split())
            if normalized in SECTION_HEADINGS:
                save_current()
                current_name = SECTION_HEADINGS[normalized]
                current_lines = []
            else:
                current_lines.append(line)
        save_current()
        return sections

    def _split_section(self, section_name: str, section_text: str) -> list[str]:
        prefix = f"{section_name}\n"
        available_size = self.chunk_size - len(prefix)
        if available_size <= 0:
            raise ValueError("chunk_size is too small to include the section heading.")

        if len(section_text) <= available_size:
            pieces = [section_text]
        else:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=available_size,
                chunk_overlap=min(self.chunk_overlap, available_size - 1),
                length_function=len,
                is_separator_regex=False,
            )
            pieces = splitter.split_text(section_text)
        return [f"{prefix}{piece}" for piece in pieces if piece.strip()]

    @staticmethod
    def _candidate_summary(safe_entities: dict[str, list[str]]) -> str:
        lines: list[str] = []
        for label in ENTITY_DISPLAY_ORDER:
            values = safe_entities.get(label, [])
            if values:
                lines.append(f"{label}: {', '.join(values)}")
        return "\n".join(lines)


def _has_searchable_text(text: str) -> bool:
    without_placeholders = re.sub(r"\[[A-Z ]+\]", "", text)
    return bool(re.search(r"[A-Za-z0-9]", without_placeholders))
