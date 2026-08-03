"""Privacy-safe preprocessing for raw resume documents."""

from __future__ import annotations

from hashlib import sha256
import re
from typing import Any

from langchain_core.documents import Document


SENSITIVE_LABELS = {"name", "email address", "location"}
SEARCH_SAFE_LABELS = {
    "skills",
    "college name",
    "graduation year",
    "designation",
    "companies worked at",
    "degree",
    "years of experience",
}

EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
URL_PATTERN = re.compile(
    r"(?:https?://|www\.|indeed\.com/)[^\s]+", re.IGNORECASE
)
PHONE_PATTERN = re.compile(r"(?<!\w)(?:\+?\d[\d \t().-]{6,}\d)(?!\w)")
YEAR_RANGE_PATTERN = re.compile(r"\d{4}\s*-\s*\d{4}")
CONTACT_LINE_PATTERN = re.compile(
    r"^.*(?:email me on indeed|indeed\.com/r/).*$", re.IGNORECASE | re.MULTILINE
)


def sanitize_resume_document(document: Document) -> Document:
    """Remove direct identifiers while retaining job-relevant evidence."""

    annotations = document.metadata.get("annotations", [])
    sensitive_values = _annotation_values(annotations, SENSITIVE_LABELS)
    safe_entities = _safe_entities(annotations, sensitive_values)

    sanitized_text = document.page_content
    for value in sorted(sensitive_values, key=len, reverse=True):
        sanitized_text = re.sub(
            re.escape(value),
            _placeholder_for(value, annotations),
            sanitized_text,
            flags=re.IGNORECASE,
        )

    sanitized_text = CONTACT_LINE_PATTERN.sub("[CONTACT REDACTED]", sanitized_text)
    sanitized_text = EMAIL_PATTERN.sub("[EMAIL]", sanitized_text)
    sanitized_text = URL_PATTERN.sub("[URL]", sanitized_text)
    sanitized_text = PHONE_PATTERN.sub(_replace_phone_candidate, sanitized_text)
    sanitized_text = _mask_first_nonempty_line(sanitized_text)

    candidate_hash = sha256(document.page_content.encode("utf-8")).hexdigest()[:12]
    safe_metadata = {
        key: document.metadata[key]
        for key in ("source", "source_line")
        if key in document.metadata
    }
    safe_metadata.update(
        {
            "candidate_id": f"candidate-{candidate_hash}",
            "privacy_masked": True,
            "safe_entities": safe_entities,
        }
    )

    return Document(page_content=sanitized_text, metadata=safe_metadata)


def _annotation_values(
    annotations: list[dict[str, Any]], allowed_labels: set[str]
) -> list[str]:
    values: list[str] = []
    for annotation in annotations:
        labels = annotation.get("label", [])
        if isinstance(labels, str):
            labels = [labels]
        if not any(str(label).casefold() in allowed_labels for label in labels):
            continue
        for point in annotation.get("points", []):
            value = str(point.get("text", "")).strip()
            if value:
                values.append(value)
    return values


def _safe_entities(
    annotations: list[dict[str, Any]], sensitive_values: list[str]
) -> dict[str, list[str]]:
    entities: dict[str, list[str]] = {}
    for annotation in annotations:
        labels = annotation.get("label", [])
        if isinstance(labels, str):
            labels = [labels]
        for label in labels:
            label_text = str(label)
            if label_text.casefold() not in SEARCH_SAFE_LABELS:
                continue
            for point in annotation.get("points", []):
                value = str(point.get("text", "")).strip()
                safe_value = _redact_annotation_value(value, sensitive_values)
                if safe_value and safe_value not in entities.setdefault(label_text, []):
                    entities[label_text].append(safe_value)
    return entities


def _redact_annotation_value(value: str, sensitive_values: list[str]) -> str:
    redacted = value
    for sensitive_value in sorted(sensitive_values, key=len, reverse=True):
        redacted = re.sub(
            re.escape(sensitive_value), "[REDACTED]", redacted, flags=re.IGNORECASE
        )
    redacted = EMAIL_PATTERN.sub("[EMAIL]", redacted)
    redacted = URL_PATTERN.sub("[URL]", redacted)
    redacted = PHONE_PATTERN.sub(_replace_phone_candidate, redacted)
    return redacted.strip()


def _placeholder_for(value: str, annotations: list[dict[str, Any]]) -> str:
    for annotation in annotations:
        labels = annotation.get("label", [])
        if isinstance(labels, str):
            labels = [labels]
        points = annotation.get("points", [])
        if any(str(point.get("text", "")).strip() == value for point in points):
            normalized_labels = {str(label).casefold() for label in labels}
            if "name" in normalized_labels:
                return "[NAME]"
            if "email address" in normalized_labels:
                return "[EMAIL]"
            if "location" in normalized_labels:
                return "[LOCATION]"
    return "[REDACTED]"


def _replace_phone_candidate(match: re.Match[str]) -> str:
    value = match.group(0)
    if YEAR_RANGE_PATTERN.fullmatch(value):
        return value
    return "[PHONE]" if sum(character.isdigit() for character in value) >= 8 else value


def _mask_first_nonempty_line(text: str) -> str:
    lines = text.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if line.strip():
            ending = "\n" if line.endswith("\n") else ""
            lines[index] = f"[NAME]{ending}"
            break
    return "".join(lines)
