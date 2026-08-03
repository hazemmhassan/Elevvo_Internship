from __future__ import annotations

from langchain_core.documents import Document

from talent_search.privacy import sanitize_resume_document


def test_sanitizer_masks_identity_and_contact_details_but_keeps_job_evidence() -> None:
    raw_document = Document(
        page_content=(
            "Jane Example\n"
            "Cairo, Egypt | jane@example.com | +20 100 123 4567\n"
            "https://indeed.com/r/Jane-Example/abc123\n\n"
            "SKILLS\nSQL, Python, Tableau"
        ),
        metadata={
            "source": "resumes.jsonl",
            "source_line": 7,
            "annotations": [
                {
                    "label": ["Name"],
                    "points": [{"start": 0, "end": 11, "text": "Jane Example"}],
                },
                {
                    "label": ["Location"],
                    "points": [{"start": 13, "end": 24, "text": "Cairo, Egypt"}],
                },
                {
                    "label": ["Skills"],
                    "points": [{"start": 112, "end": 131, "text": "SQL, Python, Tableau"}],
                },
            ],
            "extras": {"private": "value"},
        },
    )

    sanitized = sanitize_resume_document(raw_document)

    assert "Jane Example" not in sanitized.page_content
    assert "Cairo, Egypt" not in sanitized.page_content
    assert "jane@example.com" not in sanitized.page_content
    assert "+20 100 123 4567" not in sanitized.page_content
    assert "indeed.com/r/" not in sanitized.page_content
    assert "SQL, Python, Tableau" in sanitized.page_content
    assert sanitized.metadata["privacy_masked"] is True
    assert sanitized.metadata["candidate_id"].startswith("candidate-")
    assert sanitized.metadata["safe_entities"] == {
        "Skills": ["SQL, Python, Tableau"]
    }
    assert "annotations" not in sanitized.metadata
    assert "extras" not in sanitized.metadata


def test_sanitizer_uses_annotation_text_when_dataset_offsets_are_wrong() -> None:
    raw_document = Document(
        page_content="Alex Example\nWORK EXPERIENCE\nData Analyst",
        metadata={
            "source": "resumes.jsonl",
            "source_line": 1,
            "annotations": [
                {
                    "label": ["Name"],
                    "points": [
                        {"start": 999, "end": 1005, "text": "Alex Example"}
                    ],
                }
            ],
        },
    )

    sanitized = sanitize_resume_document(raw_document)

    assert "Alex Example" not in sanitized.page_content
    assert "Data Analyst" in sanitized.page_content


def test_candidate_id_is_stable_for_the_same_resume() -> None:
    raw_document = Document(
        page_content="A Person\nSKILLS\nSQL",
        metadata={"source": "one.jsonl", "source_line": 2, "annotations": []},
    )

    first = sanitize_resume_document(raw_document)
    second = sanitize_resume_document(raw_document)

    assert first.metadata["candidate_id"] == second.metadata["candidate_id"]


def test_safe_annotation_summaries_cannot_reintroduce_contact_details() -> None:
    raw_document = Document(
        page_content="A Person\nSKILLS\nSQL and https://portfolio.example/profile",
        metadata={
            "source": "one.jsonl",
            "source_line": 3,
            "annotations": [
                {
                    "label": ["Skills"],
                    "points": [
                        {
                            "start": 16,
                            "end": 56,
                            "text": "SQL and https://portfolio.example/profile",
                        }
                    ],
                }
            ],
        },
    )

    sanitized = sanitize_resume_document(raw_document)

    assert sanitized.metadata["safe_entities"] == {"Skills": ["SQL and [URL]"]}


def test_phone_masking_does_not_erase_employment_year_ranges() -> None:
    raw_document = Document(
        page_content="A Person\nWORK EXPERIENCE\nData Analyst | 2018 - 2022",
        metadata={"source": "one.jsonl", "source_line": 4, "annotations": []},
    )

    sanitized = sanitize_resume_document(raw_document)

    assert "2018 - 2022" in sanitized.page_content
