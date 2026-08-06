from __future__ import annotations

from langchain_core.documents import Document
import pytest

from talent_search.splitting import ResumeSectionSplitter


def test_splitter_creates_searchable_resume_sections_with_safe_metadata() -> None:
    sanitized = Document(
        page_content=(
            "[NAME]\nData Analyst\n\n"
            "WORK EXPERIENCE\nBuilt Tableau dashboards with SQL.\n\n"
            "EDUCATION\nBSc in Information Systems\n\n"
            "SKILLS\nSQL, Python, Tableau"
        ),
        metadata={
            "source": "resumes.jsonl",
            "source_line": 4,
            "candidate_id": "candidate-123",
            "privacy_masked": True,
            "safe_entities": {
                "Designation": ["Data Analyst"],
                "Skills": ["SQL, Python, Tableau"],
            },
        },
    )

    chunks = ResumeSectionSplitter(chunk_size=200, chunk_overlap=20).split_document(
        sanitized
    )

    assert [chunk.metadata["section"] for chunk in chunks] == [
        "CANDIDATE SUMMARY",
        "PROFILE",
        "WORK EXPERIENCE",
        "EDUCATION",
        "SKILLS",
    ]
    assert "Designation: Data Analyst" in chunks[0].page_content
    assert all(chunk.metadata["candidate_id"] == "candidate-123" for chunk in chunks)
    assert [chunk.metadata["chunk_index"] for chunk in chunks] == list(
        range(len(chunks))
    )
    assert all("safe_entities" not in chunk.metadata for chunk in chunks)
    assert all("annotations" not in chunk.metadata for chunk in chunks)


def test_splitter_recursively_breaks_an_oversized_section() -> None:
    paragraph_one = "Built SQL dashboards for weekly reporting. " * 4
    paragraph_two = "Automated Python data quality checks. " * 4
    sanitized = Document(
        page_content=f"[NAME]\nWORK EXPERIENCE\n{paragraph_one}\n\n{paragraph_two}",
        metadata={
            "source": "resumes.jsonl",
            "source_line": 1,
            "candidate_id": "candidate-long",
            "privacy_masked": True,
            "safe_entities": {},
        },
    )

    chunks = ResumeSectionSplitter(chunk_size=120, chunk_overlap=20).split_document(
        sanitized
    )
    experience_chunks = [
        chunk for chunk in chunks if chunk.metadata["section"] == "WORK EXPERIENCE"
    ]

    assert len(experience_chunks) > 1
    assert all(len(chunk.page_content) <= 120 for chunk in experience_chunks)
    assert [chunk.metadata["section_chunk_index"] for chunk in experience_chunks] == list(
        range(len(experience_chunks))
    )
    assert all(
        chunk.page_content.startswith("WORK EXPERIENCE\n")
        for chunk in experience_chunks
    )


def test_splitter_rejects_a_raw_resume_to_protect_privacy() -> None:
    raw_document = Document(
        page_content="Jane Example\nSKILLS\nSQL",
        metadata={"candidate_id": "candidate-raw"},
    )

    with pytest.raises(ValueError, match="privacy"):
        ResumeSectionSplitter().split_document(raw_document)
