from __future__ import annotations

import json
from pathlib import Path

from langchain_core.documents import Document
import pytest

from talent_search.loaders import ResumeDatasetError, ResumeJsonlLoader


def test_loader_converts_each_jsonl_record_to_a_langchain_document(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "resumes.jsonl"
    annotation = {
        "label": ["Skills"],
        "points": [{"start": 17, "end": 27, "text": "SQL, Python"}],
    }
    dataset_path.write_text(
        json.dumps(
            {
                "content": "Junior analyst\nSQL, Python",
                "annotation": [annotation],
                "extras": None,
            }
        ),
        encoding="utf-8",
    )

    documents = ResumeJsonlLoader(dataset_path).load()

    assert documents == [
        Document(
            page_content="Junior analyst\nSQL, Python",
            metadata={
                "source": str(dataset_path),
                "source_line": 1,
                "annotations": [annotation],
                "extras": None,
            },
        )
    ]


def test_loader_reports_the_source_line_for_invalid_json(tmp_path: Path) -> None:
    dataset_path = tmp_path / "invalid.jsonl"
    dataset_path.write_text(
        '{"content": "Valid", "annotation": [], "extras": null}\n{broken',
        encoding="utf-8",
    )

    with pytest.raises(ResumeDatasetError, match="line 2"):
        ResumeJsonlLoader(dataset_path).load()


@pytest.mark.parametrize(
    "record, expected_message",
    [
        ({"annotation": [], "extras": None}, "content"),
        ({"content": "   ", "annotation": [], "extras": None}, "content"),
        ({"content": "Resume", "annotation": {}, "extras": None}, "annotation"),
    ],
)
def test_loader_rejects_records_that_break_the_dataset_contract(
    tmp_path: Path,
    record: dict,
    expected_message: str,
) -> None:
    dataset_path = tmp_path / "invalid-record.jsonl"
    dataset_path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ResumeDatasetError, match=expected_message):
        ResumeJsonlLoader(dataset_path).load()
