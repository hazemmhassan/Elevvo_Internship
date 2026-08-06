from __future__ import annotations

from langchain_core.documents import Document
import pytest

from talent_search.bias import (
    audit_candidate_matches,
    compare_counterfactual_rankings,
    sanitize_recruiter_query,
)
from talent_search.retrieval import CandidateMatch, ChunkEvidence


def match(
    candidate_id: str,
    *,
    metadata: dict[str, object] | None = None,
) -> CandidateMatch:
    safe_metadata: dict[str, object] = {
        "candidate_id": candidate_id,
        "section": "SKILLS",
        "privacy_masked": True,
    }
    safe_metadata.update(metadata or {})
    document = Document(
        page_content="SKILLS\nSQL Python Tableau",
        metadata=safe_metadata,
    )
    return CandidateMatch(
        candidate_id=candidate_id,
        score=1.0,
        evidence=(ChunkEvidence(document=document, score=0.8),),
    )


def test_query_sanitizer_removes_protected_preferences_but_keeps_job_terms() -> None:
    result = sanitize_recruiter_query(
        "Find a young female data analyst with SQL and Tableau"
    )

    assert result.cleaned_query == "Find a data analyst with SQL and Tableau"
    assert set(result.protected_terms) == {"young", "female"}
    assert result.was_modified is True


def test_query_sanitizer_rejects_a_protected_only_search() -> None:
    with pytest.raises(ValueError, match="job-related"):
        sanitize_recruiter_query("young female")


def test_bias_audit_flags_sensitive_metadata_and_unmasked_evidence() -> None:
    report = audit_candidate_matches(
        [
            match(
                "candidate-a",
                metadata={"privacy_masked": False, "gender": "redacted"},
            )
        ]
    )

    assert report.passed is False
    assert report.unsafe_evidence_count == 1
    assert report.sensitive_metadata_keys == ("gender",)
    assert report.flags


def test_counterfactual_comparison_flags_changed_rankings() -> None:
    report = compare_counterfactual_rankings(
        [match("candidate-a"), match("candidate-b")],
        {
            "gender-neutral rewrite": [
                match("candidate-b"),
                match("candidate-a"),
            ]
        },
    )

    assert report.counterfactual_consistent is False
    assert "gender-neutral rewrite" in report.changed_variants


def test_safe_bias_audit_passes_with_an_honest_limitation() -> None:
    report = audit_candidate_matches([match("candidate-a")])

    assert report.passed is True
    assert report.unsafe_evidence_count == 0
    assert "does not prove" in report.disclaimer.lower()
