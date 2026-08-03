"""Practical bias-risk guardrails for recruiter queries and search results."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import re

from talent_search.retrieval import CandidateMatch


_PROTECTED_TERM_PATTERN = re.compile(
    r"\b(?:"
    r"young|old|older|age|aged|male|female|man|men|woman|women|gender|"
    r"race|racial|ethnicity|ethnic|religion|religious|muslim|christian|"
    r"jewish|nationality|married|single|pregnant|disabled|disability"
    r")\b",
    re.IGNORECASE,
)
_JOB_RELATED_TOKEN_PATTERN = re.compile(r"[a-z0-9+#.]+", re.IGNORECASE)
_SEARCH_FILLER = {
    "a",
    "an",
    "and",
    "candidate",
    "candidates",
    "find",
    "for",
    "looking",
    "or",
    "the",
    "with",
}
_SENSITIVE_METADATA_KEYS = {
    "age",
    "birth_date",
    "date_of_birth",
    "disability",
    "ethnicity",
    "gender",
    "marital_status",
    "name",
    "nationality",
    "race",
    "religion",
    "sex",
}
_EMAIL_PATTERN = re.compile(r"\b[^\s@]+@[^\s@]+\.[^\s@]+\b")
_URL_PATTERN = re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE)
_PHONE_PATTERN = re.compile(r"(?<!\w)(?:\+?\d[\s().-]*){8,}(?!\w)")
_ANONYMOUS_ID_PATTERN = re.compile(r"candidate-[a-z0-9-]+\Z", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class QuerySafetyResult:
    """A job-focused query plus any protected terms removed from it."""

    original_query: str
    cleaned_query: str
    protected_terms: tuple[str, ...]
    was_modified: bool


@dataclass(frozen=True, slots=True)
class BiasAuditReport:
    """Observable safety checks for one result set."""

    passed: bool
    unsafe_evidence_count: int
    sensitive_metadata_keys: tuple[str, ...]
    flags: tuple[str, ...]
    disclaimer: str


@dataclass(frozen=True, slots=True)
class CounterfactualRankingReport:
    """Whether neutral query rewrites preserve candidate ranking."""

    counterfactual_consistent: bool
    changed_variants: tuple[str, ...]
    disclaimer: str


def sanitize_recruiter_query(query: str) -> QuerySafetyResult:
    """Remove protected-demographic filters before resume retrieval."""

    original = query.strip()
    if not original:
        raise ValueError("The recruiter query cannot be blank.")

    protected_terms = tuple(
        dict.fromkeys(match.group(0).casefold() for match in _PROTECTED_TERM_PATTERN.finditer(original))
    )
    cleaned = _PROTECTED_TERM_PATTERN.sub(" ", original)
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"\ba (?=[aeiou])", "an ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+([,;:.!?])", r"\1", cleaned).strip(" ,;:")
    meaningful_tokens = [
        token.casefold()
        for token in _JOB_RELATED_TOKEN_PATTERN.findall(cleaned)
        if token.casefold() not in _SEARCH_FILLER
    ]
    if not meaningful_tokens:
        raise ValueError(
            "The query needs job-related skills, experience, or qualifications."
        )

    return QuerySafetyResult(
        original_query=original,
        cleaned_query=cleaned,
        protected_terms=protected_terms,
        was_modified=cleaned != original,
    )


def audit_candidate_matches(
    matches: Sequence[CandidateMatch],
) -> BiasAuditReport:
    """Audit privacy markers, sensitive metadata, PII patterns, and IDs."""

    unsafe_evidence_count = 0
    sensitive_keys: set[str] = set()
    flags: list[str] = []

    for match in matches:
        if not _ANONYMOUS_ID_PATTERN.fullmatch(match.candidate_id):
            flags.append(f"Candidate ID is not anonymous: {match.candidate_id}")
        for evidence in match.evidence:
            metadata = evidence.document.metadata
            found_sensitive = {
                str(key).casefold()
                for key in metadata
                if str(key).casefold() in _SENSITIVE_METADATA_KEYS
            }
            sensitive_keys.update(found_sensitive)
            text = evidence.document.page_content
            has_pii_pattern = any(
                pattern.search(text)
                for pattern in (_EMAIL_PATTERN, _URL_PATTERN, _PHONE_PATTERN)
            )
            if metadata.get("privacy_masked") is not True or has_pii_pattern:
                unsafe_evidence_count += 1

    if unsafe_evidence_count:
        flags.append(
            f"{unsafe_evidence_count} evidence chunk(s) failed the privacy audit."
        )
    if sensitive_keys:
        flags.append(
            "Sensitive metadata was present: " + ", ".join(sorted(sensitive_keys))
        )

    return BiasAuditReport(
        passed=not flags,
        unsafe_evidence_count=unsafe_evidence_count,
        sensitive_metadata_keys=tuple(sorted(sensitive_keys)),
        flags=tuple(flags),
        disclaimer=(
            "These checks reduce observable bias risks; passing them does not prove "
            "that the data, retrieval model, or hiring process is unbiased."
        ),
    )


def compare_counterfactual_rankings(
    baseline: Sequence[CandidateMatch],
    variants: Mapping[str, Sequence[CandidateMatch]],
) -> CounterfactualRankingReport:
    """Compare candidate order across job-equivalent neutral query rewrites."""

    baseline_ids = [match.candidate_id for match in baseline]
    changed = tuple(
        label
        for label, matches in variants.items()
        if [match.candidate_id for match in matches] != baseline_ids
    )
    return CounterfactualRankingReport(
        counterfactual_consistent=not changed,
        changed_variants=changed,
        disclaimer=(
            "Counterfactual stability is a diagnostic for the tested rewrites, not "
            "a guarantee of fairness across people or groups."
        ),
    )
