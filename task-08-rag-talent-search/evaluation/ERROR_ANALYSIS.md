# Retrieval Error Analysis

## Case: machine-learning-flask-001

Query:

```text
machine learning candidate with Flask experience
```

The privacy-safe ground truth contains one anonymous relevant candidate. Both
required terms occur in that candidate's `WORK EXPERIENCE` chunks. No resume
text or direct identifier is included in this report.

## Trace through the pipeline

| Boundary | Observed result | Conclusion |
|---|---:|---|
| Sanitized chunks | Both required terms remain in two chunks | Evidence was not removed by privacy processing |
| Exact FAISS chunk search | Relevant chunks ranked 8 and 9 | Evidence was embedded and is retrievable |
| `fetch_k=50` | Relevant candidate was present | Retrieval depth did not remove the candidate |
| Candidate aggregation | Candidate ranked 6 | Best-chunk grouping preserved its strongest score |
| Final `candidate_k=3` | Candidate was excluded | The top-three cutoff exposed the earlier scoring order |

The relevant chunk scores were `0.4127` and `0.4082`. Among the top 10 chunks,
seven contained neither required term, one contained only `machine learning`,
and two contained both terms. Among the top 50 chunks, 45 contained neither
term, three contained only `machine learning`, none contained only `Flask`, and
two contained both.

## Query-wording hypothesis

The first hypothesis was that generic words such as `candidate` and
`experience` diluted the query. Controlled query variants did not support it:

| Query form | Relevant candidate rank | Relevant chunk rank | In top 3? |
|---|---:|---:|---|
| Original natural query | 6 | 8 | No |
| Core terms only | 7 | 10 | No |
| Required-skills wording | 24 | 28 | No |

## Root cause

The current retriever uses dense semantic similarity alone. It rewards chunks
whose overall meaning resembles a machine-learning profile, but it does not
enforce the recruiter constraint that both `machine learning` **and** `Flask`
must be present. FAISS correctly returns the cosine-similarity order produced
by the embedding model; candidate aggregation correctly ranks candidates by
their best chunk.

This is therefore a retrieval-objective mismatch, not a missing-document,
privacy, FAISS-depth, or aggregation bug.

## Decision boundary

This case was not used alone for tuning. The follow-up baseline now contains
eight manually labelled recruiter queries spanning skills, domain experience,
seniority, leadership, and multiple constraints. Its mean candidate Recall@3
is `0.281` and mean Precision@3 is `0.250`, confirming that the dense-only
limitation is broader than this first case. The next controlled experiment can
therefore compare hybrid lexical-plus-dense retrieval or explicit must-have
coverage against the unchanged baseline.
