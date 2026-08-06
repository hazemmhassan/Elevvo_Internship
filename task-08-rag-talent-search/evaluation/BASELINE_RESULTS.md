# Eight-Case Dense Retrieval Baseline

This report measures the current dense-only candidate retriever at
`candidate_k=3`, `fetch_k=50`, and three evidence sections per candidate. The
eight ground-truth cases were labelled independently from privacy-safe chunks
and approved job-related annotations. No raw resume text or direct identifier
is stored here.

## Aggregate results

| Metric | Result |
|---|---:|
| Cases | 8 |
| Hit rate at 3 | 0.500 |
| Mean candidate Recall@3 | 0.281 |
| Mean candidate Precision@3 | 0.250 |
| Mean reciprocal rank | 0.438 |
| Mean evidence-section recall at 3 | 0.219 |

`Hit rate at 3 = 0.500` means at least one relevant candidate appeared in the
top three for four of the eight queries. It does not mean half of all relevant
candidates were retrieved; mean candidate recall measures that separately.

## Results by case

| Case | Type | Recall@3 | Precision@3 | First relevant rank | Evidence recall |
|---|---|---:|---:|---:|---:|
| Machine learning + Flask | Exact skills | 0.000 | 0.000 | Not found | 0.000 |
| SQL + Tableau | Broad exact skills | 0.250 | 0.333 | 1 | 0.250 |
| AWS + Docker + Kubernetes | Multiple constraints | 0.500 | 0.333 | 2 | 0.500 |
| Python + Django | Framework skills | 1.000 | 1.000 | 1 | 1.000 |
| Natural language processing + Python | Domain experience | 0.000 | 0.000 | Not found | 0.000 |
| Forecasting + Python | Cross-section experience | 0.500 | 0.333 | 1 | 0.000 |
| Senior manager + three data tools | Seniority and constraints | 0.000 | 0.000 | Not found | 0.000 |
| Team lead + Agile | Leadership and methodology | 0.000 | 0.000 | Not found | 0.000 |

## Interpretation

The dense retriever succeeds on some closely expressed technology pairs, most
clearly Python and Django. It is unreliable when a query contains several
must-have constraints, a role or seniority requirement, or evidence distributed
across sections.

The forecasting case also separates candidate retrieval from evidence quality:
one relevant candidate ranked first, but the selected evidence sections did not
include the manually expected work-experience and skills combination.

The next retrieval experiment should combine semantic similarity with explicit
term or structured-entity coverage. It must be compared against this unchanged
baseline on all eight cases; improving only one case would be overfitting.

## Phrase-aware hybrid comparison

The implemented experiment combines candidate-level BM25, exact term coverage,
multiword phrase coverage, and dense FAISS ranks through reciprocal-rank fusion.
The evaluation labels and top-three limit remained unchanged.

| Metric | Dense baseline | Hybrid | Change |
|---|---:|---:|---:|
| Hit rate at 3 | 0.500 | 1.000 | +0.500 |
| Mean candidate Recall@3 | 0.281 | 0.906 | +0.625 |
| Mean candidate Precision@3 | 0.250 | 0.625 | +0.375 |
| Mean reciprocal rank | 0.438 | 0.938 | +0.500 |
| Mean evidence-section recall at 3 | 0.219 | 0.875 | +0.656 |

The first hybrid run reached `0.844` mean recall but still missed the team-lead
case because token matching treated `team lead` as two unrelated words. A
general phrase-preservation regression test exposed that issue. After the
phrase-aware fix, every case returned at least one relevant candidate.

This is strong evidence that constraint-aware retrieval is more appropriate
for these recruiter queries. It is not a production-quality estimate: the set
contains only eight cases and is intentionally rich in explicit terms. Future
real recruiter queries and semantic paraphrases must be added before making
broad quality claims.
