# Learning Log

## Phase 1, Step 1: Raw dataset loading

### Goal

Convert each line of the Resume Entities for NER JSONL file into one raw
LangChain `Document` while validating the minimum dataset contract.

### Concepts learned

- JSONL stores one complete JSON object per line. It is not one large JSON array.
- A LangChain loader connects an external data source to LangChain's common
  `Document` interface.
- `Document.page_content` contains the searchable text.
- `Document.metadata` carries source information and other context.
- `BaseLoader.lazy_load()` yields documents incrementally; the inherited
  `load()` convenience method collects them into a list.
- Raw annotations are preserved temporarily because the next step needs them for
  privacy masking.

### Test-first cycle

1. A valid JSONL record was expected to become one `Document`.
2. The empty loader failed that test.
3. Minimal parsing made the test pass.
4. Additional failing tests required useful errors for malformed JSON, missing
   content, blank content, and invalid annotations.

### Important boundary

The Step 1 output is raw and contains personal information. It is safe only as
an in-memory input to the next preprocessing step. It is not ready for
embeddings, FAISS, or an LLM.

### Not started

- Formal retrieval evaluation
- LLM evaluation
- Streamlit UI

## Phase 1, Step 2: Privacy-safe resume documents

### Goal

Prevent direct identity and contact information from reaching embeddings,
vector storage, or an LLM while retaining evidence needed for talent search.

### What the sanitizer does

- Replaces annotated names, locations, and email addresses using annotation text
  rather than trusting annotation offsets.
- Detects common email, URL, profile-link, and phone patterns.
- Masks the first non-empty line because this dataset conventionally stores the
  candidate name there.
- Creates a stable anonymous `candidate_id` from a one-way SHA-256 digest of the
  original resume text.
- Keeps job-related annotation values such as skills, designations, degrees, and
  years of experience for an anonymous candidate summary.
- Removes raw annotations and `extras` from downstream metadata.

The layers are intentional. Dataset annotations are useful but imperfect: the
inspection found 223 annotation span/text mismatches. Pattern matching catches
contact details that annotations miss, while the dataset-specific header rule
covers unannotated names.

### Test-first lessons

The tests exposed two easy-to-miss leaks:

1. URLs could reappear through otherwise safe annotation summaries.
2. A broad phone pattern could incorrectly erase employment ranges such as
   `2018 - 2022`.

Both paths now have regression tests. On all 220 resumes, the final chunk audit
found no email or URL patterns and no raw annotation metadata.

## Phase 1, Step 3: Section-aware chunking

### Goal

Create search units that each carry one reasonably focused meaning.

A whole resume may discuss unrelated things at once. If it has only one vector,
strong education text can dilute relevant work experience. We therefore divide
the resume at headings such as `WORK EXPERIENCE`, `EDUCATION`, and `SKILLS`.

Each candidate also receives an anonymous `CANDIDATE SUMMARY` built from the
job-related entities kept during sanitization. This makes structured signals
such as skills and job titles easy to retrieve without retaining identity.

If a section is still too large, LangChain's
`RecursiveCharacterTextSplitter` tries paragraph breaks, then line breaks, then
spaces. This keeps natural groups together when possible. The baseline settings
are 1,000 characters with 150 characters of overlap. Overlap repeats a little
context so a fact near a boundary is less likely to lose its meaning.

Every output chunk includes:

- anonymous candidate ID;
- source row;
- section name;
- chunk index within the candidate;
- chunk index within the section;
- confirmation that privacy masking occurred.

The verified run produced 1,956 chunks, with a maximum length of 998 characters.

## Phase 1, Step 4: Local embeddings

### Goal

Turn each privacy-safe text chunk into numbers that represent its meaning.

The project uses LangChain's `HuggingFaceEmbeddings` with
`sentence-transformers/all-MiniLM-L6-v2`. It runs locally and maps each short
text to 384 floating-point numbers. The vectors are normalized to length 1 so
their dot product is cosine similarity.

A synthetic check compared the query `junior data analyst with SQL and Tableau`
with two invented passages. The SQL/Tableau passage scored `0.7238`; an unrelated
retail passage scored `0.1555`. The expected passage was therefore closer in the
embedding space.

The real in-memory run produced a `1956 x 384` `float32` matrix. All values were
finite and every vector norm was 1. Nothing was persisted.

### Current boundary

Embeddings are ready for a vector index. At this historical checkpoint, FAISS
had not started and required the next approval.

## Phase 1, Step 5: Exact FAISS chunk search

### Goal

Store the 1,956 embedding vectors in a searchable index and recover the original
privacy-safe LangChain chunks for a natural-language recruiter query.

### Why an exact flat index

This dataset is small enough to compare a query against every stored vector.
`IndexFlatIP` therefore gives exact results without training, clustering, or an
approximation tradeoff. Both stored vectors and query vectors are normalized,
so the inner product returned by FAISS is cosine similarity: larger scores mean
closer semantic meaning.

### LangChain and FAISS responsibilities

- `faiss-cpu` stores normalized `float32` vectors and returns vector positions.
- `FaissVectorStore` implements LangChain Core's `VectorStore` interface.
- A unique ID connects each vector position to its anonymous candidate, source
  row, and chunk index.
- The document map recovers the matching `Document` text and safe metadata.

The adapter is implemented in the project instead of using
`langchain-community`, which is sunset and archived. This keeps the integration
small, visible, and based on maintained LangChain Core abstractions.

### Test-first cycle

The tests first failed because `talent_search.vector_store` did not exist. The
minimal implementation then had to demonstrate observable behavior:

- index every privacy-safe chunk;
- return the SQL chunk for a SQL query with its metadata and score;
- reject raw documents, empty input, duplicate chunk identities, blank queries,
  and non-positive result counts.

### Verified real run

The full pipeline produced an exact 384-dimensional index containing all 1,956
chunks. For `junior data analyst with SQL and Tableau`, the top 10 chunks
represented seven distinct anonymous candidates and returned cosine scores from
`0.5588` down to `0.4711`.

Several candidates appeared through more than one section. This proves why the
next layer must group chunks by `candidate_id` before ranking complete
candidates. A FAISS score measures semantic closeness, not hiring suitability or
a percentage probability.

### Current boundary

At this historical checkpoint, chunk search was ready in memory while candidate
aggregation, candidate ranking, index persistence/loading, LLM evaluation, bias
checks, and the Streamlit interface had not started.

## Phase 1, Step 6: Candidate-level aggregation and ranking

### Goal

Convert chunk-level FAISS results into a ranked list of distinct candidates
without allowing candidates with more chunks to receive an unfair score boost.

### Baseline ranking rule

The pipeline retrieves more chunks than the number of candidates it will
return. It then groups those chunks using anonymous `candidate_id` metadata.

For the first transparent baseline:

```text
candidate score = highest chunk similarity for that candidate
```

This is best-chunk scoring. A candidate with ten medium-quality chunks cannot
outrank another candidate solely because they have more text. It also avoids
inventing untested weights for work experience, skills, education, or other
sections.

The candidate keeps the strongest evidence chunk from up to three distinct
sections. Evidence supports later explanation but does not add ranking points.

### Test-first cycle

The tests first failed because `talent_search.retrieval` did not exist. The
implementation then had to demonstrate that it:

- returns unique candidates ordered by their best chunk;
- does not let multiple chunks inflate a candidate score;
- keeps only the strongest evidence chunk per section;
- rejects invalid candidate, evidence, and fetch limits;
- works with the real FAISS adapter rather than only a mocked search result.

### Verified real run

For `junior data analyst with SQL and Tableau`, the pipeline retrieved 50 chunks
and returned three distinct candidates:

1. candidate score `0.5588`, with work experience, summary, and profile evidence;
2. candidate score `0.5485`, with work experience, summary, and skills evidence;
3. candidate score `0.5344`, with work experience evidence available in the top
   50 chunks.

The scores were descending and all three candidate IDs were unique. This is a
functional verification of aggregation, not proof that the ranking is accurate
for real recruiters. That requires a retrieval evaluation set with expected
candidates or relevant evidence.

### Current boundary

Distinct candidate ranking is ready in memory. Formal retrieval evaluation,
complete evidence assembly for shortlisted resumes, index persistence/loading,
LLM evaluation, bias checks, and the Streamlit interface have not started.

## Phase 1, Step 7: First retrieval-evaluation case

### Goal

Replace the vague question "do the results look good?" with a repeatable check
against manually verified ground truth.

### What ground truth means here

Ground truth is the answer we expect before running the retriever. It must not
be copied from FAISS results, because that would judge the system using its own
answer.

The first query is:

```text
machine learning candidate with Flask experience
```

An independent scan of privacy-safe chunks found exactly one anonymous
candidate containing both required terms. Both terms occur in `WORK EXPERIENCE`.
The evaluation JSONL stores the anonymous candidate ID and expected section but
does not store raw resume text or a direct identifier.

### Measurements

- `Recall@3`: what fraction of the relevant candidates appeared in the top 3;
- reciprocal rank: `1 / rank` for the first relevant candidate, or zero if it
  was not retrieved;
- evidence-section recall: what fraction of the expected candidate-and-section
  pairs appeared in the supporting evidence.

With one relevant candidate, `Recall@3` is simple: it is `1.0` if that candidate
appears in the top three and `0.0` otherwise.

### Test-first cycle

The first tests failed because `talent_search.evaluation` did not exist. The
minimal implementation then loaded the labelled JSONL case and calculated the
three metrics. A second failing-test cycle required a positive `k`, at least one
relevant candidate, and at least one expected evidence section so incomplete
labels cannot silently produce misleading results.

### Real result

The manually labelled candidate did not appear in the top three. The first
case therefore produced:

```text
Recall@3 = 0.0
reciprocal rank = 0.0
evidence-section recall = 0.0
```

This does not mean the whole system has zero retrieval quality. One case cannot
estimate overall performance. It means the evaluation successfully found one
specific miss that we can trace and understand.

### Current boundary

The evaluation format, metrics, and first manually labelled case are ready.

### Error analysis of the first miss

A full 1,956-chunk trace found the expected candidate's two relevant `WORK
EXPERIENCE` chunks at ranks 8 and 9. The candidate was already present within
the 50 fetched chunks, but best-chunk aggregation placed the candidate sixth.
Therefore, neither missing evidence nor `fetch_k=50` caused the top-three miss.

The top-ranked chunks showed the key pattern: seven of the first ten contained
neither required term, while the two chunks containing both terms ranked eighth
and ninth. Shortening the query to only the core terms placed the candidate
seventh, and phrasing the terms as required skills placed it twenty-fourth.
Generic query wording was therefore not the root cause.

Dense cosine similarity measures overall semantic closeness; it does not
enforce an exact `machine learning AND Flask` constraint. FAISS and candidate
aggregation are correctly following their current scoring rules, but the
scoring objective does not fully represent a recruiter's must-have criteria.

### Current boundary

The first retrieval miss has been traced to a dense-only retrieval-objective
mismatch. No fix has been implemented. The next checkpoint is to add a small,
varied set of manually labelled cases before choosing between hybrid retrieval,
explicit must-have filtering, or reranking. LLM evaluation, index persistence,
bias checks, and UI work remain unstarted.

## Phase 1, Step 8: Expanded retrieval baseline

### Goal

Test whether the first failure represents a broader pattern instead of tuning
the retriever around one query.

### Evaluation set

Eight privacy-safe cases now cover exact and broad skills, multiple technical
constraints, a framework pair, domain experience, evidence spread across
sections, seniority, and leadership methodology. Labels were derived
independently from sanitized chunks and approved job-related annotations.

One proposed seniority case was rejected during curation because `junior`
appeared in an education context rather than a job designation. It was replaced
with a verified `Senior Development Manager` designation plus SQL, Python, and
Tableau requirements. This demonstrates why ground truth needs human review;
keyword occurrence alone is not always relevance.

### Why Precision@3 was added

Some queries have more than three relevant candidates. Even a perfect top-three
result cannot retrieve all of them, so Recall@3 alone would look artificially
low. Candidate Precision@3 measures how many of the three returned candidates
are relevant. Hit rate records whether at least one relevant candidate was
found, and reciprocal rank records how early the first relevant candidate
appeared.

### Dense-only baseline

```text
cases                                      8
hit rate@3                             0.500
mean candidate Recall@3               0.281
mean candidate Precision@3            0.250
mean reciprocal rank                  0.438
mean evidence-section recall@3        0.219
```

Python and Django achieved perfect candidate and evidence results. Machine
learning with Flask, NLP with Python, the senior-manager query, and team lead
with Agile returned no relevant candidate in the top three. Forecasting with
Python found one relevant candidate at rank one, but did not return the expected
work-experience and skills evidence.

### Current boundary

The dense baseline is now large enough to justify one controlled retrieval
experiment, while still being a small seed dataset rather than a production
benchmark. The next checkpoint is to design and test a hybrid or
constraint-aware approach against the unchanged eight cases. Index persistence,
LLM evaluation, bias checks, Streamlit, and final delivery remain unstarted.

## Phase 1, Step 9: Phrase-aware hybrid retrieval

### Goal

Preserve semantic search while making explicit recruiter requirements affect
candidate order and evidence selection.

### Design

The hybrid retriever removes recruiter boilerplate, then evaluates every
anonymous candidate using:

- their dense FAISS candidate rank;
- a candidate-level BM25 lexical rank;
- coverage of meaningful query terms;
- coverage of multiword phrases such as `machine learning`, `natural language
  processing`, and `team lead`;
- reciprocal-rank fusion to retain both semantic and lexical signals.

Evidence chunks are chosen separately after candidate ranking. Chunks covering
query terms or phrases are preferred, and only the strongest chunk from each
section is kept. This supports queries whose evidence is distributed across
work experience and skills.

### Test-first lessons

The first tests created a misleading dense result and required the explicit
Flask candidate to rank above it. Another test required forecasting and Python
evidence from two different sections.

The first real hybrid run improved mean Recall@3 to `0.844` but still missed the
team-lead case. A new failing test reproduced the reason: individual `team` and
`lead` tokens are not equivalent to the `team lead` role. Preserving multiword
phrases resolved that general problem.

### Verified comparison

```text
Metric                              Dense       Hybrid
Hit rate@3                          0.500        1.000
Mean candidate Recall@3             0.281        0.906
Mean candidate Precision@3          0.250        0.625
Mean reciprocal rank                0.438        0.938
Mean evidence-section recall@3      0.219        0.875
```

The eight cases are deliberately rich in explicit terms and remain too small
for production-quality claims. The comparison establishes that hybrid ranking
fits this recruiter-search objective better than dense-only ranking.

## Phase 1, Step 10: Validated FAISS persistence

### Goal

Avoid embedding all resumes every time the application starts while ensuring a
saved index cannot silently load with incompatible settings or modified
documents.

### Persisted files

- `index.faiss`: 1,956 normalized vectors;
- `documents.jsonl`: privacy-masked chunks and anonymous metadata;
- `manifest.json`: format version, embedding model, vector count, dimensions,
  and SHA-256 document fingerprint.

Saving is atomic and refuses to overwrite an existing directory. Loading checks
all manifest fields, the document fingerprint, unique IDs, and the
`privacy_masked=True` boundary before returning a vector store.

### Verified real run

```text
vectors                         1,956
dimensions                        384
total persisted bytes       4,625,825
embedding/index build          18.486 s
save                            0.024 s
load                            0.043 s
candidate order identical          yes
hybrid scores identical            yes
persisted contact patterns           0
```

The local `.artifacts/faiss_index` directory is Git-ignored. It contains
privacy-masked employment evidence, so it remains private local application
data even though direct identifiers have been removed.

### Current boundary

Retrieval improvement and index persistence are complete. The next checkpoint
is bounded evidence assembly for the top three candidates before an LLM is
connected. LLM generation, generation evaluation, bias checks, Streamlit, and
final portfolio delivery remain unstarted.

## Phase 2, Step 11: Grounded LLM candidate evaluation

### Goal

Use an LLM to explain retrieved evidence without allowing it to control ranking
or invent candidate facts.

The evaluator receives bounded evidence for the top candidates after retrieval
and privacy auditing. Scores are deliberately omitted so the explanation does
not turn similarity into a fake probability. A strict Pydantic schema requires
candidate ID, summary, strengths, gaps, cited evidence sections, and
uncertainty.

Output validation rejects a changed query, duplicate or unknown candidate IDs,
and evidence sections that were not supplied. The prompt prohibits demographic
inference and hiring decisions. Five focused tests verify evidence bounds,
privacy enforcement, score removal, rank preservation, and grounding checks.

The provider is configurable through environment variables. OpenAI is the
implemented provider and `gpt-5.6` is the default model. No live call was made
because the development environment had no API key; a deterministic LangChain
chain verifies the complete application boundary without pretending that a
paid-provider response was tested.

## Phase 2, Step 12: Bias-risk guardrails

### Goal

Reduce observable bias risks while making no universal fairness claim.

Protected-demographic words such as age, gender, race, religion, nationality,
marital status, and disability terms are removed before retrieval. The job
requirements remain, and a query containing only protected preferences is
rejected.

The result audit checks anonymous IDs, the privacy marker, sensitive metadata
keys, and contact-pattern leakage. A counterfactual helper compares candidate
order across job-equivalent neutral rewrites. Passing these tests means only
that the checked risks were not observed; it does not prove that the data,
embedding model, ranking system, or hiring process is unbiased.

## Phase 2, Step 13: Streamlit recruiter chat

### Goal

Expose the complete workflow without requiring a recruiter to write Python.

The UI loads the validated local index once, accepts a natural-language chat
query, shows three anonymous candidate cards, displays term, phrase, and dense
retrieval diagnostics, and keeps supporting evidence inside expanders. It also
shows privacy-audit results and the fairness limitation.

If an OpenAI key is configured, the same cards include structured strengths,
gaps, and uncertainty. Without a key, the app runs in retrieval-only mode and
explains what is unavailable instead of crashing.

Streamlit's application test harness submitted a real query and verified three
candidates plus nine diagnostics. A second test opened the running app in a
real Playwright browser, submitted the same query, inspected the accessibility
snapshot, and captured the checked-in screenshot.

## Phase 2, Step 14: Reproducibility and final delivery

Two commands now make the operational workflow repeatable:

- `scripts/build_index.py` runs load, privacy masking, chunking, embeddings,
  FAISS construction, and safe persistence without overwriting an index;
- `scripts/evaluate_retrieval.py` reloads that index and prints the fixed
  eight-case benchmark as JSON.

The final real benchmark reproduces hit rate `1.000`, mean Recall@3 `0.906`,
mean Precision@3 `0.625`, MRR `0.938`, and evidence-section recall `0.875`.
All 64 automated tests pass and the installed dependency graph is consistent.

The project is complete as a local, portfolio-ready implementation. Its honest
remaining boundary is a live paid-provider smoke test with the owner's API key;
that is optional because retrieval, evidence, UI, and the tested LLM contract
work without committing or requiring a secret.
