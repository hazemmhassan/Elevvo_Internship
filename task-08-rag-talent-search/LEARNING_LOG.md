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

- Candidate-level grouping and ranking
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

Chunk search is ready and remains entirely in memory. Candidate aggregation,
candidate ranking, index persistence/loading, LLM evaluation, bias checks, and
the Streamlit interface have not started.
