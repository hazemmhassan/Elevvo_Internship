# Task 08: RAG-Powered Talent Search Engine

This project is being built as a guided, LangChain-first implementation. Each
pipeline step is explained, implemented, tested, and reviewed before work begins
on the next step.

## Current checkpoint

Phase 1 is complete through distinct candidate ranking:

1. Load and validate the JSONL dataset as raw LangChain `Document` objects.
2. Replace direct identifiers and remove unsafe metadata.
3. Divide each resume by meaning, then recursively split oversized sections.
4. Convert every privacy-safe chunk into a normalized local embedding.
5. Index the embeddings in FAISS and return the closest LangChain documents.
6. Group matching chunks by candidate and rank distinct candidates.

The verified dataset run produced 1,956 chunks from 220 resumes. The local
`sentence-transformers/all-MiniLM-L6-v2` model produced a `1956 x 384` embedding
matrix in memory.

The verified FAISS run stores 1,956 vectors with 384 dimensions in an exact
`IndexFlatIP` index. The index remains in memory; no vector matrix, FAISS index,
or resume content is written to disk by the current code.

## Pipeline so far

```text
JSONL file
  -> ResumeJsonlLoader
  -> raw Documents (private)
  -> sanitize_resume_document
  -> anonymous resume Documents
  -> ResumeSectionSplitter
  -> small section Documents
  -> HuggingFaceEmbeddings
  -> 384-number meaning vectors (in memory)
  -> exact FAISS index
  -> closest resume chunks + cosine similarity scores
  -> group chunks by anonymous candidate ID
  -> top distinct candidates + supporting evidence
  -> STOP: retrieval evaluation is the next checkpoint
```

The section splitter starts with `chunk_size=1000` characters and
`chunk_overlap=150`. These are sensible baseline settings, not final optimized
values. Retrieval evaluation later will tell us whether to adjust them.

## Core code flow

```python
from pathlib import Path

from talent_search.embeddings import create_local_embeddings
from talent_search.loaders import ResumeJsonlLoader
from talent_search.privacy import sanitize_resume_document
from talent_search.retrieval import retrieve_candidates
from talent_search.splitting import ResumeSectionSplitter
from talent_search.vector_store import build_faiss_index

DATASET_PATH = Path("path/to/Entity Recognition in Resumes.json")

raw_documents = ResumeJsonlLoader(DATASET_PATH).load()
safe_documents = [sanitize_resume_document(doc) for doc in raw_documents]
chunks = ResumeSectionSplitter().split_documents(safe_documents)

embedding_model = create_local_embeddings()
vector_store = build_faiss_index(chunks, embedding_model)
candidate_matches = retrieve_candidates(
    vector_store,
    "junior data analyst with SQL and Tableau",
    candidate_k=3,
    fetch_k=50,
    evidence_per_candidate=3,
)
```

`FaissVectorStore` implements LangChain Core's `VectorStore` interface directly
on top of `faiss-cpu`. This avoids depending on the archived
`langchain-community` package while keeping LangChain `Document` and
`Embeddings` compatibility.

Candidate ranking uses the strongest matching chunk as the candidate score.
Additional chunks are evidence only, so candidates with more chunks cannot gain
extra ranking points. Evidence keeps at most one chunk from each section.

## Install and test

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
$env:PYTHONPATH = "$PWD\src"
pytest
```

On Windows, a deeply nested checkout can exceed the path-length limit while
installing PyTorch. In that case, keep the repository where it is and create
only the virtual environment at a shorter path, for example:

```powershell
python -m venv "$env:USERPROFILE\.venvs\elevvo08"
& "$env:USERPROFILE\.venvs\elevvo08\Scripts\Activate.ps1"
pip install -r requirements-dev.txt
```

The first embedding-model run downloads the model weights. Later runs reuse the
Hugging Face cache.

The Kaggle dataset is intentionally not stored in Git.
