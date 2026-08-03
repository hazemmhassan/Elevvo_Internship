# Task 08: RAG-Powered Talent Search Engine

This project is being built as a guided, LangChain-first implementation. Each
pipeline step is explained, implemented, tested, and reviewed before work begins
on the next step.

## Current checkpoint

Phase 1 is complete through embeddings:

1. Load and validate the JSONL dataset as raw LangChain `Document` objects.
2. Replace direct identifiers and remove unsafe metadata.
3. Divide each resume by meaning, then recursively split oversized sections.
4. Convert every privacy-safe chunk into a normalized local embedding.

The verified dataset run produced 1,956 chunks from 220 resumes. The local
`sentence-transformers/all-MiniLM-L6-v2` model produced a `1956 x 384` embedding
matrix in memory.

FAISS indexing has deliberately not started. No vector matrix or resume content
is written to disk by the current code.

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
  -> STOP: FAISS is the next checkpoint
```

The section splitter starts with `chunk_size=1000` characters and
`chunk_overlap=150`. These are sensible baseline settings, not final optimized
values. Retrieval evaluation later will tell us whether to adjust them.

## Core code flow

```python
from pathlib import Path

from talent_search.embeddings import create_local_embeddings, embed_documents
from talent_search.loaders import ResumeJsonlLoader
from talent_search.privacy import sanitize_resume_document
from talent_search.splitting import ResumeSectionSplitter

DATASET_PATH = Path("path/to/Entity Recognition in Resumes.json")

raw_documents = ResumeJsonlLoader(DATASET_PATH).load()
safe_documents = [sanitize_resume_document(doc) for doc in raw_documents]
chunks = ResumeSectionSplitter().split_documents(safe_documents)

embedding_model = create_local_embeddings()
vectors = embed_documents(chunks, embedding_model)
```

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
