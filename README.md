# Elevvo NLP Engineering Internship Projects

This repository contains four completed, independent NLP projects from the
Elevvo internship. Each project has its own source code, dependencies, setup
instructions, data policy, and runnable entry point.

## Projects

| Task | Project | Main techniques | Status |
|---|---|---|---|
| 05 | [BBC News Topic Modeling](task-05-topic-modeling/) | spaCy, LDA, NMF, coherence tuning, pyLDAvis | Complete |
| 06 | [Extractive Question Answering](task-06-question-answering/) | Hugging Face Transformers, SQuAD, answer-span extraction, EM/F1, Streamlit | Complete |
| 08 | [RAG-Powered Talent Search](task-08-rag-talent-search/) | LangChain, local embeddings, FAISS, hybrid retrieval, optional OpenAI evaluation, Streamlit | Complete |
| 10 | [Autonomous Text-to-SQL Agent](task-10-text-to-sql-agent/) | LangChain, Gemini, SQLGlot, read-only SQLite, self-correction, Streamlit | Complete |

## Independent setup

The projects do not share an environment or a dependency file. Enter the
project you want to run and follow its README.

```powershell
# Task 05
cd task-05-topic-modeling
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Or start separately with Task 08
cd task-08-rag-talent-search
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
```

Tasks 06 and 10 follow the same independent-environment pattern. See each
project README for its exact setup and run commands.

The datasets are intentionally excluded from Git because they are third-party
source data. Each project README explains the expected local data path and how
to run its pipeline.

## Repository structure

```text
Elevvo_Internship/
├── task-05-topic-modeling/
│   ├── data/
│   ├── reports/
│   ├── src/
│   ├── tests/
│   ├── main.py
│   ├── requirements.txt
│   ├── requirements-dev.txt
│   └── README.md
├── task-06-question-answering/
│   ├── reports/
│   ├── scripts/
│   ├── src/
│   ├── tests/
│   ├── app.py
│   ├── requirements.txt
│   ├── requirements-dev.txt
│   └── README.md
├── task-08-rag-talent-search/
│   ├── evaluation/
│   ├── output/
│   ├── scripts/
│   ├── src/
│   ├── tests/
│   ├── streamlit_app.py
│   ├── requirements.txt
│   ├── requirements-dev.txt
│   └── README.md
└── task-10-text-to-sql-agent/
    ├── data/
    ├── scripts/
    ├── src/
    ├── tests/
    ├── app.py
    ├── requirements.txt
    ├── requirements-dev.txt
    └── README.md
```

Generated models, private vector indexes, local environments, and third-party
datasets are not shared between projects or committed to the repository.
