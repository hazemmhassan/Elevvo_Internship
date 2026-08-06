# Task 05: BBC News Unsupervised Topic Modeling

An independent NLP pipeline that discovers themes in BBC news articles and
compares Latent Dirichlet Allocation (LDA) with Non-negative Matrix
Factorization (NMF). It includes linguistic preprocessing, phrase detection,
coherence-based topic-count selection, keyword extraction, and interactive
pyLDAvis output.

## Objective

Turn a large collection of news articles into structured topics that can help
analysts explore recurring themes and compare two common unsupervised modeling
approaches.

## Pipeline

```text
BBC news CSV
  -> validate the description column
  -> clean HTML, URLs, punctuation, and numbers
  -> spaCy tokenization, lemmatization, and stop-word removal
  -> Gensim phrase detection
  -> dictionary and bag-of-words corpus
  -> tune LDA topic count with c_v coherence
  -> train final LDA and NMF models
  -> extract topic keywords
  -> export coherence curve and pyLDAvis dashboard
```

## Project files

```text
task-05-topic-modeling/
├── data/
│   └── .gitkeep
├── reports/
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   └── models.py
├── tests/
│   └── test_pipeline_smoke.py
├── .gitignore
├── main.py
├── requirements-dev.txt
├── requirements.txt
└── README.md
```

## Setup

Run these commands from `task-05-topic-modeling` so this project remains
independent from the other internship tasks:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

For development and tests, install `requirements-dev.txt` instead.

## Dataset

The BBC News dataset is third-party data and is intentionally not committed.
Place it at:

```text
data/BBC news dataset.csv
```

The CSV must contain a non-empty `description` column. The application raises a
clear error if the file or required column is missing.

## Run

```powershell
python main.py
```

The pipeline prints the selected topic count and the top LDA and NMF keywords.
It generates these local artifacts:

```text
reports/coherence_score_curve.png
reports/lda_visualization.html
```

Generated reports are excluded from Git because they are reproducible outputs.

## Test

The regression suite uses synthetic topic groups, so it validates preprocessing,
LDA, NMF, the coherence plot, and the pyLDAvis export without requiring the
third-party BBC dataset or exposing its contents.

```powershell
pip install -r requirements-dev.txt
pytest -q
```

## Main components

- `src/preprocessing.py` loads the CSV, cleans text, lemmatizes it with spaCy,
  removes stop words, and detects common phrases.
- `src/models.py` builds the corpus, tunes topic count, trains LDA and NMF,
  extracts keywords, and exports visualizations.
- `main.py` connects preprocessing and modeling into one executable workflow.

## Reproducibility and limitations

- Random seeds are fixed for the final LDA and NMF models.
- The best topic count is chosen from 3 through 8 using `c_v` coherence.
- Topic labels still require human interpretation; unsupervised topics are not
  automatically equivalent to business categories.
- Results depend on the exact dataset version and installed library versions.
- The first full run can take several minutes because multiple LDA models are
  trained during coherence tuning.
