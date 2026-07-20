"""Training, evaluation, and export utilities for unsupervised topic modeling."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gensim import corpora
from gensim.models import CoherenceModel, LdaModel
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

try:
    import pyLDAvis.gensim_models as gensim_vis
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError("pyLDAvis is required. Install it with: pip install pyLDAvis") from exc

class ModelingError(Exception):
    """Raised when topic modeling fails due to invalid inputs or runtime issues."""


def build_corpus_and_dictionary(tokenized_docs: List[List[str]]) -> Tuple[corpora.Dictionary, List[List[Tuple[int, int]]]]:
    """Build a Gensim dictionary and bag-of-words corpus from tokenized documents."""
    if not tokenized_docs:
        raise ModelingError("At least one document is required to build the topic model.")

    dictionary = corpora.Dictionary(tokenized_docs)
    dictionary.filter_extremes(no_below=5, no_above=0.85, keep_n=100000)
    corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]
    return dictionary, corpus


def compute_coherence_values(
    dictionary: corpora.Dictionary,
    corpus: List[List[Tuple[int, int]]],
    texts: List[List[str]],
    start: int = 3,
    limit: int = 8,
    step: int = 1,
) -> Tuple[List[int], List[float]]:
    """Train LDA models for a range of topic counts and compute c_v coherence scores."""
    coherence_values: List[float] = []
    model_topics: List[int] = []

    for num_topics in tqdm(range(start, limit + 1, step), desc="Tuning Topics"):
        print(f"Training LDA model with {num_topics} topics...")
        lda_model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            random_state=42,
            chunksize=2000,
            passes=10,
            iterations=400,
            alpha="auto",
            eta="auto",
        )
        coherence_model = CoherenceModel(
            model=lda_model,
            texts=texts,
            dictionary=dictionary,
            coherence="c_v",
        )
        coherence_values.append(float(coherence_model.get_coherence()))
        model_topics.append(num_topics)

    return model_topics, coherence_values


def plot_coherence_curve(
    topics: List[int],
    coherence_values: List[float],
    output_path: str | Path = "reports/coherence_score_curve.png",
) -> Path:
    """Plot and save the coherence score curve."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(topics, coherence_values, marker="o")
    plt.title("Topic Coherence Score by Number of Topics")
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence Score")
    plt.xticks(topics)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved coherence plot to {output_path}")
    return output_path


def train_lda_model(
    dictionary: corpora.Dictionary,
    corpus: List[List[Tuple[int, int]]],
    texts: List[List[str]],
    num_topics: int,
    random_state: int = 42,
) -> LdaModel:
    """Train a final LDA model with the selected number of topics."""
    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=random_state,
        chunksize=2000,
        passes=10,
        iterations=400,
        alpha="auto",
        eta="auto",
    )
    return lda_model


def get_topic_keywords(lda_model: LdaModel, num_words: int = 10) -> List[List[str]]:
    """Return the top terms for each topic from an LDA model."""
    print("Extracting LDA topic keywords...")
    return [
        [word for word, _ in lda_model.show_topic(topic_id, topn=num_words)]
        for topic_id in tqdm(range(lda_model.num_topics), desc="Extracting LDA keywords")
    ]


def train_nmf_model(
    texts: List[List[str]],
    n_topics: int,
    random_state: int = 42,
) -> Tuple[NMF, TfidfVectorizer]:
    """Train an NMF model on the preprocessed tokens using TF-IDF features."""
    documents = [" ".join(doc) for doc in texts]
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(documents)
    nmf_model = NMF(n_components=n_topics, random_state=random_state, init="nndsvda")
    nmf_model.fit(X)
    return nmf_model, vectorizer


def get_nmf_keywords(nmf_model: NMF, vectorizer: TfidfVectorizer, num_words: int = 10) -> List[List[str]]:
    """Return the top terms for each topic from an NMF model."""
    print("Extracting NMF topic keywords...")
    feature_names = vectorizer.get_feature_names_out()
    return [
        [feature_names[idx] for idx in topic.argsort()[:-num_words - 1:-1]]
        for topic in tqdm(nmf_model.components_, desc="Extracting NMF keywords")
    ]


def export_pyldavis(
    lda_model: LdaModel,
    dictionary: corpora.Dictionary,
    corpus: List[List[Tuple[int, int]]],
    output_path: str | Path = "reports/lda_visualization.html",
) -> Path:
    """Generate and save an interactive pyLDAvis dashboard."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    vis = gensim_vis.prepare(lda_model, corpus, dictionary)
    gensim_vis.save_html(vis, str(output_path))
    print(f"Saved pyLDAvis dashboard to {output_path}")
    return output_path


def select_best_topic_count(topics: List[int], coherence_values: List[float]) -> int:
    """Select the topic count with the highest coherence score."""
    if not topics or not coherence_values:
        raise ModelingError("Coherence scores are empty.")
    best_index = int(np.argmax(coherence_values))
    return topics[best_index]


def run_topic_modeling(
    df: pd.DataFrame,
    text_column: str = "cleaned_tokens",
    start: int = 3,
    limit: int = 8,
    step: int = 1,
    num_topics: Optional[int] = None,
) -> Dict[str, Any]:
    """Run the complete LDA and NMF topic modeling workflow."""
    if text_column not in df.columns:
        raise ModelingError(f"Column '{text_column}' not found in dataframe.")

    texts = [list(doc) for doc in df[text_column].tolist()]
    dictionary, corpus = build_corpus_and_dictionary(texts)

    topics, coherence_values = compute_coherence_values(dictionary, corpus, texts, start, limit, step)
    plot_coherence_curve(topics, coherence_values)

    best_topics = num_topics or select_best_topic_count(topics, coherence_values)
    lda_model = train_lda_model(dictionary, corpus, texts, best_topics)
    nmf_model, vectorizer = train_nmf_model(texts, n_topics=best_topics)

    lda_keywords = get_topic_keywords(lda_model)
    nmf_keywords = get_nmf_keywords(nmf_model, vectorizer)
    export_pyldavis(lda_model, dictionary, corpus)

    return {
        "dictionary": dictionary,
        "corpus": corpus,
        "texts": texts,
        "topics": topics,
        "coherence_values": coherence_values,
        "best_topics": best_topics,
        "lda_model": lda_model,
        "nmf_model": nmf_model,
        "vectorizer": vectorizer,
        "lda_keywords": lda_keywords,
        "nmf_keywords": nmf_keywords,
    }
