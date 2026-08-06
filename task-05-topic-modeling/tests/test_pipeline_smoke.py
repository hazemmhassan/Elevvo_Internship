from __future__ import annotations

import warnings

from src.models import (
    build_corpus_and_dictionary,
    export_pyldavis,
    get_nmf_keywords,
    get_topic_keywords,
    plot_coherence_curve,
    train_lda_model,
    train_nmf_model,
)
from src.preprocessing import TextPreprocessor


def test_text_cleaning_and_phrase_output_are_model_ready() -> None:
    preprocessor = TextPreprocessor.__new__(TextPreprocessor)
    preprocessor.min_token_length = 2
    preprocessor.remove_numbers = True

    assert (
        preprocessor._clean_text(
            "<p>Markets 2026</p> https://example.invalid Growth!"
        )
        == "markets growth"
    )

    phrase_model = preprocessor._build_phrases(
        [["new", "york", "market"]] * 12
    )
    transformed = preprocessor._apply_phrases(
        phrase_model,
        ["new", "york", "market"],
    )
    assert transformed
    assert all(isinstance(token, str) for token in transformed)


def test_lda_nmf_and_visual_exports_work_together(tmp_path) -> None:
    groups = [
        ["football", "team", "match", "league", "coach", "player", "goal"],
        ["market", "bank", "economy", "trade", "stock", "finance", "growth"],
        [
            "software",
            "computer",
            "internet",
            "technology",
            "data",
            "system",
            "digital",
        ],
        [
            "election",
            "government",
            "minister",
            "policy",
            "party",
            "vote",
            "parliament",
        ],
    ]
    texts = [
        words + [f"group{group}", f"doc{document % 5}"]
        for group, words in enumerate(groups)
        for document in range(25)
    ]

    dictionary, corpus = build_corpus_and_dictionary(texts)
    lda_model = train_lda_model(dictionary, corpus, texts, num_topics=4)
    nmf_model, vectorizer = train_nmf_model(texts, n_topics=4)

    assert len(get_topic_keywords(lda_model, num_words=3)) == 4
    assert len(get_nmf_keywords(nmf_model, vectorizer, num_words=3)) == 4

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        coherence_path = plot_coherence_curve(
            [3, 4],
            [0.40, 0.55],
            tmp_path / "coherence.png",
        )
        dashboard_path = export_pyldavis(
            lda_model,
            dictionary,
            corpus,
            tmp_path / "topics.html",
        )

    assert coherence_path.is_file()
    assert dashboard_path.is_file()
    assert dashboard_path.stat().st_size > 1_000
