"""Production-grade text preprocessing for BBC news topic modeling.

This module loads the BBC news dataset, cleans the description column,
creates token sequences with spaCy and Gensim phrase modeling, and stores
final cleaned tokens in a new column for downstream topic modeling.
"""

from __future__ import annotations

import logging
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, List, Optional

import pandas as pd

try:
    import spacy
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError("spaCy is required. Install it with: pip install spacy") from exc

try:
    from gensim.models.phrases import Phrases, Phraser
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError("gensim is required. Install it with: pip install gensim") from exc


logger = logging.getLogger(__name__)


def setup_logging(log_level: int = logging.INFO) -> None:
    """Configure project logging for preprocessing tasks."""
    if not logger.handlers:
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        )


class PreprocessingError(Exception):
    """Raised when preprocessing fails due to invalid input or runtime issues."""


class TextPreprocessor:
    """Advanced text preprocessing pipeline for BBC news descriptions."""

    def __init__(
        self,
        data_path: str | Path,
        text_column: str = "description",
        output_column: str = "cleaned_tokens",
        model_name: str = "en_core_web_sm",
        min_token_length: int = 2,
        remove_numbers: bool = True,
    ) -> None:
        self.data_path = Path(data_path)
        self.text_column = text_column
        self.output_column = output_column
        self.model_name = model_name
        self.min_token_length = min_token_length
        self.remove_numbers = remove_numbers
        self.nlp = None
        self._load_spacy_model()

    def _load_spacy_model(self) -> None:
        """Load and validate the spaCy model."""
        try:
            self.nlp = spacy.load(self.model_name)
            logger.info("Loaded spaCy model: %s", self.model_name)
        except OSError as exc:
            logger.warning(
                "spaCy model '%s' not found; attempting automatic download.",
                self.model_name,
            )
            try:
                subprocess.run(
                    [sys.executable, "-m", "spacy", "download", self.model_name],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                self.nlp = spacy.load(self.model_name)
                logger.info("Loaded spaCy model: %s", self.model_name)
            except (subprocess.CalledProcessError, OSError) as download_exc:
                raise PreprocessingError(
                    f"spaCy model '{self.model_name}' not found and could not be downloaded automatically. "
                    f"Install it manually with: python -m spacy download {self.model_name}"
                ) from download_exc

    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        """Validate that the input dataframe contains the expected text column."""
        if df is None or df.empty:
            raise PreprocessingError("Input dataframe is empty.")
        if self.text_column not in df.columns:
            raise PreprocessingError(
                f"Column '{self.text_column}' not found. Available columns: {list(df.columns)}"
            )

    def _clean_text(self, text: Any) -> str:
        """Normalize a single text field prior to tokenization."""
        if pd.isna(text):
            return ""

        text = str(text).lower()
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"http\S+|www\.\S+", " ", text)
        text = re.sub(r"[^a-z\s]", " ", text)
        if self.remove_numbers:
            text = re.sub(r"\d+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize cleaned text using spaCy and apply basic filtering."""
        if not text:
            return []

        doc = self.nlp(text)
        tokens: List[str] = []
        for token in doc:
            if token.is_stop or token.is_punct or token.is_space:
                continue
            lemma = token.lemma_.lower().strip()
            if not lemma or len(lemma) < self.min_token_length:
                continue
            if lemma.isdigit():
                continue
            tokens.append(lemma)
        return tokens

    def _build_phrases(self, tokenized_docs: Iterable[Iterable[str]]) -> Any:
        """Create a phrase model from token sequences."""
        tokenized_docs = [list(doc) for doc in tokenized_docs]
        if not tokenized_docs:
            return None

        byte_docs = [[token.encode("utf-8") for token in doc] for doc in tokenized_docs]
        return Phrases(byte_docs, min_count=10, threshold=20, delimiter=b"_")

    def _apply_phrases(self, phrase_model: Any, tokens: Iterable[str]) -> List[str]:
        """Apply the trained phrase model to a single token list."""
        if not phrase_model:
            return list(tokens)

        try:
            encoded_tokens = [token.encode("utf-8") for token in tokens]
            transformed_tokens = phrase_model[encoded_tokens]
            if hasattr(transformed_tokens, "__iter__") and not isinstance(transformed_tokens, (list, tuple)):
                transformed_tokens = list(transformed_tokens)
            return [token.decode("utf-8") if isinstance(token, bytes) else str(token) for token in transformed_tokens]
        except Exception:
            return list(tokens)

    def preprocess(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Run the full preprocessing workflow and return the dataframe."""
        setup_logging()

        if df is None:
            logger.info("Loading dataset from %s", self.data_path)
            try:
                df = pd.read_csv(self.data_path)
            except FileNotFoundError as exc:
                raise PreprocessingError(f"Dataset file not found: {self.data_path}") from exc
            except Exception as exc:  # pragma: no cover - runtime safety
                raise PreprocessingError(f"Failed to read dataset: {exc}") from exc

        self._validate_dataframe(df)

        logger.info("Starting preprocessing for %s rows", len(df))
        df = df.copy()

        # Clean text and create tokenized documents
        cleaned_texts = []
        tokenized_docs = []
        for raw_text in df[self.text_column].tolist():
            cleaned_text = self._clean_text(raw_text)
            cleaned_texts.append(cleaned_text)
            tokens = self._tokenize(cleaned_text)
            tokenized_docs.append(tokens)

        logger.info("Building phrase models from tokenized descriptions")
        phrase_model = self._build_phrases(tokenized_docs)

        cleaned_tokens = []
        for tokens in tokenized_docs:
            phrase_tokens = self._apply_phrases(phrase_model, tokens)
            cleaned_tokens.append(phrase_tokens)

        df["cleaned_text"] = cleaned_texts
        df[self.output_column] = cleaned_tokens
        logger.info("Preprocessing complete. Output column: %s", self.output_column)
        return df

    def inspect_samples(self, df: pd.DataFrame, n: int = 5) -> None:
        """Print a quick inspection of the cleaned tokens for verification."""
        if self.output_column not in df.columns:
            raise PreprocessingError(
                f"Column '{self.output_column}' not found. Run preprocess() first."
            )
        for idx, row in df[[self.text_column, self.output_column]].head(n).iterrows():
            print(f"Row {idx}: {row[self.text_column]}")
            print(f"Tokens: {row[self.output_column]}")
            print("-" * 80)


def preprocess_dataset(
    data_path: str | Path = "BBC news dataset.csv",
    text_column: str = "description",
    output_column: str = "cleaned_tokens",
    model_name: str = "en_core_web_sm",
) -> pd.DataFrame:
    """Convenience wrapper to run the full preprocessing pipeline."""
    preprocessor = TextPreprocessor(
        data_path=data_path,
        text_column=text_column,
        output_column=output_column,
        model_name=model_name,
    )
    return preprocessor.preprocess()


if __name__ == "__main__":
    try:
        df = preprocess_dataset()
        print(df[["description", "cleaned_tokens"]].head(3).to_string(index=False))
    except Exception as exc:  # pragma: no cover - CLI safety
        logger.exception("Preprocessing failed: %s", exc)
