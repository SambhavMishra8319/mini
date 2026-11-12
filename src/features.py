"""
Feature engineering: TF-IDF vectorization.
"""

from typing import Tuple

from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import TFIDF_PARAMS


def build_vectorizer() -> TfidfVectorizer:
    """Create a fresh, configured TfidfVectorizer."""
    return TfidfVectorizer(**TFIDF_PARAMS)


def fit_transform_text(vectorizer: TfidfVectorizer, texts) -> Tuple[TfidfVectorizer, object]:
    """Fit the vectorizer on training text and return (vectorizer, matrix)."""
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


def transform_text(vectorizer: TfidfVectorizer, texts):
    """Transform new text with an already-fitted vectorizer."""
    return vectorizer.transform(texts)
