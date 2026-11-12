"""
Prediction pipeline.

Uses the fine-tuned transformer in models/transformer/ when it exists (and
`torch` + `transformers` are installed). Otherwise it falls back to the
scikit-learn model. The sklearn artifacts are always required because the
"influential terms" shown in the UI come from the TF-IDF vectorizer.
"""

import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional

import joblib

from src import config
from src.preprocessing import basic_text_stats, preprocess_text
from src.sentiment import score_sentiment


class ArtifactsNotFoundError(Exception):
    """Raised when the trained model/vectorizer are missing."""


@dataclass
class PredictionResult:
    raw_text: str
    clean_text: str
    predicted_class: str
    risk_label: str
    confidence: Optional[float]
    supports_probability: bool
    class_probabilities: Dict[str, float] = field(default_factory=dict)
    top_terms: List[str] = field(default_factory=list)
    sentiment_label: str = "Neutral"
    sentiment_score: float = 0.0
    text_stats: Dict = field(default_factory=dict)


def artifacts_available() -> bool:
    return (
        os.path.exists(config.BEST_MODEL_PATH)
        and os.path.exists(config.VECTORIZER_PATH)
        and os.path.exists(config.LABEL_ENCODER_PATH)
    )


@lru_cache(maxsize=1)
def load_artifacts():
    """Load sklearn model, vectorizer, label encoder, and metadata from disk."""
    if not artifacts_available():
        raise ArtifactsNotFoundError(
            "Trained model artifacts not found in models/. "
            "Run `python -m src.train` first."
        )
    model = joblib.load(config.BEST_MODEL_PATH)
    vectorizer = joblib.load(config.VECTORIZER_PATH)
    label_encoder = joblib.load(config.LABEL_ENCODER_PATH)

    metadata = {}
    if os.path.exists(config.METADATA_PATH):
        with open(config.METADATA_PATH) as f:
            metadata = json.load(f)

    return model, vectorizer, label_encoder, metadata


@lru_cache(maxsize=1)
def _load_transformer():
    """Return (tokenizer, model, torch) or None if unavailable."""
    if not os.path.isdir(config.TRANSFORMER_DIR):
        return None
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError:
        return None
    tokenizer = AutoTokenizer.from_pretrained(config.TRANSFORMER_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(config.TRANSFORMER_DIR)
    model.eval()
    return tokenizer, model, torch


def transformer_available() -> bool:
    return _load_transformer() is not None


def _transformer_probabilities(text: str) -> Optional[Dict[str, float]]:
    bundle = _load_transformer()
    if bundle is None:
        return None
    tokenizer, model, torch = bundle
    enc = tokenizer(
        text, truncation=True, max_length=config.TRANSFORMER_MAX_LEN, return_tensors="pt"
    )
    with torch.no_grad():
        logits = model(**enc).logits[0]
    probs = torch.softmax(logits, dim=-1).tolist()
    id2label = model.config.id2label
    return {
        str(id2label.get(i, id2label.get(str(i), i))): float(p)
        for i, p in enumerate(probs)
    }


def _risk_label_for(predicted_class: str) -> str:
    return config.RISK_LABEL_MAP.get(
        predicted_class.lower(), f"Signal Detected: {predicted_class.title()}"
    )


def _top_tfidf_terms(vectorizer, clean_text: str, top_n: int = 8) -> List[str]:
    """Return the highest-weighted TF-IDF terms present in this text."""
    vec = vectorizer.transform([clean_text])
    if vec.nnz == 0:
        return []
    feature_names = vectorizer.get_feature_names_out()
    row = vec.tocoo()
    pairs = sorted(zip(row.col, row.data), key=lambda p: p[1], reverse=True)
    return [feature_names[i] for i, _ in pairs[:top_n]]


def predict(text: str) -> PredictionResult:
    """
    Full prediction pipeline for a single piece of raw text.
    Raises ArtifactsNotFoundError if the model hasn't been trained yet.
    Raises ValueError for empty/invalid input.
    """
    if text is None or not str(text).strip():
        raise ValueError("Input text is empty. Please enter some text to analyze.")

    model, vectorizer, label_encoder, _metadata = load_artifacts()

    clean_text = preprocess_text(text)
    if not clean_text:
        raise ValueError(
            "After cleaning, no meaningful words remained in the input. "
            "Please enter a longer or more descriptive piece of text."
        )

    class_probabilities: Dict[str, float] = {}
    confidence = None

    # The transformer is fed the RAW text (it has its own tokenizer);
    # the sklearn model is fed the cleaned text.
    transformer_probs = _transformer_probabilities(text)
    if transformer_probs is not None:
        class_probabilities = transformer_probs
        predicted_class = max(class_probabilities, key=class_probabilities.get)
        confidence = float(class_probabilities[predicted_class])
        supports_probability = True
    else:
        X = vectorizer.transform([clean_text])
        pred_idx = model.predict(X)[0]
        predicted_class = label_encoder.inverse_transform([pred_idx])[0]

        supports_probability = hasattr(model, "predict_proba")
        if supports_probability:
            proba = model.predict_proba(X)[0]
            class_probabilities = {
                label_encoder.inverse_transform([i])[0]: float(p)
                for i, p in enumerate(proba)
            }
            confidence = float(max(proba))

    top_terms = _top_tfidf_terms(vectorizer, clean_text)
    sentiment_label, sentiment_score = score_sentiment(clean_text)
    stats = basic_text_stats(text)

    return PredictionResult(
        raw_text=text,
        clean_text=clean_text,
        predicted_class=predicted_class,
        risk_label=_risk_label_for(predicted_class),
        confidence=confidence,
        supports_probability=supports_probability,
        class_probabilities=class_probabilities,
        top_terms=top_terms,
        sentiment_label=sentiment_label,
        sentiment_score=sentiment_score,
        text_stats=stats,
    )
