"""
Explainability module.

This project avoids adding SHAP/LIME as hard dependencies (they are heavy
and were not available to install in this offline environment). Instead it
uses the model's OWN native coefficients / log-probabilities, which for the
linear models trained here (Logistic Regression, Linear SVM, Multinomial
Naive Bayes) is an exact, mathematically honest explanation -- not an
approximation, and not fabricated.

If SHAP/LIME are installed in your environment, you can swap them in here;
`is_explainable()` / `explanation_method_name()` tell the UI what is
actually available so it never shows a misleading explanation.
"""

from typing import Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


def is_explainable(model) -> bool:
    return isinstance(model, (LogisticRegression, LinearSVC, MultinomialNB))


def explanation_method_name(model) -> str:
    if isinstance(model, (LogisticRegression, LinearSVC)):
        return "Linear coefficient importance (native model weights)"
    if isinstance(model, MultinomialNB):
        return "Log-probability feature importance (native Naive Bayes weights)"
    return "Not available for this model type"


def _class_weight_vector(model, class_idx: int) -> np.ndarray:
    """Return a 1-D weight vector for a given class index, model-type aware."""
    if isinstance(model, (LogisticRegression, LinearSVC)):
        coef = model.coef_
        if coef.shape[0] == 1:
            # Binary case: sklearn stores a single row; flip sign for class 0.
            return coef[0] if class_idx == 1 else -coef[0]
        return coef[class_idx]
    if isinstance(model, MultinomialNB):
        return model.feature_log_prob_[class_idx]
    raise ValueError("Model type not supported for explanation.")


def global_top_features(model, vectorizer, class_idx: int, top_n: int = 15) -> List[Tuple[str, float]]:
    """Top N most influential terms (by weight) for a given class."""
    weights = _class_weight_vector(model, class_idx)
    feature_names = vectorizer.get_feature_names_out()
    order = np.argsort(weights)[::-1][:top_n]
    return [(feature_names[i], float(weights[i])) for i in order]


def local_explanation(model, vectorizer, clean_text: str, class_idx: int, top_n: int = 10) -> List[Tuple[str, float]]:
    """
    Per-word contribution = tfidf_value(word) * class_weight(word) for the
    words actually present in this specific text.
    """
    weights = _class_weight_vector(model, class_idx)
    vec = vectorizer.transform([clean_text]).tocoo()
    feature_names = vectorizer.get_feature_names_out()

    contributions = [
        (feature_names[col], float(val * weights[col])) for col, val in zip(vec.col, vec.data)
    ]
    contributions.sort(key=lambda x: x[1], reverse=True)
    return contributions[:top_n]
