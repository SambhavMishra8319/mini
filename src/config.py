"""
Central configuration for the Mental Health Risk Prediction project.
Change dataset column names, paths, and model settings here only.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

DATASET_PATH = os.path.join(DATA_DIR, "dataset.csv")
SAMPLE_DATASET_PATH = os.path.join(DATA_DIR, "dataset_sample.csv")

BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.joblib")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.joblib")
METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")
METRICS_PATH = os.path.join(MODELS_DIR, "evaluation_metrics.json")

# Fine-tuned transformer (trained in Colab, copied into models/)
TRANSFORMER_DIR = os.path.join(MODELS_DIR, "transformer")
TRANSFORMER_METRICS_PATH = os.path.join(MODELS_DIR, "transformer_metrics.json")
TRANSFORMER_MAX_LEN = 256

# ---------------------------------------------------------------------------
# Dataset column configuration (Kaggle "Sentiment Analysis for Mental Health")
# ---------------------------------------------------------------------------
TEXT_COLUMN = "statement"
TARGET_COLUMN = "status"

# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1  # taken out of the training split

# ---------------------------------------------------------------------------
# TF-IDF configuration
# token_pattern keeps single-letter tokens such as "i" (the sklearn default
# silently drops them) and keeps apostrophes inside words.
# ---------------------------------------------------------------------------
TFIDF_PARAMS = {
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_df": 0.9,
    "max_features": 100000,
    "sublinear_tf": True,
    "token_pattern": r"(?u)\b\w[\w']*\b",
}

# ---------------------------------------------------------------------------
# Hyperparameter grids (searched on the validation split in train.py)
# ---------------------------------------------------------------------------
LR_C_GRID = [1, 3, 10]
SVM_C_GRID = [0.1, 0.3, 1]
NB_ALPHA_GRID = [0.1, 0.3, 1.0]

# ---------------------------------------------------------------------------
# Model selection metric (computed on the VALIDATION split)
# ---------------------------------------------------------------------------
SELECTION_METRIC = "f1_macro"

# ---------------------------------------------------------------------------
# Risk label mapping (dataset-category -> human-friendly, non-clinical label)
# ---------------------------------------------------------------------------
RISK_LABEL_MAP = {
    "normal": "No Significant Risk Signal",
    "stress": "Moderate Risk Signal",
    "anxiety": "Moderate Risk Signal",
    "depression": "High Risk Signal",
    "suicidal": "High Risk Signal",
    "bipolar": "Moderate Risk Signal",
    "personality disorder": "Moderate Risk Signal",
}

DISCLAIMER = (
    "Our project is an educational and research prototype for analyzing "
    "social-media text using NLP and Machine Learning. It identifies potential "
    "mental-health risk signals and does not provide medical diagnosis or treatment."
)
