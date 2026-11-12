# """
# End-to-end training pipeline.

# Run:
#     python -m src.train

# What it does:
# 1. Loads data/dataset.csv (falls back to data/dataset_sample.csv).
# 2. Cleans + preprocesses text.
# 3. Splits into train/validation/test.
# 4. Fits TF-IDF on the training split only.
# 5. Trains Logistic Regression, Multinomial Naive Bayes, and Linear SVM.
# 6. Evaluates every model on the test split with real metrics.
# 7. Selects the best model by macro F1.
# 8. Saves the model, vectorizer, label encoder, metadata, and metrics to disk.
# """

# import json
# import os
# import sys
# from datetime import datetime

# import joblib
# import numpy as np
# import pandas as pd
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.preprocessing import LabelEncoder
# from sklearn.svm import LinearSVC

# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from src import config
# from src.evaluate import evaluate_predictions
# from src.features import build_vectorizer
# from src.preprocessing import preprocess_series


# def load_dataset() -> tuple[pd.DataFrame, bool]:
#     """Load the real dataset if present, otherwise fall back to the demo one."""
#     if os.path.exists(config.DATASET_PATH):
#         df = pd.read_csv(config.DATASET_PATH)
#         is_demo = False
#     elif os.path.exists(config.SAMPLE_DATASET_PATH):
#         df = pd.read_csv(config.SAMPLE_DATASET_PATH)
#         is_demo = True
#         print(
#             "[WARNING] data/dataset.csv not found. Falling back to the small "
#             "synthetic demo dataset (data/dataset_sample.csv). See data/README.md "
#             "to install the real recommended dataset."
#         )
#     else:
#         raise FileNotFoundError(
#             "No dataset found. Place a CSV at data/dataset.csv (see data/README.md), "
#             "or generate the demo dataset with: python data/generate_sample_dataset.py"
#         )

#     if config.TEXT_COLUMN not in df.columns or config.TARGET_COLUMN not in df.columns:
#         raise ValueError(
#             f"Dataset must contain columns '{config.TEXT_COLUMN}' and "
#             f"'{config.TARGET_COLUMN}'. Found columns: {list(df.columns)}. "
#             f"Update TEXT_COLUMN / TARGET_COLUMN in src/config.py to match your CSV."
#         )

#     return df, is_demo


# def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
#     df = df.copy()
#     df = df.dropna(subset=[config.TEXT_COLUMN, config.TARGET_COLUMN])
#     df[config.TEXT_COLUMN] = df[config.TEXT_COLUMN].astype(str).str.strip()
#     df = df[df[config.TEXT_COLUMN].str.len() > 0]
#     df = df.drop_duplicates(subset=[config.TEXT_COLUMN])
#     df[config.TARGET_COLUMN] = df[config.TARGET_COLUMN].astype(str).str.strip().str.lower()
#     return df.reset_index(drop=True)


# def get_models():
#     return {
#         "Logistic Regression": LogisticRegression(
#             max_iter=1000, class_weight="balanced", random_state=config.RANDOM_STATE
#         ),
#         "Multinomial Naive Bayes": MultinomialNB(),
#         "Linear SVM": LinearSVC(
#             class_weight="balanced", random_state=config.RANDOM_STATE
#         ),
#     }


# def main():
#     os.makedirs(config.MODELS_DIR, exist_ok=True)

#     print("Phase 1: Loading dataset...")
#     df, is_demo = load_dataset()
#     print(f"  Loaded {len(df)} raw rows. Demo dataset: {is_demo}")

#     print("Phase 1: Cleaning dataset...")
#     df = clean_dataset(df)
#     print(f"  {len(df)} rows after cleaning (dropna + dedupe).")
#     class_counts = df[config.TARGET_COLUMN].value_counts().to_dict()
#     print(f"  Class distribution: {class_counts}")

#     print("Phase 2: NLP preprocessing...")
#     df["clean_text"] = preprocess_series(df[config.TEXT_COLUMN])
#     df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)
#     print(f"  {len(df)} rows remain after preprocessing (empty-after-clean rows dropped).")

#     print("Encoding labels...")
#     label_encoder = LabelEncoder()
#     y = label_encoder.fit_transform(df[config.TARGET_COLUMN])
#     classes = label_encoder.classes_
#     print(f"  Classes: {list(classes)}")

#     print("Splitting train/val/test...")
#     X_temp, X_test, y_temp, y_test = train_test_split(
#         df["clean_text"], y, test_size=config.TEST_SIZE,
#         random_state=config.RANDOM_STATE, stratify=y,
#     )
#     val_ratio = config.VALIDATION_SIZE / (1 - config.TEST_SIZE)
#     X_train, X_val, y_train, y_val = train_test_split(
#         X_temp, y_temp, test_size=val_ratio,
#         random_state=config.RANDOM_STATE, stratify=y_temp,
#     )
#     print(f"  Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

#     print("Phase 3: Fitting TF-IDF vectorizer on training data...")
#     vectorizer = build_vectorizer()
#     X_train_vec = vectorizer.fit_transform(X_train)
#     X_val_vec = vectorizer.transform(X_val)
#     X_test_vec = vectorizer.transform(X_test)
#     print(f"  Vocabulary size: {len(vectorizer.vocabulary_)}")

#     print("Phase 4: Training models...")
#     models = get_models()
#     results = {}
#     fitted_models = {}

#     for name, model in models.items():
#         print(f"  Training {name}...")
#         model.fit(X_train_vec, y_train)
#         fitted_models[name] = model

#         # Predict on test set (final honest metrics)
#         y_pred = model.predict(X_test_vec)

#         y_proba = None
#         if hasattr(model, "predict_proba"):
#             y_proba = model.predict_proba(X_test_vec)

#         metrics = evaluate_predictions(y_test, y_pred, y_proba, classes=list(range(len(classes))))
#         metrics["classes"] = list(classes)  # replace numeric with real names
#         # re-key per_class with real class names
#         metrics["per_class"] = {
#             classes[int(k)]: v for k, v in metrics["per_class"].items()
#         }
#         metrics["supports_probability"] = hasattr(model, "predict_proba")
#         results[name] = metrics
#         print(
#             f"    Accuracy={metrics['accuracy']:.3f}  "
#             f"F1-macro={metrics['f1_macro']:.3f}  "
#             f"F1-weighted={metrics['f1_weighted']:.3f}"
#         )

#     print("Phase 5-6: Selecting best model by macro F1...")
#     best_name = max(results, key=lambda n: results[n][config.SELECTION_METRIC])
#     best_model = fitted_models[best_name]
#     print(f"  Best model: {best_name} (F1-macro={results[best_name]['f1_macro']:.3f})")

#     print("Saving artifacts...")
#     joblib.dump(best_model, config.BEST_MODEL_PATH)
#     joblib.dump(vectorizer, config.VECTORIZER_PATH)
#     joblib.dump(label_encoder, config.LABEL_ENCODER_PATH)

#     metadata = {
#         "model_name": best_name,
#         "training_date": datetime.now().isoformat(timespec="seconds"),
#         "classes": list(classes),
#         "is_demo_dataset": is_demo,
#         "dataset_rows": len(df),
#         "class_distribution": class_counts,
#         "text_column": config.TEXT_COLUMN,
#         "target_column": config.TARGET_COLUMN,
#         "tfidf_vocab_size": len(vectorizer.vocabulary_),
#         "selection_metric": config.SELECTION_METRIC,
#         "best_model_metrics": {
#             k: v for k, v in results[best_name].items()
#             if k not in ("confusion_matrix",)
#         },
#         "all_models_compared": list(results.keys()),
#     }
#     with open(config.METADATA_PATH, "w") as f:
#         json.dump(metadata, f, indent=2)

#     with open(config.METRICS_PATH, "w") as f:
#         json.dump(results, f, indent=2)

#     print("\nDone. Artifacts saved to models/:")
#     print(f"  - {config.BEST_MODEL_PATH}")
#     print(f"  - {config.VECTORIZER_PATH}")
#     print(f"  - {config.LABEL_ENCODER_PATH}")
#     print(f"  - {config.METADATA_PATH}")
#     print(f"  - {config.METRICS_PATH}")


# if __name__ == "__main__":
#     main()
"""
End-to-end training pipeline.

Run:
    python -m src.train

What it does:
1. Loads data/dataset.csv (falls back to data/dataset_sample.csv).
2. Cleans + preprocesses text.
3. Splits into train/validation/test.
4. Fits TF-IDF on the training split only.
5. For each model (Logistic Regression, Naive Bayes, Linear SVM), tunes one
   hyperparameter on the VALIDATION split.
6. Picks the best model by validation macro F1.
7. Reports final metrics for every model on the held-out TEST split.
8. Saves the model, vectorizer, label encoder, metadata, and metrics to disk.
"""

import json
import os
import sys
from datetime import datetime

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src.evaluate import evaluate_predictions
from src.features import build_vectorizer
from src.preprocessing import preprocess_series


def load_dataset() -> tuple[pd.DataFrame, bool]:
    """Load the real dataset if present, otherwise fall back to the demo one."""
    if os.path.exists(config.DATASET_PATH):
        df = pd.read_csv(config.DATASET_PATH)
        is_demo = False
    elif os.path.exists(config.SAMPLE_DATASET_PATH):
        df = pd.read_csv(config.SAMPLE_DATASET_PATH)
        is_demo = True
        print(
            "[WARNING] data/dataset.csv not found. Falling back to "
            "data/dataset_sample.csv. Copy your real CSV to data/dataset.csv "
            "so the app does not flag it as a demo dataset."
        )
    else:
        raise FileNotFoundError(
            "No dataset found. Place a CSV at data/dataset.csv (see data/README.md)."
        )

    if config.TEXT_COLUMN not in df.columns or config.TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Dataset must contain columns '{config.TEXT_COLUMN}' and "
            f"'{config.TARGET_COLUMN}'. Found columns: {list(df.columns)}. "
            f"Update TEXT_COLUMN / TARGET_COLUMN in src/config.py to match your CSV."
        )

    return df, is_demo


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna(subset=[config.TEXT_COLUMN, config.TARGET_COLUMN])
    df[config.TEXT_COLUMN] = df[config.TEXT_COLUMN].astype(str).str.strip()
    df = df[df[config.TEXT_COLUMN].str.len() > 0]
    df = df.drop_duplicates(subset=[config.TEXT_COLUMN])
    df[config.TARGET_COLUMN] = df[config.TARGET_COLUMN].astype(str).str.strip().str.lower()
    return df.reset_index(drop=True)


def get_candidates():
    """
    Returns {model_name: (param_name, grid, factory)} where factory(value)
    builds a fresh, unfitted model for that hyperparameter value.
    """
    return {
        "Logistic Regression": (
            "C",
            config.LR_C_GRID,
            lambda c: LogisticRegression(
                C=c, max_iter=2000, class_weight="balanced",
                random_state=config.RANDOM_STATE,
            ),
        ),
        "Multinomial Naive Bayes": (
            "alpha",
            config.NB_ALPHA_GRID,
            lambda a: MultinomialNB(alpha=a),
        ),
        "Linear SVM": (
            "C",
            config.SVM_C_GRID,
            lambda c: LinearSVC(
                C=c, class_weight="balanced", random_state=config.RANDOM_STATE,
            ),
        ),
    }


def tune_model(name, param_name, grid, factory, X_train, y_train, X_val, y_val):
    """Try every value in grid; return (best_model, best_value, best_val_f1)."""
    best = (None, None, -1.0)
    for value in grid:
        model = factory(value)
        model.fit(X_train, y_train)
        val_f1 = f1_score(y_val, model.predict(X_val), average="macro", zero_division=0)
        print(f"    {name}: {param_name}={value}  val F1-macro={val_f1:.3f}")
        if val_f1 > best[2]:
            best = (model, value, val_f1)
    return best


def main():
    os.makedirs(config.MODELS_DIR, exist_ok=True)

    print("Phase 1: Loading dataset...")
    df, is_demo = load_dataset()
    print(f"  Loaded {len(df)} raw rows. Demo dataset: {is_demo}")

    print("Phase 1: Cleaning dataset...")
    df = clean_dataset(df)
    print(f"  {len(df)} rows after cleaning (dropna + dedupe).")
    class_counts = df[config.TARGET_COLUMN].value_counts().to_dict()
    print(f"  Class distribution: {class_counts}")

    print("Phase 2: NLP preprocessing...")
    df["clean_text"] = preprocess_series(df[config.TEXT_COLUMN])
    df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)
    print(f"  {len(df)} rows remain after preprocessing (empty-after-clean rows dropped).")

    print("Encoding labels...")
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[config.TARGET_COLUMN])
    classes = label_encoder.classes_
    print(f"  Classes: {list(classes)}")

    print("Splitting train/val/test...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        df["clean_text"], y, test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE, stratify=y,
    )
    val_ratio = config.VALIDATION_SIZE / (1 - config.TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio,
        random_state=config.RANDOM_STATE, stratify=y_temp,
    )
    print(f"  Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

    print("Phase 3: Fitting TF-IDF vectorizer on training data...")
    vectorizer = build_vectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    X_test_vec = vectorizer.transform(X_test)
    print(f"  Vocabulary size: {len(vectorizer.vocabulary_)}")

    print("Phase 4: Tuning + training models (hyperparameters chosen on validation)...")
    results = {}
    fitted_models = {}
    best_params = {}
    val_scores = {}

    for name, (param_name, grid, factory) in get_candidates().items():
        print(f"  {name}")
        model, value, val_f1 = tune_model(
            name, param_name, grid, factory, X_train_vec, y_train, X_val_vec, y_val
        )
        fitted_models[name] = model
        best_params[name] = {param_name: value}
        val_scores[name] = val_f1

        # Final honest metrics on the untouched test split
        y_pred = model.predict(X_test_vec)
        y_proba = model.predict_proba(X_test_vec) if hasattr(model, "predict_proba") else None

        metrics = evaluate_predictions(y_test, y_pred, y_proba, classes=list(range(len(classes))))
        metrics["classes"] = list(classes)
        metrics["per_class"] = {
            classes[int(k)]: v for k, v in metrics["per_class"].items()
        }
        metrics["supports_probability"] = hasattr(model, "predict_proba")
        metrics["best_params"] = {param_name: value}
        metrics["val_f1_macro"] = float(val_f1)
        results[name] = metrics
        print(
            f"    -> best {param_name}={value}  TEST Accuracy={metrics['accuracy']:.3f}  "
            f"F1-macro={metrics['f1_macro']:.3f}  F1-weighted={metrics['f1_weighted']:.3f}"
        )

    print("Phase 5-6: Selecting best model by VALIDATION macro F1...")
    best_name = max(val_scores, key=val_scores.get)
    best_model = fitted_models[best_name]
    print(
        f"  Best model: {best_name} {best_params[best_name]} "
        f"(val F1-macro={val_scores[best_name]:.3f}, "
        f"test F1-macro={results[best_name]['f1_macro']:.3f})"
    )

    print("Saving artifacts...")
    joblib.dump(best_model, config.BEST_MODEL_PATH)
    joblib.dump(vectorizer, config.VECTORIZER_PATH)
    joblib.dump(label_encoder, config.LABEL_ENCODER_PATH)

    metadata = {
        "model_name": best_name,
        "best_params": best_params[best_name],
        "training_date": datetime.now().isoformat(timespec="seconds"),
        "classes": list(classes),
        "is_demo_dataset": is_demo,
        "dataset_rows": len(df),
        "class_distribution": class_counts,
        "text_column": config.TEXT_COLUMN,
        "target_column": config.TARGET_COLUMN,
        "tfidf_vocab_size": len(vectorizer.vocabulary_),
        "selection_metric": "validation macro F1",
        "best_model_metrics": {
            k: v for k, v in results[best_name].items()
            if k not in ("confusion_matrix",)
        },
        "all_models_compared": list(results.keys()),
    }
    with open(config.METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    with open(config.METRICS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print("\nDone. Artifacts saved to models/:")
    print(f"  - {config.BEST_MODEL_PATH}")
    print(f"  - {config.VECTORIZER_PATH}")
    print(f"  - {config.LABEL_ENCODER_PATH}")
    print(f"  - {config.METADATA_PATH}")
    print(f"  - {config.METRICS_PATH}")


if __name__ == "__main__":
    main()