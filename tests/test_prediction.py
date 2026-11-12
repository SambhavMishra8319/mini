import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict import ArtifactsNotFoundError, artifacts_available, predict


def test_empty_text_raises_value_error():
    if not artifacts_available():
        pytest.skip("Model not trained yet -- run `python -m src.train` first.")
    with pytest.raises(ValueError):
        predict("")


def test_none_text_raises_value_error():
    if not artifacts_available():
        pytest.skip("Model not trained yet -- run `python -m src.train` first.")
    with pytest.raises(ValueError):
        predict(None)


def test_predict_returns_valid_result():
    if not artifacts_available():
        pytest.skip("Model not trained yet -- run `python -m src.train` first.")
    result = predict("I feel exhausted and empty, nothing makes me happy anymore")
    assert result.predicted_class
    assert result.risk_label
    assert isinstance(result.text_stats, dict)
    assert result.text_stats["word_count"] > 0


def test_predict_confidence_in_valid_range():
    if not artifacts_available():
        pytest.skip("Model not trained yet -- run `python -m src.train` first.")
    result = predict("Had a great day today, feeling happy and relaxed.")
    if result.supports_probability:
        assert 0.0 <= result.confidence <= 1.0
        assert abs(sum(result.class_probabilities.values()) - 1.0) < 1e-6


def test_missing_artifacts_raises_clear_error(monkeypatch):
    import src.predict as predict_module

    monkeypatch.setattr(predict_module, "artifacts_available", lambda: False)
    with pytest.raises(ArtifactsNotFoundError):
        predict_module.load_artifacts()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
