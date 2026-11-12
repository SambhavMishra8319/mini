import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import basic_text_stats, preprocess_text


def test_lowercases_text():
    assert "hello" in preprocess_text("HELLO world")


def test_removes_urls():
    result = preprocess_text("Check this out https://example.com/page now")
    assert "http" not in result
    assert "example" not in result


def test_removes_mentions_and_keeps_hashtag_word():
    result = preprocess_text("@john said #anxiety is real")
    assert "john" not in result
    assert "anxiety" in result


def test_preserves_negation():
    happy = preprocess_text("I am happy today")
    not_happy = preprocess_text("I am not happy today")
    assert "happy" in happy
    assert "not" in not_happy and "happy" in not_happy


def test_empty_input_returns_empty_string():
    assert preprocess_text("") == ""
    assert preprocess_text(None) == ""
    assert preprocess_text("   ") == ""


def test_removes_stopwords():
    result = preprocess_text("This is the best day of my life")
    tokens = result.split()
    assert "the" not in tokens
    assert "is" not in tokens


def test_basic_text_stats():
    stats = basic_text_stats("I am okay today")
    assert stats["word_count"] == 4
    assert stats["character_count"] == 16


def test_basic_text_stats_empty():
    stats = basic_text_stats("")
    assert stats["word_count"] == 0
    assert stats["avg_word_length"] == 0.0


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
