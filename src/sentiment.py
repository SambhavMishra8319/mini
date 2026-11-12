"""
Lightweight lexicon-based sentiment scorer.

This intentionally avoids adding heavy/optional dependencies (e.g. VADER,
TextBlob corpora) so the project installs and runs on a clean, offline
machine. It is explicitly presented in the UI as a supporting textual
signal only -- NOT a mental-health indicator on its own.
"""

from typing import Tuple

POSITIVE_WORDS = {
    "good", "great", "happy", "joy", "joyful", "excited", "love", "loved",
    "amazing", "awesome", "wonderful", "grateful", "thankful", "hopeful",
    "hope", "calm", "relaxed", "peace", "peaceful", "confident", "proud",
    "excellent", "fantastic", "glad", "positive", "motivated", "energetic",
    "fun", "blessed", "optimistic", "strong", "better", "improving",
}

NEGATIVE_WORDS = {
    "bad", "sad", "depressed", "depression", "anxious", "anxiety", "stress",
    "stressed", "stressful", "exhausted", "tired", "hopeless", "worthless",
    "alone", "lonely", "empty", "hurt", "pain", "painful", "cry", "crying",
    "fear", "afraid", "scared", "panic", "worry", "worried", "angry",
    "hate", "hated", "terrible", "awful", "miserable", "suffering",
    "overwhelmed", "numb", "broken", "tired of", "give up", "worthlessness",
    "suicidal", "die", "death", "hurting",
}

NEGATION_WORDS = {"not", "no", "never", "n't", "cant", "cannot", "dont"}


def score_sentiment(text: str) -> Tuple[str, float]:
    """
    Returns (label, score) where score is in [-1, 1].
    A very simple bag-of-words lexicon approach with basic negation flipping.
    This is a supporting textual signal only.
    """
    if not text:
        return "Neutral", 0.0

    tokens = text.lower().split()
    score = 0
    negate_next = False

    for tok in tokens:
        if tok in NEGATION_WORDS:
            negate_next = True
            continue
        polarity = 0
        if tok in POSITIVE_WORDS:
            polarity = 1
        elif tok in NEGATIVE_WORDS:
            polarity = -1

        if polarity != 0 and negate_next:
            polarity *= -1

        score += polarity
        negate_next = False

    if not tokens:
        return "Neutral", 0.0

    normalized = max(-1.0, min(1.0, score / max(1, len(tokens)) * 5))

    if normalized > 0.15:
        label = "Positive"
    elif normalized < -0.15:
        label = "Negative"
    else:
        label = "Neutral"

    return label, round(normalized, 3)
