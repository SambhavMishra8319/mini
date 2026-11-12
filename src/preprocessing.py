# """
# NLP preprocessing pipeline.

# The SAME `preprocess_text` function is used during training and during
# inference so that the vectorizer/model always see text in the same format.

# Design notes:
# - No external corpus downloads (e.g. NLTK's `stopwords`) are required, so the
#   project works out-of-the-box on a clean machine with no internet access.
# - Negation words (no, not, never, n't, etc.) are preserved because they
#   change meaning ("not happy" vs "happy").
# """

# import re
# import string
# from typing import List

# import pandas as pd

# # A compact, curated English stopword list. Negation & intensifier words are
# # deliberately EXCLUDED so meaning-changing words are preserved.
# STOPWORDS = {
#     "a", "an", "the", "and", "or", "but", "if", "then", "so", "of", "at",
#     "by", "for", "with", "about", "against", "between", "into", "through",
#     "during", "before", "after", "above", "below", "to", "from", "up",
#     "down", "in", "out", "on", "off", "over", "under", "again", "further",
#     "once", "here", "there", "when", "where", "why", "how", "all", "any",
#     "both", "each", "few", "more", "most", "other", "some", "such", "only",
#     "own", "same", "than", "too", "very", "s", "t", "can", "will", "just",
#     "should", "now", "is", "am", "are", "was", "were", "be", "been", "being",
#     "have", "has", "had", "having", "do", "does", "did", "doing", "would",
#     "could", "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
#     "you", "your", "yours", "yourself", "yourselves", "he", "him", "his",
#     "himself", "she", "her", "hers", "herself", "it", "its", "itself",
#     "they", "them", "their", "theirs", "themselves", "what", "which", "who",
#     "whom", "this", "that", "these", "those", "as", "until", "while", "im",
# }

# # Negation / intensifier tokens explicitly kept even if they resemble stopwords.
# NEGATIONS = {
#     "no", "not", "nor", "never", "none", "n't", "cant", "cannot", "wont",
#     "dont", "doesnt", "didnt", "isnt", "arent", "wasnt", "werent", "hasnt",
#     "havent", "hadnt", "shouldnt", "wouldnt", "couldnt",
# }

# URL_RE = re.compile(r"https?://\S+|www\.\S+")
# MENTION_RE = re.compile(r"@\w+")
# HASHTAG_RE = re.compile(r"#(\w+)")
# EMOJI_RE = re.compile(
#     "["
#     "\U0001F300-\U0001FAFF"
#     "\U00002600-\U000027BF"
#     "\U0001F1E0-\U0001F1FF"
#     "]+",
#     flags=re.UNICODE,
# )
# MULTISPACE_RE = re.compile(r"\s+")
# NON_ALPHA_RE = re.compile(r"[^a-z\s']")


# def clean_raw_text(text: str) -> str:
#     """Lowercase and strip URLs, mentions, hashtags (keep the word), emojis."""
#     text = text.lower()
#     text = URL_RE.sub(" ", text)
#     text = MENTION_RE.sub(" ", text)
#     text = HASHTAG_RE.sub(r"\1", text)  # keep hashtag word, drop the '#'
#     text = EMOJI_RE.sub(" ", text)
#     return text


# def tokenize(text: str) -> List[str]:
#     text = NON_ALPHA_RE.sub(" ", text)
#     text = MULTISPACE_RE.sub(" ", text).strip()
#     return text.split()


# def remove_stopwords(tokens: List[str]) -> List[str]:
#     return [t for t in tokens if (t not in STOPWORDS or t in NEGATIONS) and len(t) > 1]


# def preprocess_text(text: str) -> str:
#     """
#     Full preprocessing pipeline applied to a single raw text string.
#     Returns a cleaned string ready for TF-IDF vectorization.
#     Safe against None / NaN / empty input.
#     """
#     if text is None or (isinstance(text, float) and pd.isna(text)):
#         return ""
#     text = str(text).strip()
#     if not text:
#         return ""

#     text = clean_raw_text(text)
#     tokens = tokenize(text)
#     tokens = remove_stopwords(tokens)
#     return " ".join(tokens)


# def preprocess_series(texts: pd.Series) -> pd.Series:
#     """Vectorized helper to preprocess an entire pandas Series of text."""
#     return texts.astype(str).apply(preprocess_text)


# def basic_text_stats(text: str) -> dict:
#     """Return simple, honest textual statistics (no clinical claims)."""
#     if text is None:
#         text = ""
#     text = str(text)
#     words = text.split()
#     return {
#         "character_count": len(text),
#         "word_count": len(words),
#         "avg_word_length": round(sum(len(w) for w in words) / len(words), 2) if words else 0.0,
#     }

"""
NLP preprocessing pipeline.

The SAME `preprocess_text` function is used during training and during
inference so that the vectorizer/model always see text in the same format.

Design notes:
- No external corpus downloads (e.g. NLTK's `stopwords`) are required.
- Negation words (no, not, never, n't, etc.) are preserved.
- First-person pronouns (i, me, my, myself) and intensifiers (very, too,
  just, so, all, only) are KEPT: they are known signals in depression and
  suicidal text, so removing them hurts accuracy.
"""

import re
from typing import List

import pandas as pd

# Compact stopword list. Pronouns / intensifiers are deliberately NOT here.
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "of", "at",
    "by", "for", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "to", "from", "up",
    "down", "in", "out", "on", "off", "over", "under", "again", "further",
    "once", "here", "there", "when", "where", "why", "how", "each", "few",
    "more", "most", "other", "some", "such", "own", "same", "than", "s", "t",
    "can", "will", "should", "now", "is", "am", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "having", "do", "does", "did",
    "doing", "would", "could", "you", "your", "yours", "yourself",
    "yourselves", "he", "him", "his", "himself", "she", "her", "hers",
    "herself", "it", "its", "itself", "they", "them", "their", "theirs",
    "themselves", "what", "which", "who", "whom", "this", "that", "these",
    "those", "as", "until", "while",
}

# Negation tokens explicitly kept even if they resemble stopwords.
NEGATIONS = {
    "no", "not", "nor", "never", "none", "n't", "cant", "cannot", "wont",
    "dont", "doesnt", "didnt", "isnt", "arent", "wasnt", "werent", "hasnt",
    "havent", "hadnt", "shouldnt", "wouldnt", "couldnt",
}

# Single-character tokens that carry meaning and must not be dropped.
KEEP_SHORT = {"i"}

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F1E0-\U0001F1FF"
    "]+",
    flags=re.UNICODE,
)
MULTISPACE_RE = re.compile(r"\s+")
NON_ALPHA_RE = re.compile(r"[^a-z\s']")


def clean_raw_text(text: str) -> str:
    """Lowercase and strip URLs, mentions, hashtags (keep the word), emojis."""
    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = HASHTAG_RE.sub(r"\1", text)  # keep hashtag word, drop the '#'
    text = EMOJI_RE.sub(" ", text)
    return text


def tokenize(text: str) -> List[str]:
    text = NON_ALPHA_RE.sub(" ", text)
    text = MULTISPACE_RE.sub(" ", text).strip()
    return text.split()


def remove_stopwords(tokens: List[str]) -> List[str]:
    return [
        t for t in tokens
        if (t not in STOPWORDS or t in NEGATIONS)
        and (len(t) > 1 or t in KEEP_SHORT)
    ]


def preprocess_text(text: str) -> str:
    """
    Full preprocessing pipeline applied to a single raw text string.
    Returns a cleaned string ready for TF-IDF vectorization.
    Safe against None / NaN / empty input.
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    text = str(text).strip()
    if not text:
        return ""

    text = clean_raw_text(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    return " ".join(tokens)


def preprocess_series(texts: pd.Series) -> pd.Series:
    """Vectorized helper to preprocess an entire pandas Series of text."""
    return texts.astype(str).apply(preprocess_text)


def basic_text_stats(text: str) -> dict:
    """Return simple, honest textual statistics (no clinical claims)."""
    if text is None:
        text = ""
    text = str(text)
    words = text.split()
    return {
        "character_count": len(text),
        "word_count": len(words),
        "avg_word_length": round(sum(len(w) for w in words) / len(words), 2) if words else 0.0,
    }