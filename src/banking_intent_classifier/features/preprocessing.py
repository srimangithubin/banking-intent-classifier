import re
import unicodedata


def normalize_text(text: str) -> str:
    """Apply minimal normalization to input text."""

    if not isinstance(text, str):
        raise TypeError(
            f"Expected text to be str, got {type(text).__name__}."
        )

    text = unicodedata.normalize("NFKC", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()

def normalize_texts(texts):
    """Normalize a collection of text strings"""

    return [
        normalize_text(text) for text in texts
    ]
