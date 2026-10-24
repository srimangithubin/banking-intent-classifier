import pytest

from banking_intent_classifier.features.preprocessing import normalize_text


def test_normalize_text_removes_extra_whitespace():
    text = "   my    card   has   not arrived   "

    result = normalize_text(text)

    assert result == "my card has not arrived"


def test_normalize_text_preserves_punctuation():
    text = "Why hasn't my card arrived?"

    result = normalize_text(text)

    assert result == "Why hasn't my card arrived?"


def test_normalize_text_preserves_case():
    text = "My CARD Has Arrived"

    result = normalize_text(text)

    assert result == "My CARD Has Arrived"


def test_normalize_text_rejects_non_string():
    with pytest.raises(TypeError):
        normalize_text(123)
