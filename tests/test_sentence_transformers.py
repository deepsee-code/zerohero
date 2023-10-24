"""Tests the package when using a sentence transformers model for embeddings."""
from zerohero import make_zero_shot_classifier


def test_sentence_transformers():
    """Tests whether correctly specified sentence transformers models work with the package."""
    categories = ["positive", "neutral", "negative"]
    zsc = make_zero_shot_classifier(
        categories=categories,
        model_type="sentence-transformers",
        model_name="paraphrase-albert-small-v2",
    )
    zsc("I really like zerohero!")
