"""Tests the package when using a sentence transformers model for embeddings."""
# make pylint play nice with pytest fixtures
# pylint: disable=redefined-outer-name, unused-import
from tests.fixtures import cat_text
from zerohero import make_zero_shot_classifier


def test_sentence_transformers(cat_text):
    """Tests whether correctly specified sentence transformers models work with the package."""
    categories = ["cat", "dog", "mouse", "human"]
    zsc = make_zero_shot_classifier(
        categories=categories,
        model_type="sentence-transformers",
        model_name="paraphrase-albert-small-v2",
    )
    result = zsc(cat_text)
    assert result["category"] == "cat"
