"""Tests the package when using an OpenAI model for embeddings."""
# make pylint play nice with pytest fixtures
# pylint: disable=redefined-outer-name, unused-import
from tests.fixtures import openai_api_key, cat_text
from zerohero import make_zero_shot_classifier


def test_sentence_transformers(openai_api_key, cat_text):
    """Tests whether correctly specified OpenAI model works with the package."""

    model_names = [
        "text-embedding-3-small",
        "text-embedding-3-large",
        "text-embedding-ada-002",
    ]

    for model_name in model_names:
        categories = ["cat", "dog", "mouse", "human"]
        zsc = make_zero_shot_classifier(
            categories=categories,
            model_type="openai",
            model_name=model_name,
            openai_api_key=openai_api_key,
        )
        result = zsc(cat_text)
        assert result["category"] == "cat"
