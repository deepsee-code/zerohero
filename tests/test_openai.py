"""Tests the package when using an OpenAI model for embeddings."""
# make pylint play nice with pytest fixtures
# pylint: disable=redefined-outer-name, unused-import
from tests.fixtures import openai_api_key
from zerohero import make_zero_shot_classifier


def test_sentence_transformers(openai_api_key):
    """Tests whether correctly specified OpenAI model works with the package."""
    categories = ["positive", "neutral", "negative"]
    zsc = make_zero_shot_classifier(
        categories=categories,
        model_type="openai",
        model_name="text-embedding-ada-002",
        openai_api_key=openai_api_key,
    )
    print(zsc("I really like zerohero!"))
