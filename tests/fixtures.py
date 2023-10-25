"""Fixtures for pytest."""

import os

import pytest
from dotenv import load_dotenv

load_dotenv()


@pytest.fixture
def openai_api_key() -> None:
    """OpenAI API Key."""
    return os.environ["OPENAI_API_KEY"]


@pytest.fixture
def cat_text() -> None:
    """Some text about cats."""
    return (
        "The cat (Felis catus), "
        "commonly referred to as the domestic cat or house cat, "
        "is the only domesticated species in the family Felidae. "
        "Recent advances in archaeology and genetics have shown that "
        "the domestication of the cat occurred in the Near East around 7500 BC. "
        "It is commonly kept as a house pet and farm cat, "
        "but also ranges freely as a feral cat avoiding human contact. "
        "It is valued by humans for companionship and its ability to kill vermin. "
        "Because of its retractable claws it is adapted to killing small prey like mice and rats. "
        "It has a strong flexible body, "
        "quick reflexes, "
        "sharp teeth, "
        "and its night vision and sense of smell are well developed. "
        "It is a social species, "
        "but a solitary hunter and a crepuscular predator. "
        "Cat communication includes vocalizations like meowing, "
        "purring, "
        "trilling, "
        "hissing, "
        "growling, "
        "and grunting as well as cat body language. "
        "It can hear sounds too faint or too high in frequency for human ears, "
        "such as those made by small mammals. "
        "It also secretes and perceives pheromones."
    )
