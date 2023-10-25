"""Fixtures for pytest."""

import os

import pytest
from dotenv import load_dotenv

load_dotenv()


@pytest.fixture
def openai_api_key() -> None:
    """OpenAI API Key."""
    return os.environ["OPENAI_API_KEY"]
