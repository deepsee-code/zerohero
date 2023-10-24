from collections import OrderedDict

import openai
import torch
from sentence_transformers import util

from zerohero.embedding_functions.sentence_transformers import (
    make_sentence_transformers_embedding_function,
)
from zerohero.embedding_functions.openai import make_openai_embedding_function


MODEL_TYPES = {"openai", "sentence-transformers"}


def make_zero_shot_classifier(categories, model_type, model_name, openai_api_key=None):
    if not model_type in MODEL_TYPES:
        raise ValueError(
            f"{model_type=} not valid, must be one of {', '.join(MODEL_TYPES)}"
        )

    if model_type == "sentence-transformers":
        embedding_function = make_sentence_transformers_embedding_function(
            model_name=model_name
        )
    if model_type == "openai":
        if not openai_api_key:
            raise ValueError(
                "When using model_type=openai, openai_api_key must be passed."
            )

        openai.api_key = openai_api_key

        model_names = [model["root"] for model in openai.Model.list()["data"]]

        if not model_name in model_names:
            raise ValueError(
                f"{model_name=} not valid, must be one of {', '.join(model_names)}"
            )

        embedding_function = make_openai_embedding_function(
            model_name=model_name, openai_api_key=openai_api_key
        )

    return _make_zero_shot_embedding_classifier(
        categories=categories, embedding_function=embedding_function
    )


def _make_zero_shot_embedding_classifier(categories, embedding_function):
    categories_encoded = torch.tensor(
        [embedding_function(category) for category in categories]
    )

    def embedding_classifier(text):
        text_encoded = torch.tensor(embedding_function(text))
        similarities = util.cos_sim(text_encoded, categories_encoded).flatten()

        category_similarities = dict(
            zip(categories, [float(similarity) for similarity in similarities])
        )

        softmax_similarities = [
            float(sf_sim)
            for sf_sim in torch.nn.functional.softmax(similarities, dim=-1)
        ]

        distribution_sorted = OrderedDict(
            sorted(
                zip(categories, softmax_similarities),
                key=lambda t: t[1],
                reverse=True,
            )
        )

        category = list(distribution_sorted.keys())[0]
        category_confidence = list(distribution_sorted.values())[0]

        return {
            "predicted_category": category,
            "predicted_category_confidence": category_confidence,
            "predicted_category_distribution": dict(distribution_sorted),
            "category_similarities": category_similarities,
        }

    return embedding_classifier
