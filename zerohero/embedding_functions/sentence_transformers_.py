"""Makes a embedding function from a sentence_transformers model."""
from sentence_transformers import SentenceTransformer


def make_sentence_transformers_embedding_function(model_name: str):
    """Makes a embedding function from a sentence_transformers model.

    Args:
        model_name (str): Name of a pretrained Sentence Transformers model.

    Returns:
        Callable: A function that maps text to an embedding.
    """
    model = SentenceTransformer(model_name)

    def sentence_transformers_embedding_function(text):
        return model.encode(text)

    return sentence_transformers_embedding_function
