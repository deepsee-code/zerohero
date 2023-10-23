from sentence_transformers import SentenceTransformer


def make_sentence_transformers_embedding_function(model_name):
    model = SentenceTransformer(model_name)

    def sentence_transformers_embedding_function(text):
        return model.encode(text)

    return sentence_transformers_embedding_function
