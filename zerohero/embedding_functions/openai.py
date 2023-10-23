import backoff
import tiktoken

OPENAI_MODEL_NAME_TO_MAX_TOKENS = {"text-embedding-ada-002": 8191}


def _backoff_hdlr(details):
    print(
        "Backing off {wait:0.1f} seconds after {tries} tries "
        "\n {exception}".format(**details)
    )


def _openai_lookup_max_tokens(model_name):
    return OPENAI_MODEL_NAME_TO_MAX_TOKENS.get(model_name, 2046)


def make_openai_embedding_function(model_name, openai_api_key):
    openai.api_key = openai_api_key

    max_tokens = _openai_lookup_max_tokens(model_name=model_name)

    encoding = tiktoken.encoding_for_model(model_name)

    # do a dry run to see if embeddings are supported by the passed model
    openai.Embedding.create(input="test", model=model_name)["data"][0]["embedding"]

    # openai can make ai, but can they make api? no.
    @backoff.on_exception(
        backoff.constant, openai.error.RateLimitError, on_backoff=_backoff_hdlr
    )
    @backoff.on_exception(
        backoff.constant, openai.error.APIConnectionError, on_backoff=_backoff_hdlr
    )
    @backoff.on_exception(backoff.expo, openai.error.Timeout, on_backoff=_backoff_hdlr)
    @backoff.on_exception(backoff.expo, openai.error.APIError, on_backoff=_backoff_hdlr)
    def _openai_embedding_function(text):
        truncated_text = encoding.decode(encoding.encode(text)[:max_tokens])

        return openai.Embedding.create(input=truncated_text, model=model_name)["data"][
            0
        ]["embedding"]

    return _openai_embedding_function
