from zerohero import make_zero_shot_classifier

def test_sentence_transformers():
    categories = ["positive", "neutral", "negative"]
    zsc = make_zero_shot_classifier(categories=categories, model_type="sentence-transformers", model_name="paraphrase-albert-small-v2")
    print(zsc("I really like zerohero!") )

