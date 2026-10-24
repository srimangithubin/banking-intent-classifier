from banking_intent_classifier.models.predict import load_model, predict_intent

def test_predict_intent_returns_string():
    model = load_model()
    prediction = predict_intent(
        "Why hasn't my card arrived?", model=model
    )

    assert isinstance(prediction, str)

def test_predict_intent_returns_known_class():
    model = load_model()
    prediction = predict_intent(
        "Why hasn't my card arrived?", model=model
    )

    known_class = set(model.named_steps['classifier'].classes_)

    assert prediction in known_class

def test_card_arrival_prediction():
    model = load_model()
    prediction = predict_intent(
        "Why hasn't my card arrived?", model=model
    )
    assert prediction == "card_arrival"
