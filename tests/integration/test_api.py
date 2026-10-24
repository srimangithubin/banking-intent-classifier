import pytest
from fastapi.testclient import TestClient
from banking_intent_classifier.api.main import app

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client

def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict(client):
    response = client.post(
        "/predict",
        json={
            "text": "Where is my card? It hasn't arrived."
        },
    )

    assert response.status_code == 200

    result = response.json()
    assert isinstance(result["intent"], str)

    known_classes = set(
        app.state.model.named_steps["classifier"].classes_
    )
    assert result["intent"] in known_classes

def test_predict_rejects_empty_text(client):
    response = client.post(
        "/predict",
        json={"text": "     "},
    )

    assert response.status_code == 422
