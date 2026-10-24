from contextlib import asynccontextmanager
from fastapi import FastAPI, Request

from pydantic import BaseModel, ConfigDict, Field

from banking_intent_classifier.models.predict import load_model, predict_intent

class PredictionRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    text : str = Field(
        min_length=1,
        max_length=500,
    )

class PredictionResnponse(BaseModel):
    intent: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_model()
    yield
    del app.state.model

app = FastAPI(
    title = "Banking Intent Classifier",
    version = "1.0.0",
    lifespan=lifespan
)

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post('/predict', response_model=PredictionResnponse)
def predict(payload : PredictionRequest, request: Request):
    model = request.app.state.model

    intent = predict_intent(
        payload.text,
        model
    )

    return PredictionResnponse(intent=intent)


### @asynccontextmanager are decorators applied directly.

### @app.post("/predict") is a decorator factory: app.post("/predict") first returns a decorator, which then receives predict.
