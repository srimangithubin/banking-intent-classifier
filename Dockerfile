FROM python:3.13-slim

# 1. Creates /app and moves inside it
WORKDIR /app

# 2. Copies pyproject.toml ./ from your laptop
COPY pyproject.toml README.md ./

COPY src src/.

RUN python -m pip install --no-cache-dir .

COPY artifacts/models/banking_intent_classifier_model.joblib artifacts/models/.

ENV BANKING77_MODEL_PATH=/app/artifacts/models/banking_intent_classifier_model.joblib

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "banking_intent_classifier.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
