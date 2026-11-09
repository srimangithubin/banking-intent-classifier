# BANKING77 — Banking Intent Classification API

[![BANKING77 CI](https://github.com/srimangithubin/banking-intent-classifier/actions/workflows/ci.yml/badge.svg)](https://github.com/srimangithubin/banking-intent-classifier/actions/workflows/ci.yml)

An end-to-end machine learning project that classifies customer banking questions into **77 intents**. I trained and evaluated a text-classification pipeline, packaged it as a FastAPI service, containerized it with Docker, and deployed it to **Azure Container Apps** with automated GitHub Actions CI/CD.

**Repository:** [srimangithubin/banking-intent-classifier](https://github.com/srimangithubin/banking-intent-classifier)

## Project highlights

| Area | Implementation |
|---|---|
| Task | Multiclass banking intent classification (77 categories) |
| Dataset | BANKING77 — 10,003 training examples and 3,080 test examples |
| Selected model | TF-IDF (unigrams) + Linear Support Vector Classifier (`LinearSVC`, `C=1.0`) |
| Held-out test performance | **89.29% accuracy**, approximately **0.893 macro-F1** |
| Inference API | FastAPI, Pydantic validation, Uvicorn |
| Testing | pytest and FastAPI TestClient |
| Packaging | Docker, Python 3.13 |
| CI/CD | GitHub Actions → GitHub Container Registry (GHCR) → Azure Container Apps |
| Deployment authentication | GitHub OIDC federation, Azure user-assigned managed identity and scoped RBAC |

## Architecture

![BANKING77 CI/CD and Azure architecture](docs/architecture.png)

**Deployment flow:** push to `main` → install and test → Ruff lint → Docker build → publish SHA-tagged image to GHCR → GitHub OIDC login to Azure → update Container App → verify the configured image and `/health` endpoint.

**Request flow:** HTTPS request → Azure Container Apps ingress (target port `8000`) → Uvicorn/FastAPI → Pydantic input validation → loaded TF-IDF + LinearSVC pipeline → predicted banking intent returned as JSON.

The model is loaded during the application's startup lifespan and reused across prediction requests; it is **not retrained per request**.

## Example API usage

### Health check

```http
GET /health
```

```json
{"status": "ok"}
```

### Predict an intent

```http
POST /predict
Content-Type: application/json
```

```json
{"text": "How can I change my card PIN?"}
```

Example response observed from the deployed API:

```json
{"intent": "change_pin"}
```

FastAPI also provides interactive API documentation at **`/docs`** and the OpenAPI schema at **`/openapi.json`**.

> **Live API URL:** Add the current Azure Container App URL here if you want the endpoint public. It can change when an Azure resource is recreated or deleted.

## Run locally

Requirements: Python 3.13, with the project and model artifact available.

```bash
git clone https://github.com/srimangithubin/banking-intent-classifier.git
cd banking-intent-classifier
python -m pip install .
python -m uvicorn banking_intent_classifier.api.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` to try the API, or call `/health`.

### Run the tests

```bash
python -m pip install pytest httpx ruff
python -m pytest tests/ -v
python -m ruff check src tests
```

### Run with Docker

```bash
docker build -t banking77-api:local .
docker run --rm -p 8000:8000 banking77-api:local
```

Then open `http://localhost:8000/docs`.

## Automated deployment

The [GitHub Actions workflow](.github/workflows/ci.yml) runs quality checks for changes and publishes/deploys on pushes to `main`:

1. **Quality:** install dependencies, run automated tests and Ruff, build the Docker image.
2. **Publish:** authenticate to GHCR and push an image tagged with the exact commit SHA.
3. **Deploy:** authenticate to Azure using short-lived OIDC tokens; update the existing `banking77-api` Container App to use that image; verify its configured image and perform a public health check.

The Azure managed identity uses a federated credential to trust the GitHub repository's `main` branch. Azure RBAC grants it permission to manage the target Container App. **No long-lived Azure client secret is stored in GitHub.**

## Engineering choices

- **Classical ML baseline:** TF-IDF + LinearSVC provides a compact, relatively fast classifier for this labelled-intent task.
- **Training/serving consistency:** preprocessing and classification are saved together in one scikit-learn pipeline.
- **Input validation:** Pydantic checks the prediction request's text field (1–1,000 characters).
- **Reproducibility:** Git commit SHA tags connect a published container image to its source revision.
- **Separation of responsibilities:** GHCR stores the image; Azure runs the API; GitHub Actions handles verification and deployment.

## Limitations and future improvements

- Test metrics measure performance on the held-out BANKING77 dataset, not real-world live banking traffic.
- `/health` confirms API availability; it is not a complete model-quality or deployment-readiness assessment.
- The current project does not implement long-term production observability, drift alerts, automated retraining or end-to-end deployment rollback.
- Future iterations could add structured monitoring, privacy-aware drift evaluation and a controlled candidate-model approval process.

## Skills demonstrated

Python · scikit-learn · NLP · TF-IDF · LinearSVC · FastAPI · Pydantic · pytest · Ruff · Docker · GitHub Actions · GHCR · Azure Container Apps · OIDC · Azure RBAC
