from pathlib import Path
import joblib
import os

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_MODEL_PATH = (PROJECT_ROOT
    / 'artifacts'
    / 'models'
    / 'banking_intent_classifier_model.joblib')

MODEL_PATH = Path(
    os.getenv(
        "BANKING77_MODEL_PATH",
        str(DEFAULT_MODEL_PATH)
    )
)

def load_model():
    """Load the trained intent classification pipeline."""

    return joblib.load(MODEL_PATH)

def predict_intent(
        text: str,
        model
) -> str:
    """Predict the intent category for a given text using the trained model."""

    prediction = model.predict(
        [text]
    )

    return prediction[0]

def main() -> None:
    """Load the trained model and predict the intent for a sample text"""

    model = load_model()

    example_text = ("why hasn't my card arrived?")

    predicted_intent = predict_intent(
        text=example_text,
        model=model
    )

    print(f"Predicted intent for the text '{example_text}': {predicted_intent}")

if __name__ == '__main__':
    main()
