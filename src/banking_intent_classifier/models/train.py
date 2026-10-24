# step 1 #
from pathlib import Path

import joblib
import pandas as pd

from sklearn import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FunctionTransformer, Pipeline
from sklearn.svm import LinearSVC

from banking_intent_classifier.features.preprocessing import normalize_texts

# step 2 #
PROJECT_ROOT = Path(__file__).resolve().parents[3]

TRAIN_PATH = (PROJECT_ROOT / "data" / "processed" / "train.csv")
VALIDATION_PATH = (PROJECT_ROOT / "data" / "processed" / "validation.csv")

MODEL_DIR = PROJECT_ROOT / "artifacts" / "models"

MODEL_PATH = MODEL_DIR / 'banking_intent_classifier_model.joblib'

# Step 3 #
def build_model() -> pipeline:
     """Build the final Version 1 intent classification pipeline."""

     return Pipeline(
          [('normalize', FunctionTransformer(normalize_texts, validate=False)),
           ('vectorizer', TfidfVectorizer(lowercase=False, ngram_range=(1,2), min_df = 1, sublinear_tf = True)),
           ('classifier', LinearSVC(C=1.0, random_state=42))
           ]
     )

# step 4 #
def load_training_data() -> pd.DataFrame:
    """Load and combine train and validation data."""

    train_df = pd.read_csv(TRAIN_PATH)
    validation_df = pd.read_csv(VALIDATION_PATH)

    combined_df = pd.concat([train_df, validation_df], ignore_index=True)
    return combined_df

# step 5 #

def train_model(
          training_df: pd.DataFrame,
) -> Pipeline:
    """Train the final intent classification pipeline."""

    model = build_model()

    X_train = training_df['text']
    y_train = training_df['category']

    model.fit(X_train, y_train)

    return model

# step 6 #
def save_model(
        model: Pipeline,
) -> None:
    """Save the trained model to disk."""

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(
        f'Model saved to {MODEL_PATH}'
    )

# step 7 #
def main() -> None:
    """Train and save the final Version 1 Model"""

    training_df = load_training_data()
    print(
        f'Training data loaded. Shape: {training_df.shape}'
    )

    print(
        f'Number of classes:'
        f'{training_df['category'].nunique()}'
    )

    model = train_model(training_df)

    save_model(model)

if __name__ == '__main__':
    main()
