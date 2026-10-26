from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "test.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / 'artifacts'
    /'models'
    /'banking_intent_classifier_model.joblib'
)
METRICS_DIR = (
    PROJECT_ROOT
    /'artifacts'
    /'metrics'
)


def load_model():
    """Load the trained intent classification pipeline."""

    return joblib.load(MODEL_PATH)

def load_test_data() -> pd.DataFrame:
    """Load the test data."""

    return pd.read_csv(TEST_PATH)

def evaluate_model(
        model,
        test_df : pd.DataFrame
) -> dict:
    """Evaluate the trained model on official Test set"""

    X_test = test_df['text']
    y_test = test_df['category']

    predictions = model.predict(X_test)

    metrics = {
        'accuracy': accuracy_score(
            y_test,
            predictions
        ),

        'macro_f1': f1_score(
            y_test,
            predictions,
            average='macro'
        ),

        'weighted_f1': f1_score(
            y_test,
            predictions,
            average='weighted'
        ),
    }

    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0
    )

    return metrics, report

def save_results(
        metrics: dict,
        report: dict,
) -> None:
    """Save the evaluation results to disk."""

    METRICS_DIR.mkdir(parents=True, exist_ok = True)

    metrics_df = pd.DataFrame([metrics])

    metrics_df.to_csv(
        METRICS_DIR / 'final_test_metrics.csv',
        index = False
    )

    report_df = pd.DataFrame(report).T

    report_df.to_csv(
        METRICS_DIR / 'final_test_classification_report.csv',
        index = False
        )

def main() -> None:
    """Load the trained model and evaluate it on the official test set."""

    model = load_model()
    test_df = load_test_data()

    print(
        f"Test rows: {len(test_df)}"
    )

    metrics, report = evaluate_model(
        model,
        test_df,
    )

    print(
        "Final test accuracy:",
        metrics["accuracy"],
    )

    print(
        "Final test Macro F1:",
        metrics["macro_f1"],
    )

    print(
        "Final test Weighted F1:",
        metrics["weighted_f1"],
    )

    save_results(
        metrics,
        report,
    )


if __name__ == "__main__":
    main()
