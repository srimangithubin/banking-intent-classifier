from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = {"text", "category"}

EXPECTED_ROWS = {
    "train": 10003,
    "test": 3080,
}

EXPECTED_NUM_CLASSES = 77

def load_data(file_path: Path) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame."""

    if not file_path.exists():
        raise FileNotFoundError(
            f'data file not found: {file_path}'
        )

    return pd.read_csv(file_path)

def validate_dataframe(
        df: pd.DataFrame,
        split_name: str,
) -> None:
    """Validate one BANKING77 dataset split."""

    # 1. check columns
    actual_columns = set(df.columns)

    if actual_columns != REQUIRED_COLUMNS:
        missing_columns = REQUIRED_COLUMNS - actual_columns
        unexpected_columns = actual_columns - REQUIRED_COLUMNS

        raise ValueError(
            f'{split_name} invalid columns.'
            f' Missing: {missing_columns}, '
            f'Unexpected: {unexpected_columns}'
        )

    # 2. check dataset is not empty
    if df.empty:
        raise ValueError(f'{split_name} dataset is empty.')

    # 3. check number of rows
    expected_rows = EXPECTED_ROWS[split_name]
    if len(df) != expected_rows:
        raise ValueError(
            f'{split_name} dataset has {len(df)} rows, '
            f'expected {expected_rows}.'
        )

    # 4. check missing values
    null_counts = df.isnull().sum()

    if null_counts.any():
        raise ValueError(
            f'{split_name} dataset has missing values:\n'
            f'{null_counts}'
        )

    # 5. check blank text
    blank_text_count = (
        df['text'].astype(str).str.strip().eq('').sum()
    )

    if blank_text_count > 0:
        raise ValueError(
            f'{split_name} dataset has {blank_text_count} blank text entries.'
        )


    # 6. check blank category
    blank_category_count = (
        df['category'].astype(str).str.strip().eq('').sum()
    )

    if blank_category_count > 0:
        raise ValueError(
            f'{split_name} dataset has {blank_category_count} blank category entries.'
        )

    # 7. check number of classes
    number_of_classes = df['category'].nunique()

    if number_of_classes != EXPECTED_NUM_CLASSES:
        raise ValueError(
            f'{split_name} dataset has {number_of_classes} unique classes, '
            f'expected {EXPECTED_NUM_CLASSES}.'
        )

    # 8. Report duplicates

    duplicate_count = df.duplicated().sum()

    print(
        f"{split_name}: "
        f"rows={len(df)}, "
        f"classes={number_of_classes}, "
        f"duplicates={duplicate_count}"
    )

def validate_label_consistency(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame
) -> None:
    """Validate that the labels in the test set are a subset of the train set."""

    train_labels = set(train_df['category'].unique())
    test_labels = set(test_df['category'].unique())

    if not test_labels.issubset(train_labels):
        missing_labels = test_labels - train_labels
        raise ValueError(
            f'Test dataset has labels not present in train dataset: {missing_labels}'
)

def main() -> None:
    project_root = Path(__file__).resolve().parents[3]

    raw_data_dir = project_root/'data/raw'

    train_path = raw_data_dir/'train.csv'
    test_path = raw_data_dir/'test.csv'

    train_df = load_data(train_path)
    test_df = load_data(test_path)

    validate_dataframe(train_df, 'train')
    validate_dataframe(test_df, 'test')
    validate_label_consistency(train_df, test_df)

    print("All validations passed successfully.")

if __name__ == "__main__":
    main()
