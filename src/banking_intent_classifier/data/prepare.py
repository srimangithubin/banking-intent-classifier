from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


def load_config(config_path: Path) -> dict:
    """Load a YAML configuration file."""

    if not config_path.exists():
        raise FileNotFoundError(
            f'config file not found: {config_path}'
        )

    with config_path.open('r', encoding='utf-8') as file:
        return yaml.safe_load(file)

def prepare_data(config: dict,
                 project_root: Path) -> None:
    """Create train, validation, and test datasets"""

    data_config = config['data']
    split_config = config['split']
    random_state = config['project']['random_state']

    raw_train_path = project_root / data_config['raw_train_path']
    raw_test_path = project_root / data_config['raw_test_path']

    train_df = pd.read_csv(raw_train_path)
    test_df = pd.read_csv(raw_test_path)

    train_df, validation_df = train_test_split(
        train_df,
        test_size = split_config['validation_size'],
        random_state = random_state,
        stratify = train_df['category'],
    )

    output_paths = {
        'train': project_root / data_config['processed_train_path'],
        'validation': project_root / data_config['processed_validation_path'],
        'test': project_root / data_config['processed_test_path'],
    }

    for output_path in output_paths.values():
        output_path.parent.mkdir(parents=True, exist_ok=True)


    train_df.to_csv(output_paths['train'], index=False)
    validation_df.to_csv(output_paths['validation'], index=False)
    test_df.to_csv(output_paths['test'], index=False)

    print(f"Train shape: {train_df.shape}")
    print(f"Validation shape: {validation_df.shape}")
    print(f"Test shape: {test_df.shape}")

    print("Processed datasets saved successfully.")

def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    config_path = project_root / 'configs' / 'config.yaml'
    config = load_config(config_path)

    prepare_data(config, project_root)

if __name__ == "__main__":
    main()
