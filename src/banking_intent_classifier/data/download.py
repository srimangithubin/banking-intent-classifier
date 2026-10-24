from pathlib import Path
from urllib.request import urlretrieve

TRAIN_URL = (
    "https://raw.githubusercontent.com/"
    "PolyAI-LDN/task-specific-datasets/"
    "master/banking_data/train.csv"
)


TEST_URL = (
    "https://raw.githubusercontent.com/"
    "PolyAI-LDN/task-specific-datasets/"
    "master/banking_data/test.csv"
)


def download_file (url: str, output_path: Path) -> None:
    """Download a file from a URL to a local path."""

    if output_path.exists():
        print(f"File already exists at {output_path}. Skipping download.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    urlretrieve(url, output_path)

    print(f'saved to {output_path}')

def main():
    project_root = Path(__file__).resolve().parents[3]

    raw_data_dir = project_root / "data" / "raw"

    download_file(TRAIN_URL, raw_data_dir/"train.csv")
    download_file(TEST_URL, raw_data_dir/'test.csv')
if __name__ == '__main__':
    main()
