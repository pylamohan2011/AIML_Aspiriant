from __future__ import annotations

import argparse
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi
from requests.exceptions import HTTPError


DATASET_SLUG = "mohanpyla/titanic"
COMPETITION_SLUG = "titanic"
OUTPUT_FILE = "train.csv"


def download_from_dataset(api: KaggleApi, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    api.dataset_download_file(
        dataset=DATASET_SLUG,
        file_name=OUTPUT_FILE,
        path=str(output_dir),
        force=True,
    )
    return output_dir / OUTPUT_FILE


def download_from_competition(api: KaggleApi, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    api.competition_download_file(
        competition=COMPETITION_SLUG,
        file_name=OUTPUT_FILE,
        path=str(output_dir),
        force=True,
    )
    return output_dir / OUTPUT_FILE


def download_titanic_data(output_dir: Path) -> Path:
    api = KaggleApi()
    api.authenticate()
    try:
        print(f"Trying private dataset: {DATASET_SLUG}")
        return download_from_dataset(api, output_dir)
    except HTTPError as exc:
        print(f"Private dataset download failed: {exc}")
        print("Falling back to the public Kaggle Titanic competition dataset.")
        return download_from_competition(api, output_dir)
    except Exception as exc:
        print(f"Kaggle dataset error: {exc}")
        print("Falling back to the public Kaggle Titanic competition dataset.")
        return download_from_competition(api, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the Kaggle Titanic dataset to the local Titanic/data folder."
    )
    parser.add_argument(
        "--output",
        default="data",
        help="Local directory to save train.csv",
    )
    args = parser.parse_args()

    output_path = download_titanic_data(Path(args.output))
    print(f"Downloaded Titanic dataset to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
