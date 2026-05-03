from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import HfApi, upload_folder

BASE_DIR = Path(__file__).resolve().parent


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("Set HF_TOKEN environment variable to your Hugging Face write token.")

    repo_id = os.environ.get("HF_SPACE_ID", "USERNAME/titanic-survival-prediction")
    print(f"Using Hugging Face Space repo: {repo_id}")

    api = HfApi()
    api.create_repo(
        repo_id=repo_id,
        token=token,
        repo_type="space",
        space_sdk="gradio",
        exist_ok=True,
    )

    print("Uploading Titanic project files to Hugging Face Spaces...")
    upload_folder(
        folder_path=str(BASE_DIR),
        path_in_repo="",
        repo_id=repo_id,
        repo_type="space",
        token=token,
    )
    print("Upload complete. Your Space should rebuild automatically.")


if __name__ == "__main__":
    main()
