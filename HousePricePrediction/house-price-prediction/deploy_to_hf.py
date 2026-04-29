from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import HfApi, get_token, upload_folder


PROJECT_DIR = Path(__file__).resolve().parent
REPO_ID = "pylamohan2011/MLSpaces"


def main() -> None:
    token = os.getenv("HF_TOKEN") or get_token()
    if not token:
        raise SystemExit(
            "No Hugging Face token found. Create a write token, then run:\n"
            "$env:HF_TOKEN='hf_your_token_here'\n"
            "python deploy_to_hf.py"
        )

    api = HfApi(token=token)
    api.create_repo(repo_id=REPO_ID, repo_type="space", space_sdk="gradio", exist_ok=True)
    upload_folder(
        repo_id=REPO_ID,
        repo_type="space",
        folder_path=PROJECT_DIR,
        token=token,
        ignore_patterns=[
            "__pycache__/",
            "*.pyc",
            ".git/",
            "gradio.out.log",
            "gradio.err.log",
        ],
    )
    print(f"Deployed to https://huggingface.co/spaces/{REPO_ID}")


if __name__ == "__main__":
    main()
