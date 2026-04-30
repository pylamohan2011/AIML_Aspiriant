from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent


def main() -> None:
    subprocess.Popen(
        [sys.executable, "app.py"],
        cwd=PROJECT_DIR,
        stdout=(PROJECT_DIR / "gradio.host.out.log").open("w", encoding="utf-8"),
        stderr=(PROJECT_DIR / "gradio.host.err.log").open("w", encoding="utf-8"),
    )
    print("Gradio app starting. Open http://127.0.0.1:7860")


if __name__ == "__main__":
    main()
