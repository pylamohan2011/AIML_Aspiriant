from __future__ import annotations

from app import demo


def main() -> None:
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)


if __name__ == "__main__":
    main()
