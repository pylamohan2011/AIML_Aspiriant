from __future__ import annotations

import time


def main() -> None:
    print("Importing Gradio app...", flush=True)
    from app import demo

    print("Starting Gradio on http://127.0.0.1:7861", flush=True)
    demo.launch(server_name="127.0.0.1", server_port=7861, prevent_thread_lock=True)
    print("Gradio server is running.", flush=True)
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
