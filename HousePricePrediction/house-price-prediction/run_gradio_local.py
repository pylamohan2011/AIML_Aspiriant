from __future__ import annotations

import time

from app import demo


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, prevent_thread_lock=True)
    while True:
        time.sleep(60)
