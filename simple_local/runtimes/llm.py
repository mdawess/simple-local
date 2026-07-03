import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from ..config import Config


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class LLMRuntime:
    """Supervises a llama.cpp `llama-server` subprocess on a private port.

    The HTTP layer reverse-proxies to `base_url`; llama-server is already
    OpenAI-compatible and handles streaming, so nothing else is needed here.
    """

    def __init__(self, cfg: Config, model_path: Path):
        self.port = _free_port()
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.proc = subprocess.Popen(
            [
                cfg.inference.llama_server_path,
                "--model",
                str(model_path),
                "--ctx-size",
                str(cfg.inference.context_length),
                "--threads",
                str(cfg.inference.threads or os.cpu_count()),
                "--host",
                "127.0.0.1",
                "--port",
                str(self.port),
            ]
        )
        self._wait_healthy()

    def _wait_healthy(self, timeout: float = 120.0) -> None:
        deadline = time.monotonic() + timeout
        print("Loading model", end="", flush=True)
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"llama-server exited early (code {self.proc.returncode})"
                )
            try:
                with urllib.request.urlopen(f"{self.base_url}/health", timeout=2) as r:
                    if r.status == 200:
                        print(" ready")
                        return
            except (urllib.error.URLError, ConnectionError, OSError):
                pass
            print(".", end="", flush=True)
            time.sleep(0.5)
        self.stop()
        raise TimeoutError("llama-server did not become healthy in time")

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
