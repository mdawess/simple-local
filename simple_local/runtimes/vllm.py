import logging
import subprocess
import threading

from ..config import ModelSpec
from ..download import ModelPaths
from .llm import RESTART_BACKOFF_MAX, _free_port, _wait_for_health

log = logging.getLogger("simple_local.vllm")

# vLLM loads weights, profiles the GPU and allocates its KV cache before it
# answers anything, which takes far longer than llama.cpp's mmap.
READY_TIMEOUT = 900.0


def build_vllm_args(spec: ModelSpec, paths: ModelPaths, port: int) -> list[str]:
    """vLLM serves its own OpenAI-compatible API, so the proxy in front of it is
    the same one that fronts llama-server. Only the arguments differ."""
    settings = spec.vllm
    model = settings.model or (str(paths.model) if paths.model else None)
    if not model:
        raise ValueError(f"model '{spec.name}': vllm needs a model id or a source to download")

    args = [
        settings.executable, "serve", model,
        "--served-model-name", spec.name,
        "--host", "127.0.0.1",
        "--port", str(port),
        "--gpu-memory-utilization", str(settings.gpu_memory_utilization),
        "--tensor-parallel-size", str(settings.tensor_parallel_size),
    ]
    if settings.max_model_len:
        args += ["--max-model-len", str(settings.max_model_len)]
    if settings.max_num_seqs:
        args += ["--max-num-seqs", str(settings.max_num_seqs)]
    if settings.dtype:
        args += ["--dtype", settings.dtype]
    if spec.embeddings:
        # Pooling and normalisation are what decide whether these vectors are
        # comparable with another runtime's — never leave them to a default.
        args += ["--task", "embed"]
        if settings.pooling:
            args += ["--override-pooler-config", f'{{"pooling_type": "{settings.pooling.upper()}"}}']
    args += settings.extra_args
    return args


class VLLMRuntime:
    """Supervises a `vllm serve` subprocess on a private port, with the same
    restart-with-backoff behaviour as the llama.cpp runtime."""

    def __init__(self, spec: ModelSpec, paths: ModelPaths):
        self.spec = spec
        self.paths = paths
        self.port = _free_port()
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.adapter_ids: dict[str, int] = {}
        self.token_limit = spec.vllm.max_model_len if spec.embeddings else None
        self.ready = threading.Event()
        self._stopped = threading.Event()
        self._proc = self._spawn()
        self._wait_healthy()
        self.ready.set()
        threading.Thread(
            target=self._monitor, name=f"vllm-monitor-{spec.name}", daemon=True
        ).start()

    def endpoint(self, path: str) -> str:
        return f"{self.base_url}/v1/{path}"

    def upstream_headers(self) -> dict[str, str]:
        return {}

    def _spawn(self) -> subprocess.Popen:
        args = build_vllm_args(self.spec, self.paths, self.port)
        log.info("%s: starting %s", self.spec.name, " ".join(args))
        return subprocess.Popen(args)

    def _wait_healthy(self) -> None:
        _wait_for_health(
            self._proc, self.base_url, self.spec.name, "vllm", timeout=READY_TIMEOUT
        )

    def _monitor(self) -> None:
        backoff = 1.0
        while True:
            self._proc.wait()
            if self._stopped.is_set():
                return
            self.ready.clear()
            log.warning(
                "%s: vllm exited (code %s); restarting in %.0fs",
                self.spec.name, self._proc.returncode, backoff,
            )
            if self._stopped.wait(backoff):
                return
            try:
                self._proc = self._spawn()
                self._wait_healthy()
                self.ready.set()
                backoff = 1.0
                log.info("%s: recovered", self.spec.name)
            except Exception as e:
                log.warning("%s: restart failed: %s", self.spec.name, e)
                backoff = min(backoff * 2, RESTART_BACKOFF_MAX)

    def stop(self) -> None:
        self._stopped.set()
        self.ready.clear()
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._proc.kill()
