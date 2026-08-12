import logging
import os
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request

from ..config import ModelSpec
from ..download import ModelPaths

log = logging.getLogger("simple_local.llm")

RESTART_BACKOFF_MAX = 30.0
# What llama.cpp pins n_batch/n_ubatch to when embeddings are enabled and no
# explicit sizes are given. An embedding input longer than one ubatch has hung
# and crashed llama-server, so it doubles as the request limit.
EMBEDDING_DEFAULT_UBATCH = 512
_UBATCH_FLAGS = {"-ub", "--ubatch-size"}


def embedding_token_limit(spec: ModelSpec) -> int:
    inf = spec.inference
    if inf.max_input_tokens:
        return inf.max_input_tokens
    if inf.ubatch_size:
        return inf.ubatch_size
    for flag, value in zip(inf.extra_args, inf.extra_args[1:]):
        if flag in _UBATCH_FLAGS and value.isdigit():
            return int(value)
    return EMBEDDING_DEFAULT_UBATCH


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def build_llama_args(spec: ModelSpec, paths: ModelPaths, port: int) -> list[str]:
    inf = spec.inference
    args = [
        inf.llama_server_path,
        "--model", str(paths.model),
        "--alias", spec.name,
        "--ctx-size", str(inf.context_length),
        "--threads", str(inf.threads or os.cpu_count()),
        "--host", "127.0.0.1",
        "--port", str(port),
        "--metrics",
    ]
    if spec.embeddings:
        args += ["--embeddings"]
    if inf.batch_size:
        args += ["-b", str(inf.batch_size)]
    if inf.ubatch_size:
        args += ["-ub", str(inf.ubatch_size)]
    args += ["--parallel", str(inf.parallel)]
    if inf.mlock:
        args += ["--mlock"]
    if inf.no_mmap:
        args += ["--no-mmap"]
    if inf.cache_type_k:
        args += ["--cache-type-k", inf.cache_type_k]
    if inf.cache_type_v:
        args += ["--cache-type-v", inf.cache_type_v]
    if spec.chat_template:
        args += ["--chat-template", spec.chat_template]
    if spec.chat_template_file:
        args += ["--jinja", "--chat-template-file", spec.chat_template_file]
    for adapter in spec.adapters:
        args += ["--lora", str(paths.adapters[adapter.name])]
    if paths.draft:
        args += [
            "--model-draft", str(paths.draft),
            "--draft-max", str(inf.draft.max_tokens),
            "--draft-min", str(inf.draft.min_tokens),
        ]
    args += inf.extra_args
    return args


class LLMRuntime:
    """
    Supervises a llama.cpp `llama-server` subprocess on a private port.

    The HTTP layer reverse-proxies to `base_url`. A monitor thread restarts the
    subprocess with backoff if it dies; `ready` is cleared while it is down so
    the server can answer 503 instead of proxying into a dead process.
    """

    def __init__(self, spec: ModelSpec, paths: ModelPaths):
        self.spec = spec
        self.paths = paths
        self.port = _free_port()
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.adapter_ids = {a.name: i for i, a in enumerate(spec.adapters)}
        self.token_limit = embedding_token_limit(spec) if spec.embeddings else None
        self.ready = threading.Event()
        self._stopped = threading.Event()
        self._proc = self._spawn()
        self._wait_healthy()
        self.ready.set()
        self._log_slot_context()
        self._log_chat_template()
        threading.Thread(
            target=self._monitor, name=f"llm-monitor-{spec.name}", daemon=True
        ).start()

    def endpoint(self, path: str) -> str:
        return f"{self.base_url}/v1/{path}"

    def upstream_headers(self) -> dict[str, str]:
        return {}

    def _spawn(self) -> subprocess.Popen:
        return subprocess.Popen(build_llama_args(self.spec, self.paths, self.port))

    def _wait_healthy(self, timeout: float = 180.0) -> None:
        deadline = time.monotonic() + timeout
        log.info("%s: loading model %s", self.spec.name, self.paths.model.name)
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"{self.spec.name}: llama-server exited early (code {self._proc.returncode})"
                )
            try:
                with urllib.request.urlopen(f"{self.base_url}/health", timeout=2) as r:
                    if r.status == 200:
                        log.info("%s: ready on %s", self.spec.name, self.base_url)
                        return
            except (urllib.error.URLError, ConnectionError, OSError):
                pass
            time.sleep(0.5)
        self._proc.kill()
        raise TimeoutError(f"{self.spec.name}: llama-server did not become healthy in time")

    def _log_slot_context(self) -> None:
        parallel = self.spec.inference.parallel
        if parallel > 1:
            total = self.spec.inference.context_length
            log.info(
                "%s: context_length %d is the total across %d slots — %d tokens per request",
                self.spec.name, total, parallel, total // parallel,
            )

    def _log_chat_template(self) -> None:
        if self.spec.chat_template:
            log.info("%s: chat template: built-in '%s'", self.spec.name, self.spec.chat_template)
        elif self.spec.chat_template_file:
            log.info("%s: chat template: file %s (jinja)", self.spec.name, self.spec.chat_template_file)
        else:
            log.info(
                "%s: chat template: model default — verify it matches your fine-tune's training format",
                self.spec.name,
            )

    def _monitor(self) -> None:
        backoff = 1.0
        while True:
            self._proc.wait()
            if self._stopped.is_set():
                return
            self.ready.clear()
            log.warning(
                "%s: llama-server exited (code %s); restarting in %.0fs",
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
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
