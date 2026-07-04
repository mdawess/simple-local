import os
import sys
import threading

from . import config as config_mod
from .download import ensure_model
from .runtimes.llm import LLMRuntime
from .runtimes.predictor import PredictorRuntime


def build_runtime(cfg, model_path):
    if cfg.kind == "llm":
        return LLMRuntime(cfg, model_path)
    return PredictorRuntime(cfg, model_path)


class ReloadWatcher:
    """Polls the config file and the model file; on change, rebuilds the runtime
    and swaps it onto app.state atomically. A broken new artifact is caught and
    the current model keeps serving — a bad reload never takes the server down.

    Predictor only: swapping a live llama-server is a heavier blue/green step.
    """

    def __init__(self, app, config_path: str, kind: str, model_path, interval: float = 1.0):
        self.app = app
        self.config_path = config_path
        self.kind = kind
        self.model_path = str(model_path)
        self.interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="reload-watcher", daemon=True)
        self._signature = self._current_signature()

    def _current_signature(self) -> dict:
        sig = {}
        for path in (self.config_path, self.model_path):
            try:
                sig[path] = os.stat(path).st_mtime_ns
            except OSError:
                sig[path] = None
        return sig

    def start(self) -> None:
        if self.kind != "predictor":
            print("[watch] hot-reload is predictor-only; restart to change an llm model", file=sys.stderr)
            return
        print(f"[watch] watching {self.config_path} and {self.model_path}", file=sys.stderr)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2)

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            sig = self._current_signature()
            if sig != self._signature:
                self._signature = sig
                self._reload()

    def _reload(self) -> None:
        try:
            cfg = config_mod.load(self.config_path)
            if cfg.kind != self.kind:
                print(f"[watch] kind changed {self.kind}->{cfg.kind}; restart to apply (keeping current)", file=sys.stderr)
                return
            model_path = ensure_model(cfg)
            new_runtime = build_runtime(cfg, model_path)
        except Exception as e:
            print(f"[watch] reload failed, keeping current model: {e}", file=sys.stderr)
            return

        old_runtime = self.app.state.runtime
        self.app.state.runtime = new_runtime  # atomic swap
        self.model_path = str(model_path)
        self._signature = self._current_signature()
        if hasattr(old_runtime, "stop"):
            old_runtime.stop()
        print(f"[watch] reloaded {cfg.model_name} from {model_path}", file=sys.stderr)
