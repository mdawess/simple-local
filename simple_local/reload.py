import logging
import os
import threading

from . import config as config_mod
from .registry import REBUILD_LOCK, ModelEntry, Registry, artifact_signature, build_entry

log = logging.getLogger("simple_local.reload")


def _mtimes(paths: list[str]) -> dict[str, int | None]:
    sig: dict[str, int | None] = {}
    for path in paths:
        try:
            sig[path] = os.stat(path).st_mtime_ns
        except OSError:
            sig[path] = None
    return sig


class ReloadWatcher:
    """Polls the config file and every model artifact; on change, rebuilds only
    the affected models and swaps a new registry onto app.state atomically.

    LLM swaps are blue/green: the replacement llama-server is booted and
    health-checked before the old one is stopped, so both are briefly resident.
    A broken new artifact or config is caught and the current model keeps
    serving — a bad reload never takes the server down.
    """

    def __init__(self, app, config_path: str, interval: float = 1.0):
        self.app = app
        self.config_path = config_path
        self.interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="reload-watcher", daemon=True)
        self._signature = self._current_signature()

    def _current_signature(self) -> dict:
        return _mtimes([self.config_path, *self.app.state.registry.watched_paths()])

    def start(self) -> None:
        log.info("watching %s and %d model artifacts", self.config_path, len(self._signature) - 1)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2)

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            sig = self._current_signature()
            if sig != self._signature:
                self._reload()
                self._signature = self._current_signature()

    def _reload(self) -> None:
        try:
            cfg = config_mod.load(self.config_path)
        except Exception as e:
            log.warning("config reload failed, keeping current models: %s", e)
            return

        with REBUILD_LOCK:
            self._apply(cfg)

    def _apply(self, cfg) -> None:
        old: dict[str, ModelEntry] = self.app.state.registry.entries
        new_entries: dict[str, ModelEntry] = {}
        for spec in cfg.models:
            prev = old.get(spec.name)
            if prev is not None and not _changed(prev, spec):
                new_entries[spec.name] = prev
                continue
            try:
                new_entries[spec.name] = build_entry(spec)
                log.info("reloaded %s", spec.name)
            except Exception as e:
                log.warning("reload of %s failed: %s", spec.name, e)
                if prev is not None:
                    log.warning("keeping current %s", spec.name)
                    new_entries[spec.name] = prev

        self.app.state.registry = Registry(new_entries)  # atomic swap
        for name, entry in old.items():
            if new_entries.get(name) is not entry and hasattr(entry.runtime, "stop"):
                entry.runtime.stop()
                log.info("stopped old runtime for %s", name)


def _changed(entry: ModelEntry, spec) -> bool:
    return entry.spec != spec or artifact_signature(entry.paths) != entry.artifact_sig
