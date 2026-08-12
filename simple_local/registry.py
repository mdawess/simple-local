import logging
import os
import threading
from dataclasses import dataclass, field

from .config import Adapter, Config, ModelSpec
from .download import ModelPaths, ensure_model_files
from .runtimes.custom import CustomRuntime
from .runtimes.llm import LLMRuntime
from .runtimes.predictor import PredictorRuntime
from .runtimes.remote import RemoteRuntime

log = logging.getLogger("simple_local.registry")

# Serializes registry rebuilds between the --watch poller and the reload endpoint.
REBUILD_LOCK = threading.Lock()


def artifact_signature(paths: ModelPaths) -> dict[str, int | None]:
    sig: dict[str, int | None] = {}
    for path in paths.all_paths():
        try:
            sig[str(path)] = os.stat(path).st_mtime_ns
        except OSError:
            sig[str(path)] = None
    return sig


@dataclass
class ModelEntry:
    spec: ModelSpec
    paths: ModelPaths
    runtime: LLMRuntime | PredictorRuntime | CustomRuntime
    artifact_sig: dict[str, int | None] = field(default_factory=dict)


@dataclass
class ChatTarget:
    runtime: LLMRuntime
    adapter: Adapter | None = None

    def lora_payload(self) -> list[dict] | None:
        """Per-request adapter selection for llama-server; None when the runtime
        has no adapters loaded and nothing needs injecting. Adapters absent from
        the list run at scale 0, so [] serves the plain base model."""
        if not self.runtime.adapter_ids:
            return None
        if self.adapter is None:
            return []
        return [
            {
                "id": self.runtime.adapter_ids[self.adapter.name],
                "scale": self.adapter.scale,
            }
        ]


class Registry:
    def __init__(self, entries: dict[str, ModelEntry]):
        self.entries = entries
        self.chat_targets: dict[str, ChatTarget] = {}
        self.embedding_targets: dict[str, ChatTarget] = {}
        self.predictors: dict[str, PredictorRuntime] = {}
        for entry in entries.values():
            if entry.spec.kind in ("llm", "remote"):
                targets = (
                    self.embedding_targets if entry.spec.embeddings else self.chat_targets
                )
                targets[entry.spec.name] = ChatTarget(entry.runtime)
                for adapter in entry.spec.adapters:
                    targets[adapter.name] = ChatTarget(entry.runtime, adapter)
            else:
                self.predictors[entry.spec.name] = entry.runtime

    def model_cards(self) -> list[dict]:
        cards = []
        for entry in self.entries.values():
            cards.append(
                _card(
                    entry.spec.name,
                    entry.spec.kind,
                    version=entry.paths.version,
                    embeddings=entry.spec.embeddings,
                )
            )
            for adapter in entry.spec.adapters:
                cards.append(
                    _card(adapter.name, "llm", base=entry.spec.name, embeddings=entry.spec.embeddings)
                )
        return cards

    def watched_paths(self) -> list[str]:
        return [str(p) for entry in self.entries.values() for p in entry.paths.all_paths()]

    def stop_all(self) -> None:
        for entry in self.entries.values():
            if hasattr(entry.runtime, "stop"):
                entry.runtime.stop()


def _card(
    name: str,
    kind: str,
    base: str | None = None,
    version: str | None = None,
    embeddings: bool = False,
) -> dict:
    card = {"id": name, "object": "model", "owned_by": "simple-local", "kind": kind}
    if base:
        card["base_model"] = base
    if version:
        card["version"] = version
    if embeddings:
        card["embeddings"] = True
    return card


def build_entry(spec: ModelSpec) -> ModelEntry:
    if spec.kind == "remote":
        runtime = RemoteRuntime(spec)
        return ModelEntry(spec, ModelPaths(model=None), runtime, {})
    paths = ensure_model_files(spec)
    if spec.kind == "llm":
        runtime = LLMRuntime(spec, paths)
    elif spec.kind == "predictor":
        runtime = PredictorRuntime(spec, paths.model)
    else:
        runtime = CustomRuntime(spec, paths)
    return ModelEntry(spec, paths, runtime, artifact_signature(paths))


def build_registry(cfg: Config) -> Registry:
    entries: dict[str, ModelEntry] = {}
    try:
        for spec in cfg.models:
            entries[spec.name] = build_entry(spec)
    except BaseException:
        for entry in entries.values():
            if hasattr(entry.runtime, "stop"):
                entry.runtime.stop()
        raise
    return Registry(entries)
