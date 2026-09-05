"""Load a config's models in-process, without an HTTP server in front.

For scripts that would otherwise stand up a server and immediately call it —
evaluations, batch indexing, notebooks:

    from simple_local import Models

    with Models.from_config("config.yml") as models:
        vectors = models.embed("qwen3-embed", ["a query", "another"])

The runtimes behind the two model kinds differ in a way this cannot hide.
`predictor` and `custom` run in this process, so `predict()` is a direct call.
`llm` and `vllm` supervise a subprocess that speaks HTTP on a private port, so
those go over localhost — same weights, same settings, but a socket in the
middle. Both are exposed rather than papered over, because a caller measuring
latency needs to know which one they have.
"""

from __future__ import annotations

from pathlib import Path
from types import TracebackType

import httpx

from .config import Config, load
from .registry import ModelEntry, Registry, build_registry

# Kinds whose runtime is a supervised subprocess reached over HTTP.
SERVED_KINDS = ("llm", "vllm")
# Kinds whose runtime is a Python object in this process.
IN_PROCESS_KINDS = ("predictor", "custom")


class Models:
    """The models one config describes, loaded and ready to call.

    Loading starts subprocesses and reads weights, so it is not cheap — build
    one of these and reuse it. `close()` stops everything; the context manager
    is the reliable way to be sure it happens.
    """

    def __init__(self, registry: Registry, config: Config, timeout: float = 300.0):
        self.registry = registry
        self.config = config
        self._client = httpx.Client(timeout=timeout)

    @classmethod
    def from_config(cls, path: str | Path, **kwargs) -> "Models":
        config = load(str(path))
        return cls(build_registry(config), config, **kwargs)

    @classmethod
    def from_object(cls, config: Config, **kwargs) -> "Models":
        return cls(build_registry(config), config, **kwargs)

    def __enter__(self) -> "Models":
        return self

    def __exit__(self, exc_type, exc: BaseException | None, tb: TracebackType | None) -> None:
        self.close()

    def __getitem__(self, name: str) -> ModelEntry:
        entry = self.registry.entries.get(name)
        if entry is None:
            raise KeyError(f"no model '{name}' — this config serves {', '.join(self.names) or 'nothing'}")
        return entry

    def __contains__(self, name: str) -> bool:
        return name in self.registry.entries

    def __iter__(self):
        return iter(self.registry.entries)

    def __len__(self) -> int:
        return len(self.registry.entries)

    @property
    def names(self) -> list[str]:
        return list(self.registry.entries)

    def kind(self, name: str) -> str:
        return self[name].spec.kind

    def endpoint(self, name: str, path: str = "") -> str:
        """Base URL of the subprocess serving this model. Only `llm` and `vllm`
        have one; the others run here and have no socket to point at."""
        entry = self[name]
        if entry.spec.kind not in SERVED_KINDS:
            raise ValueError(
                f"'{name}' is kind: {entry.spec.kind}, which runs in this process — "
                "call predict() instead of asking for an endpoint"
            )
        return entry.runtime.endpoint(path) if path else entry.runtime.base_url

    def predict(self, name: str, payload: dict):
        """Direct call into a `predictor` or `custom` runtime."""
        entry = self[name]
        if entry.spec.kind not in IN_PROCESS_KINDS:
            raise ValueError(
                f"'{name}' is kind: {entry.spec.kind}, which serves over HTTP — "
                "use embed(), chat() or endpoint()"
            )
        return entry.runtime.predict(payload)

    def embed(self, name: str, inputs: str | list[str], **options) -> list[list[float]]:
        """Vectors for one or more inputs.

        Works for both shapes: an `llm`/`vllm` model configured with
        `embeddings: true` goes over its local socket, and a `custom` runtime is
        called directly. Ordering follows the input, not whatever order the
        upstream happened to answer in.
        """
        entry = self[name]
        texts = [inputs] if isinstance(inputs, str) else list(inputs)

        if entry.spec.kind in IN_PROCESS_KINDS:
            result = entry.runtime.predict({"input": texts, **options})
            rows = result["data"] if isinstance(result, dict) and "data" in result else result
            return [row["embedding"] for row in sorted(rows, key=lambda r: r.get("index", 0))]

        if not entry.spec.embeddings:
            raise ValueError(f"'{name}' is not configured with embeddings: true")
        response = self._client.post(
            entry.runtime.endpoint("embeddings"),
            json={"model": name, "input": texts, **options},
            headers=entry.runtime.upstream_headers(),
        )
        response.raise_for_status()
        rows = response.json()["data"]
        return [row["embedding"] for row in sorted(rows, key=lambda r: r.get("index", 0))]

    def chat(self, name: str, messages: list[dict], **options) -> dict:
        """One chat completion, straight from the upstream — no streaming, and
        the raw OpenAI-shaped body rather than a convenience wrapper."""
        entry = self[name]
        if entry.spec.kind not in SERVED_KINDS:
            raise ValueError(f"'{name}' is kind: {entry.spec.kind}, which does not serve chat")
        response = self._client.post(
            entry.runtime.endpoint("chat/completions"),
            json={"model": name, "messages": messages, **options},
            headers=entry.runtime.upstream_headers(),
        )
        response.raise_for_status()
        return response.json()

    def cards(self) -> list[dict]:
        """What `/v1/models` would report."""
        return self.registry.model_cards()

    def close(self) -> None:
        self._client.close()
        self.registry.stop_all()
