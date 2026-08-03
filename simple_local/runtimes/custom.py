import importlib
import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Protocol, runtime_checkable

from ..config import ModelSpec
from ..download import ModelPaths


@dataclass
class ModelContext:
    """What a runtime gets handed at load time."""

    name: str
    path: Path | None  # artifact file or directory from the source; None if no source
    config: dict = field(default_factory=dict)  # the spec's opaque `config:` block
    version: str | None = None  # resolved artifact version (s3 sources)


@runtime_checkable
class Runtime(Protocol):
    def load(self, ctx: ModelContext) -> None: ...

    def predict(self, request: dict) -> dict | Iterator[dict]: ...


def resolve_runtime_class(ref: str) -> type:
    module_ref, sep, class_name = ref.rpartition(":")
    if not sep:
        raise ValueError(
            f"runtime must look like 'module:Class' or 'path/to/file.py:Class', got '{ref}'"
        )
    if module_ref.endswith(".py") or "/" in module_ref:
        path = Path(module_ref).expanduser().resolve()
        module_spec = importlib.util.spec_from_file_location(path.stem, path)
        if module_spec is None or module_spec.loader is None:
            raise ValueError(f"cannot load runtime module from {path}")
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_ref)
    return getattr(module, class_name)


class CustomRuntime:
    """Hosts a user-supplied Runtime implementation: arbitrary JSON in,
    arbitrary JSON out, or an iterator of JSON rows for streamed batches."""

    def __init__(self, spec: ModelSpec, paths: ModelPaths):
        cls = resolve_runtime_class(spec.runtime)
        self.impl = cls()
        self.impl.load(
            ModelContext(
                name=spec.name, path=paths.model, config=spec.config, version=paths.version
            )
        )

    def predict(self, request: dict) -> Any:
        return self.impl.predict(request)
