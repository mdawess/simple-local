"""simple-local: one YAML config, one endpoint, any of several model runtimes.

Three ways to use it, in rising order of how much you want it to do for you:

    # 1. Read and validate a config
    from simple_local import load
    config = load("config.yml")

    # 2. Load the models in this process and call them
    from simple_local import Models
    with Models.from_config("config.yml") as models:
        vectors = models.embed("qwen3-embed", ["a query"])

    # 3. Serve them over HTTP, or mount into an app you already have
    from simple_local import create_app
    app = create_app(config, build_registry(config))

Only the config layer is imported eagerly. Everything past it pulls fastapi,
scikit-learn or boto3, which is a poor trade for a caller that only wanted to
read a YAML file — so those resolve on first use instead.
"""

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING

from .config import Adapter, Config, Inference, ModelSpec, Server, Source, load

try:
    __version__ = version("simple-local")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0.dev0"

# name -> (module, attribute). Kept out of the import graph until touched.
_LAZY = {
    "Models": (".models", "Models"),
    "Registry": (".registry", "Registry"),
    "ModelEntry": (".registry", "ModelEntry"),
    "build_registry": (".registry", "build_registry"),
    "build_entry": (".registry", "build_entry"),
    "create_app": (".server", "create_app"),
    "ensure_model_files": (".download", "ensure_model_files"),
    "prefetch": (".download", "prefetch"),
    "ReloadWatcher": (".reload", "ReloadWatcher"),
}

if TYPE_CHECKING:  # so editors and type checkers still see the real symbols
    from .download import ensure_model_files, prefetch
    from .models import Models
    from .registry import ModelEntry, Registry, build_entry, build_registry
    from .reload import ReloadWatcher
    from .server import create_app


def __getattr__(name: str):
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attribute = target
    return getattr(import_module(module, __name__), attribute)


def __dir__() -> list[str]:
    return sorted(__all__)


__all__ = [
    "Adapter",
    "Config",
    "Inference",
    "ModelEntry",
    "ModelSpec",
    "Models",
    "Registry",
    "ReloadWatcher",
    "Server",
    "Source",
    "__version__",
    "build_entry",
    "build_registry",
    "create_app",
    "ensure_model_files",
    "load",
    "prefetch",
]
