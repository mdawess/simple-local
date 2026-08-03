import threading
from pathlib import Path

import pytest

from simple_local.config import Config, ModelSpec
from simple_local.download import ModelPaths
from simple_local.registry import ModelEntry, Registry, artifact_signature


class FakeLLM:
    def __init__(self, adapters=(), base_url="http://upstream"):
        self.base_url = base_url
        self.adapter_ids = {name: i for i, name in enumerate(adapters)}
        self.ready = threading.Event()
        self.ready.set()
        self.stopped = False

    def stop(self):
        self.stopped = True


def llm_spec(name="base", **overrides) -> ModelSpec:
    data = {"name": name, "source": {"provider": "local", "file": f"{name}.gguf"}}
    data.update(overrides)
    return ModelSpec.model_validate(data)


def make_entry(spec: ModelSpec, runtime, tmp_path: Path) -> ModelEntry:
    model_file = tmp_path / Path(spec.source.file).name
    model_file.touch()
    adapters = {}
    for adapter in spec.adapters:
        path = tmp_path / Path(adapter.source.file).name
        path.touch()
        adapters[adapter.name] = path
    paths = ModelPaths(model=model_file, adapters=adapters)
    return ModelEntry(spec, paths, runtime, artifact_signature(paths))


@pytest.fixture
def registry_factory(tmp_path):
    def factory(*spec_runtime_pairs):
        entries = {
            spec.name: make_entry(spec, runtime, tmp_path)
            for spec, runtime in spec_runtime_pairs
        }
        return Registry(entries)

    return factory


def make_config(*specs, api_key="") -> Config:
    return Config.model_validate(
        {
            "models": [s.model_dump(by_alias=True) for s in specs],
            "server": {"api_key": api_key},
        }
    )
