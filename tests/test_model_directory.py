"""A model directory is the unit of deployment, so everything a config points at
resolves relative to that config — not to whatever directory you happen to be in."""

from pathlib import Path

import pytest

from simple_local import config as config_mod
from simple_local.registry import build_registry
from simple_local.runtimes.custom import resolve_runtime_class

REPO = Path(__file__).resolve().parent.parent

# examples/ also holds settings files for the examples themselves, which are not
# server configs at all — a models: key is what distinguishes them.
CONFIGS = sorted(
    path
    for pattern in ("implementations/*/config*.yml", "examples/*/config.yml")
    for path in REPO.glob(pattern)
    if "models:" in path.read_text()
)


@pytest.mark.parametrize("config", CONFIGS, ids=lambda p: str(p.relative_to(REPO)))
def test_every_config_in_the_repo_loads(config):
    assert config_mod.load(str(config)).models


@pytest.mark.parametrize("config", CONFIGS, ids=lambda p: str(p.relative_to(REPO)))
def test_runtime_paths_resolve_from_anywhere(config, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    for spec in config_mod.load(str(config)).models:
        if spec.runtime:
            assert Path(spec.runtime.rsplit(":", 1)[0]).exists()


@pytest.mark.parametrize("config", CONFIGS, ids=lambda p: str(p.relative_to(REPO)))
def test_local_sources_resolve_from_anywhere(config, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    for spec in config_mod.load(str(config)).models:
        if spec.source and spec.source.provider == "local":
            assert Path(spec.source.file).is_absolute()


def test_a_custom_model_serves_from_a_different_working_directory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    cfg = config_mod.load(str(REPO / "examples/custom/config.yml"))
    registry = build_registry(cfg)
    entry = registry.entries["cost-ensemble"]
    assert entry.runtime.predict({"kva": 100, "phase": "1ph"})


def test_a_directory_can_be_moved_wholesale(tmp_path):
    source = REPO / "examples/custom"
    moved = tmp_path / "elsewhere"
    moved.mkdir()
    for name in ("config.yml", "runtime.py"):
        (moved / name).write_text((source / name).read_text())

    cfg = config_mod.load(str(moved / "config.yml"))
    cls = resolve_runtime_class(cfg.models[0].runtime)
    assert cls.__name__ == "DemoEnsemble"
    assert str(moved) in cfg.models[0].runtime
