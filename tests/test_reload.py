from types import SimpleNamespace

from simple_local import reload as reload_mod
from simple_local.registry import ModelEntry, Registry, artifact_signature

from conftest import FakeLLM, llm_spec, make_entry


BASE_YAML = """
models:
  - name: base
    source: {{ provider: local, file: {tmp}/base.gguf }}
    inference: {{ context_length: {ctx} }}
  - name: other
    source: {{ provider: local, file: {tmp}/other.gguf }}
"""


def setup(tmp_path, monkeypatch):
    (tmp_path / "base.gguf").touch()
    (tmp_path / "other.gguf").touch()
    config_path = tmp_path / "config.yml"
    config_path.write_text(BASE_YAML.format(tmp=tmp_path, ctx=2048))

    specs = {
        "base": llm_spec("base", source={"provider": "local", "file": str(tmp_path / "base.gguf")}, inference={"context_length": 2048}),
        "other": llm_spec("other", source={"provider": "local", "file": str(tmp_path / "other.gguf")}),
    }
    runtimes = {name: FakeLLM() for name in specs}
    entries = {name: make_entry(spec, runtimes[name], tmp_path) for name, spec in specs.items()}
    app = SimpleNamespace(state=SimpleNamespace(registry=Registry(entries)))

    built = []

    def fake_build_entry(spec):
        built.append(spec.name)
        entry = make_entry(spec, FakeLLM(), tmp_path)
        return entry

    monkeypatch.setattr(reload_mod, "build_entry", fake_build_entry)
    watcher = reload_mod.ReloadWatcher(app, str(config_path))
    return app, watcher, config_path, runtimes, built


def test_only_changed_model_rebuilds(tmp_path, monkeypatch):
    app, watcher, config_path, runtimes, built = setup(tmp_path, monkeypatch)
    config_path.write_text(BASE_YAML.format(tmp=tmp_path, ctx=4096))
    watcher._reload()

    assert built == ["base"]
    assert runtimes["base"].stopped
    assert not runtimes["other"].stopped
    reg = app.state.registry
    assert reg.entries["base"].spec.inference.context_length == 4096
    assert reg.entries["other"].runtime is runtimes["other"]


def test_artifact_touch_rebuilds(tmp_path, monkeypatch):
    app, watcher, config_path, runtimes, built = setup(tmp_path, monkeypatch)
    entry = app.state.registry.entries["base"]
    entry.paths.model.write_bytes(b"new weights")
    entry.artifact_sig["stale"] = None  # force signature mismatch regardless of mtime resolution
    watcher._reload()
    assert built == ["base"]


def test_broken_reload_keeps_current(tmp_path, monkeypatch):
    app, watcher, config_path, runtimes, built = setup(tmp_path, monkeypatch)

    def boom(spec):
        raise RuntimeError("bad artifact")

    monkeypatch.setattr(reload_mod, "build_entry", boom)
    config_path.write_text(BASE_YAML.format(tmp=tmp_path, ctx=9999))
    watcher._reload()

    reg = app.state.registry
    assert reg.entries["base"].runtime is runtimes["base"]
    assert not runtimes["base"].stopped
    assert reg.entries["base"].spec.inference.context_length == 2048


def test_removed_model_is_stopped(tmp_path, monkeypatch):
    app, watcher, config_path, runtimes, built = setup(tmp_path, monkeypatch)
    config_path.write_text(
        f"""
models:
  - name: base
    source: {{ provider: local, file: {tmp_path}/base.gguf }}
    inference: {{ context_length: 2048 }}
"""
    )
    watcher._reload()
    assert "other" not in app.state.registry.entries
    assert runtimes["other"].stopped
    assert not runtimes["base"].stopped
