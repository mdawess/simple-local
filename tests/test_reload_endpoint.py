import pytest
from fastapi.testclient import TestClient

from simple_local import server as server_mod
from simple_local.registry import Registry
from simple_local.server import create_app

from conftest import FakeLLM, llm_spec, make_config, make_entry


class Env:
    def __init__(self, client, app, runtimes, built, write_config):
        self.client = client
        self.app = app
        self.runtimes = runtimes
        self.built = built
        self.write_config = write_config


@pytest.fixture
def env(tmp_path, monkeypatch):
    for name in ("base", "other", "added"):
        (tmp_path / f"{name}.gguf").touch()
    config_path = tmp_path / "config.yml"

    def write_config(names):
        config_path.write_text(
            "models:\n"
            + "".join(
                f"  - name: {n}\n    source: {{ provider: local, file: {tmp_path}/{n}.gguf }}\n"
                for n in names
            )
        )

    write_config(["base", "other"])
    specs = {
        n: llm_spec(n, source={"provider": "local", "file": str(tmp_path / f"{n}.gguf")})
        for n in ("base", "other")
    }
    runtimes = {n: FakeLLM() for n in specs}
    entries = {n: make_entry(specs[n], runtimes[n], tmp_path) for n in specs}
    registry = Registry(entries)

    built = []

    def fake_build_entry(spec):
        built.append(spec.name)
        return make_entry(spec, FakeLLM(), tmp_path)

    monkeypatch.setattr(server_mod, "build_entry", fake_build_entry)
    app = create_app(make_config(*specs.values()), registry, config_path=str(config_path))
    return Env(TestClient(app), app, runtimes, built, write_config)


def test_targeted_reload_swaps_only_that_model(env):
    resp = env.client.post("/v1/reload", json={"model": "base"})
    assert resp.status_code == 200
    assert resp.json() == {"reloaded": ["base"], "kept": ["other"], "errors": {}}
    assert env.built == ["base"]
    assert env.runtimes["base"].stopped
    assert not env.runtimes["other"].stopped


def test_full_reload_picks_up_config_changes(env):
    env.write_config(["base", "added"])
    resp = env.client.post("/v1/reload")
    assert resp.status_code == 200
    body = resp.json()
    assert sorted(body["reloaded"]) == ["added", "base"]
    assert set(env.app.state.registry.entries) == {"base", "added"}
    assert env.runtimes["other"].stopped  # removed from config


def test_unknown_model_404(env):
    assert env.client.post("/v1/reload", json={"model": "nope"}).status_code == 404
    assert env.built == []
