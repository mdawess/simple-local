import json

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from simple_local import config as config_mod
from simple_local.config import ModelSpec
from simple_local.registry import ModelEntry, Registry, build_entry
from simple_local.download import ModelPaths
from simple_local.server import create_app

from conftest import FakeSink, llm_spec, make_config

REMOTE_URL = "https://acme--llm.modal.run/v1"


def upstream(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content) if request.content else {}
    return httpx.Response(
        200,
        json={
            "url": str(request.url),
            "auth": request.headers.get("authorization"),
            "model": body.get("model"),
            "usage": {"prompt_tokens": 7, "completion_tokens": 11},
        },
    )


def remote_spec(name="Qwen3.6-27B", **remote) -> ModelSpec:
    settings = {"url": REMOTE_URL, "api_key": "modal-token"}
    settings.update(remote)
    return ModelSpec.model_validate({"name": name, "kind": "remote", "remote": settings})


def client_for(*specs) -> TestClient:
    entries = {
        spec.name: ModelEntry(spec, ModelPaths(model=None), build_entry(spec).runtime, {})
        for spec in specs
    }
    app = create_app(make_config(*specs), Registry(entries))
    app.state.client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    return TestClient(app)


def test_chat_forwards_with_upstream_auth():
    client = client_for(remote_spec())
    body = client.post(
        "/v1/chat/completions", json={"model": "Qwen3.6-27B", "messages": []}
    ).json()
    assert body["url"] == f"{REMOTE_URL}/chat/completions"
    assert body["auth"] == "Bearer modal-token"
    assert body["model"] == "Qwen3.6-27B"


def test_upstream_model_rename():
    client = client_for(remote_spec(model="qwen3.6-27b-awq"))
    body = client.post("/v1/chat/completions", json={"model": "Qwen3.6-27B"}).json()
    assert body["model"] == "qwen3.6-27b-awq"  # our name in, their name out


def test_remote_embeddings_route_separately():
    client = client_for(
        remote_spec("remote-embedder", url="https://acme--emb.modal.run/v1"),
    )
    # a remote without embeddings: true is a chat model
    assert client.post("/v1/chat/completions", json={"model": "remote-embedder"}).status_code == 200

    spec = ModelSpec.model_validate(
        {
            "name": "remote-embedder",
            "kind": "remote",
            "embeddings": True,
            "remote": {"url": "https://acme--emb.modal.run/v1"},
        }
    )
    client = client_for(spec)
    body = client.post("/v1/embeddings", json={"input": ["hi"]}).json()
    assert body["url"] == "https://acme--emb.modal.run/v1/embeddings"
    assert body["auth"] is None  # no api_key configured -> no header
    assert client.post("/v1/chat/completions", json={"model": "remote-embedder"}).status_code == 404


def test_local_and_remote_mix_in_one_registry(registry_factory, tmp_path):
    from conftest import FakeLLM, make_entry

    local = llm_spec("local-embedder", embeddings=True)
    remote = remote_spec()
    entries = {
        "local-embedder": make_entry(local, FakeLLM(base_url="http://local"), tmp_path),
        remote.name: ModelEntry(remote, ModelPaths(model=None), build_entry(remote).runtime, {}),
    }
    app = create_app(make_config(local, remote), Registry(entries))
    app.state.client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    client = TestClient(app)

    cards = {m["id"]: m for m in client.get("/v1/models").json()["data"]}
    assert cards["Qwen3.6-27B"]["kind"] == "remote"
    assert cards["local-embedder"]["kind"] == "llm"

    health = client.get("/health")
    assert health.status_code == 200  # a remote must not make the server look degraded
    assert health.json()["models"] == {"local-embedder": "ready", "Qwen3.6-27B": "remote"}


def test_unreachable_remote_is_503():
    def dead(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    spec = remote_spec()
    entries = {spec.name: ModelEntry(spec, ModelPaths(model=None), build_entry(spec).runtime, {})}
    app = create_app(make_config(spec), Registry(entries))
    app.state.client = httpx.AsyncClient(transport=httpx.MockTransport(dead))
    app.state.db_sink = FakeSink()
    client = TestClient(app)

    resp = client.post("/v1/chat/completions", json={"model": "Qwen3.6-27B"})
    assert resp.status_code == 503
    assert app.state.db_sink.records[0]["status"] == 503


def test_modal_303_is_followed():
    """Modal answers 303 after 150s and expects the client to follow it."""
    hops = []

    def redirecting(request: httpx.Request) -> httpx.Response:
        hops.append(str(request.url))
        if "__modal_result" not in str(request.url):
            return httpx.Response(
                303, headers={"location": f"{REMOTE_URL}/chat/completions?__modal_result=1"}
            )
        return httpx.Response(200, json={"done": True, "hops": len(hops)})

    spec = remote_spec()
    entries = {spec.name: ModelEntry(spec, ModelPaths(model=None), build_entry(spec).runtime, {})}
    app = create_app(make_config(spec), Registry(entries))
    app.state.client = httpx.AsyncClient(transport=httpx.MockTransport(redirecting))
    client = TestClient(app)

    resp = client.post("/v1/chat/completions", json={"model": "Qwen3.6-27B", "messages": []})
    assert resp.status_code == 200
    assert resp.json()["done"] is True
    assert len(hops) == 2  # original, then the result URL


def test_local_upstream_does_not_follow_redirects(registry_factory, tmp_path):
    from conftest import FakeLLM, make_entry

    spec = llm_spec("local")
    registry = registry_factory((spec, FakeLLM(base_url="http://local")))
    app = create_app(make_config(spec), registry)
    app.state.client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(303, headers={"location": "http://elsewhere/v1/chat"})
        )
    )
    resp = TestClient(app).post("/v1/chat/completions", json={"model": "local"})
    assert resp.status_code == 303  # a local llama-server has no reason to redirect


def test_remote_requires_url_and_block(tmp_path):
    with pytest.raises(ValidationError, match="remote: block is required"):
        ModelSpec.model_validate({"name": "r", "kind": "remote"})
    with pytest.raises(ValidationError, match="http"):
        ModelSpec.model_validate(
            {"name": "r", "kind": "remote", "remote": {"url": "acme.modal.run"}}
        )
    with pytest.raises(ValidationError, match="only supported for kind: remote"):
        ModelSpec.model_validate(
            {
                "name": "r",
                "source": {"provider": "local", "file": "m.gguf"},
                "remote": {"url": "https://x/v1"},
            }
        )


def test_local_only_keys_rejected_on_remote():
    for key, value in [
        ("inference", {"context_length": 8192}),
        ("source", {"provider": "local", "file": "m.gguf"}),
        ("chat_template", "chatml"),
    ]:
        with pytest.raises(ValidationError, match="cannot apply to kind: remote"):
            ModelSpec.model_validate(
                {"name": "r", "kind": "remote", "remote": {"url": REMOTE_URL}, key: value}
            )


def test_remote_needs_no_source(tmp_path):
    cfg = config_mod.load(
        str(
            _write(
                tmp_path,
                """
models:
  - name: big
    kind: remote
    remote:
      url: https://acme--llm.modal.run/v1/
      api_key: ${MODAL_KEY}
""",
            )
        )
    )
    assert cfg.models[0].remote.url == "https://acme--llm.modal.run/v1"  # trailing slash trimmed
    assert cfg.models[0].source is None


def _write(tmp_path, text):
    path = tmp_path / "config.yml"
    path.write_text(text)
    return path
