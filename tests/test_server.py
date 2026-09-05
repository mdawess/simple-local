import json

import httpx
import joblib
import numpy as np
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from simple_local.registry import ModelEntry, Registry, artifact_signature
from simple_local.runtimes.llm import EMBEDDING_DEFAULT_UBATCH, embedding_token_limit
from simple_local.download import ModelPaths
from simple_local.runtimes.predictor import PredictorRuntime
from simple_local.server import create_app

from conftest import FakeLLM, FakeSink, llm_spec, make_config


def echo_upstream(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/v1/chat/completions":
        body = json.loads(request.content)
        if body.get("stream"):

            async def sse():
                yield b'data: {"choices":[]}\n\n'
                yield b'data: {"usage":{"prompt_tokens":3,"completion_tokens":5}}\n\n'
                yield b"data: [DONE]\n\n"

            return httpx.Response(
                200, content=sse(), headers={"content-type": "text/event-stream"}
            )
        return httpx.Response(
            200,
            json={
                "echo": body,
                "host": request.url.host,
                "usage": {"prompt_tokens": 3, "completion_tokens": 5},
            },
        )
    if request.url.path == "/v1/embeddings":
        body = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
                "host": request.url.host,
                "echo": body,
                "usage": {"prompt_tokens": 2, "total_tokens": 2},
            },
        )
    if request.url.path == "/tokenize":
        content = json.loads(request.content)["content"]
        # one token per 2 characters, so byte length always over-estimates
        return httpx.Response(200, json={"tokens": list(range(len(content) // 2))})
    if request.url.path == "/metrics":
        return httpx.Response(200, text="llamacpp:prompt_tokens_total 42\n")
    return httpx.Response(404)


def make_client(cfg, registry) -> TestClient:
    app = create_app(cfg, registry)
    app.state.client = httpx.AsyncClient(transport=httpx.MockTransport(echo_upstream))
    return TestClient(app)


def two_model_client(registry_factory, api_key=""):
    base = llm_spec(
        "base",
        adapters=[
            {"name": "tuned", "source": {"provider": "local", "file": "a.gguf"}, "scale": 0.7}
        ],
    )
    other = llm_spec("other")
    base_rt = FakeLLM(adapters=["tuned"], base_url="http://base-upstream")
    other_rt = FakeLLM(base_url="http://other-upstream")
    registry = registry_factory((base, base_rt), (other, other_rt))
    client = make_client(make_config(base, other, api_key=api_key), registry)
    return client, base_rt, other_rt


def test_models_listing(registry_factory):
    client, *_ = two_model_client(registry_factory)
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    ids = {m["id"]: m for m in resp.json()["data"]}
    assert set(ids) == {"base", "tuned", "other"}
    assert ids["tuned"]["base_model"] == "base"


def test_routing_and_lora_injection(registry_factory):
    client, *_ = two_model_client(registry_factory)

    resp = client.post("/v1/chat/completions", json={"model": "base", "messages": []})
    assert resp.json()["host"] == "base-upstream"
    assert resp.json()["echo"]["lora"] == []  # adapters loaded but disabled

    resp = client.post("/v1/chat/completions", json={"model": "tuned", "messages": []})
    assert resp.json()["host"] == "base-upstream"
    assert resp.json()["echo"]["lora"] == [{"id": 0, "scale": 0.7}]

    resp = client.post("/v1/chat/completions", json={"model": "other", "messages": []})
    assert resp.json()["host"] == "other-upstream"
    assert "lora" not in resp.json()["echo"]  # no adapters on this runtime


def test_unknown_and_missing_model(registry_factory):
    client, *_ = two_model_client(registry_factory)
    assert client.post("/v1/chat/completions", json={"model": "nope"}).status_code == 404
    assert client.post("/v1/chat/completions", json={}).status_code == 422


def test_missing_model_defaults_when_single(registry_factory):
    spec = llm_spec("solo")
    registry = registry_factory((spec, FakeLLM()))
    client = make_client(make_config(spec), registry)
    assert client.post("/v1/chat/completions", json={"messages": []}).status_code == 200


def test_restarting_model_returns_503(registry_factory):
    client, base_rt, _ = two_model_client(registry_factory)
    base_rt.ready.clear()
    resp = client.post("/v1/chat/completions", json={"model": "base"})
    assert resp.status_code == 503
    assert client.post("/v1/chat/completions", json={"model": "other"}).status_code == 200


def test_auth(registry_factory):
    client, *_ = two_model_client(registry_factory, api_key="k3y")
    assert client.get("/v1/models").status_code == 401
    assert client.get("/v1/models", headers={"Authorization": "Bearer k3y"}).status_code == 200


def test_legacy_env_prefix_still_works(registry_factory):
    client, *_ = two_model_client(registry_factory)
    resp = client.post(
        "/environments/development/sync/v1/chat/completions",
        json={"model": "base", "messages": []},
    )
    assert resp.status_code == 200
    assert client.get("/environments/development/sync/v1/models").status_code == 200


def test_streaming_relayed(registry_factory):
    client, *_ = two_model_client(registry_factory)
    resp = client.post("/v1/chat/completions", json={"model": "base", "stream": True})
    assert resp.status_code == 200
    assert resp.text.endswith("data: [DONE]\n\n")


def test_health_reflects_runtime_state(registry_factory):
    client, base_rt, _ = two_model_client(registry_factory)
    assert client.get("/health").status_code == 200
    base_rt.ready.clear()
    resp = client.get("/health")
    assert resp.status_code == 503
    assert resp.json()["models"]["base"] == "restarting"


def test_metrics_requires_model_when_ambiguous(registry_factory):
    client, *_ = two_model_client(registry_factory)
    assert client.get("/v1/metrics").status_code == 422
    resp = client.get("/v1/metrics", params={"model": "base"})
    assert resp.status_code == 200
    assert "prompt_tokens_total" in resp.text


def test_db_sink_records_calls_and_errors(registry_factory):
    client, *_ = two_model_client(registry_factory)
    sink = FakeSink()
    client.app.state.db_sink = sink

    client.post("/v1/chat/completions", json={"model": "base", "messages": [{"role": "user", "content": "hi"}]})
    client.post("/v1/chat/completions", json={"model": "nope"})
    client.post("/v1/chat/completions", json={"model": "base", "stream": True}).text

    ok, missing, streamed = sink.records
    assert (ok["endpoint"], ok["model"], ok["status"]) == ("chat", "base", 200)
    assert (ok["prompt_tokens"], ok["completion_tokens"]) == (3, 5)
    assert '"content": "hi"' in ok["request"]
    assert '"echo"' in ok["response"]
    assert ok["duration_ms"] >= 0 and ok["error"] is None

    assert missing["status"] == 404
    assert "unknown chat model 'nope'" in missing["error"]
    assert missing["response"] is None

    assert streamed["status"] == 200
    assert (streamed["prompt_tokens"], streamed["completion_tokens"]) == (3, 5)
    assert "[DONE]" in streamed["response"]


def test_db_sink_records_auth_failure(registry_factory):
    client, *_ = two_model_client(registry_factory, api_key="k3y")
    sink = FakeSink()
    client.app.state.db_sink = sink
    client.get("/v1/models")
    assert sink.records[0]["status"] == 401
    assert sink.records[0]["endpoint"] == "/v1/models"
    assert "invalid api key" in sink.records[0]["error"]


def test_embeddings_routing(registry_factory):
    chat = llm_spec("chat")
    embed = llm_spec("embedder", embeddings=True)
    registry = registry_factory(
        (chat, FakeLLM(base_url="http://chat-upstream")),
        (embed, FakeLLM(base_url="http://embed-upstream")),
    )
    client = make_client(make_config(chat, embed), registry)

    # single embedding model: no 'model' needed even with a chat model present
    resp = client.post("/v1/embeddings", json={"input": "hello"})
    assert resp.status_code == 200
    assert resp.json()["host"] == "embed-upstream"
    assert resp.json()["data"][0]["embedding"] == [0.1, 0.2]

    # the two serving modes don't cross
    assert client.post("/v1/chat/completions", json={"model": "embedder"}).status_code == 404
    assert client.post("/v1/embeddings", json={"model": "chat", "input": "x"}).status_code == 404

    cards = {m["id"]: m for m in client.get("/v1/models").json()["data"]}
    assert cards["embedder"]["embeddings"] is True
    assert "embeddings" not in cards["chat"]


def embedding_client(registry_factory, **inference):
    spec = llm_spec("embedder", embeddings=True, inference=inference or {"ubatch_size": 100})
    runtime = FakeLLM(base_url="http://embed-upstream")
    runtime.token_limit = embedding_token_limit(spec)
    registry = registry_factory((spec, runtime))
    return make_client(make_config(spec), registry), runtime


def test_oversized_embedding_input_rejected(registry_factory):
    client, runtime = embedding_client(registry_factory)
    assert runtime.token_limit == 100

    # 60 bytes -> under the limit by byte count alone, never tokenized
    assert client.post("/v1/embeddings", json={"input": ["x" * 60]}).status_code == 200
    # 150 bytes -> 75 tokens by the fake tokenizer, still fine
    assert client.post("/v1/embeddings", json={"input": ["y" * 150]}).status_code == 200

    # 300 bytes -> 150 tokens, over the limit
    resp = client.post("/v1/embeddings", json={"input": ["ok", "z" * 300]})
    assert resp.status_code == 413
    assert "input[1] is 150 tokens" in resp.json()["detail"]
    assert "at most 100" in resp.json()["detail"]


def test_embedding_limit_falls_back_to_extra_args(registry_factory):
    assert embedding_token_limit(llm_spec("e", embeddings=True)) == EMBEDDING_DEFAULT_UBATCH
    from_extra = llm_spec(
        "e", embeddings=True, inference={"extra_args": ["--pooling", "last", "-ub", "2048"]}
    )
    assert embedding_token_limit(from_extra) == 2048
    explicit = llm_spec(
        "e", embeddings=True, inference={"ubatch_size": 2048, "max_input_tokens": 900}
    )
    assert embedding_token_limit(explicit) == 900


def make_predictor_entry(tmp_path):
    model = LogisticRegression().fit(np.array([[0.0], [1.0]]), np.array([0, 1]))
    path = tmp_path / "m.joblib"
    joblib.dump(model, path)
    spec = llm_spec("churn", kind="predictor", predictor={"task": "classification"})
    runtime = PredictorRuntime(spec, path)
    paths = ModelPaths(model=path)
    return ModelEntry(spec, paths, runtime, artifact_signature(paths))


def test_predict(tmp_path):
    entry = make_predictor_entry(tmp_path)
    client = make_client(make_config(entry.spec), Registry({"churn": entry}))
    sink = FakeSink()
    client.app.state.db_sink = sink
    resp = client.post("/v1/predict", json={"inputs": [[0.0], [1.0]]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["predictions"] == [0, 1]
    assert len(body["probabilities"]) == 2
    assert client.post("/v1/predict", json={"inputs": "nope"}).status_code == 422
    assert client.post("/v1/predict", json={"model": "nope", "inputs": []}).status_code == 404

    ok, bad_inputs, bad_model = sink.records
    assert (ok["endpoint"], ok["model"], ok["status"]) == ("predict", "churn", 200)
    assert '"predictions"' in ok["response"]
    assert bad_inputs["status"] == 422 and "inputs" in bad_inputs["error"]
    assert bad_model["status"] == 404
