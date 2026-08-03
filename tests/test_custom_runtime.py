import json

import pytest

from simple_local.config import ModelSpec
from simple_local.registry import Registry, build_entry
from simple_local.server import create_app

from conftest import make_config
from test_server import make_client

RUNTIME_SRC = '''
class Echo:
    def load(self, ctx):
        self.ctx = ctx

    def predict(self, request):
        if request.get("boom"):
            raise ValueError("bad spec")
        if "rows" in request:
            return ({"row": r, "model": self.ctx.name} for r in request["rows"])
        return {
            "echo": request,
            "name": self.ctx.name,
            "config": self.ctx.config,
            "path": str(self.ctx.path) if self.ctx.path else None,
            "version": self.ctx.version,
        }
'''


@pytest.fixture
def custom_client(tmp_path):
    runtime_file = tmp_path / "my_runtime.py"
    runtime_file.write_text(RUNTIME_SRC)
    spec = ModelSpec.model_validate(
        {
            "name": "ensemble",
            "kind": "custom",
            "runtime": f"{runtime_file}:Echo",
            "config": {"weight": 0.7},
        }
    )
    entry = build_entry(spec)
    return make_client(make_config(spec), Registry({"ensemble": entry}))


def test_custom_predict_arbitrary_json(custom_client):
    resp = custom_client.post("/v1/predict", json={"kva": 75, "phase": "3ph"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["echo"] == {"kva": 75, "phase": "3ph"}
    assert body["name"] == "ensemble"
    assert body["config"] == {"weight": 0.7}
    assert body["path"] is None


def test_custom_batch_streams_ndjson(custom_client):
    resp = custom_client.post("/v1/predict", json={"rows": ["a", "b", "c"]})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/x-ndjson")
    rows = [json.loads(line) for line in resp.text.strip().splitlines()]
    assert [r["row"] for r in rows] == ["a", "b", "c"]


def test_custom_value_error_is_422(custom_client):
    assert custom_client.post("/v1/predict", json={"boom": True}).status_code == 422


def test_custom_listed_in_models(custom_client):
    cards = {m["id"]: m for m in custom_client.get("/v1/models").json()["data"]}
    assert cards["ensemble"]["kind"] == "custom"


def test_demo_ensemble_example():
    spec = ModelSpec.model_validate(
        {
            "name": "cost-ensemble",
            "kind": "custom",
            "runtime": "examples/custom/runtime.py:DemoEnsemble",
            "config": {"rate_per_kva": {"default": 40.0}},
        }
    )
    entry = build_entry(spec)
    result = entry.runtime.predict({"kva": 75, "phase": "3ph"})
    assert result["predicted_cost"] > 0
    assert set(result["method_estimates"]) == {"rate_card", "power_curve"}
    streamed = list(entry.runtime.predict({"specs": [{"kva": 10}, {"kva": 20}]}))
    assert [r["index"] for r in streamed] == [0, 1]
