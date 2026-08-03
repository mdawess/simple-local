# Custom runtime example

A miniature inference-shaped ensemble behind `kind: custom` — the escape hatch for
serving arbitrary Python inference (ensembles, business rules, rich outputs)
through the standard server: same routing, auth, health, reload, and logging.

The contract (`simple_local.runtimes.custom.Runtime`):

```python
class MyRuntime:
    def load(self, ctx: ModelContext) -> None: ...   # ctx.path, ctx.config, ctx.version
    def predict(self, request: dict) -> dict | Iterator[dict]: ...
```

Return a dict for a single result, or a generator to stream batch rows as
NDJSON. Raise `ValueError` for bad input → the server answers 422.

## Run

```bash
uv run simple-local serve -c examples/custom/config.yml
```

Single prediction (arbitrary JSON in, rich JSON out):

```bash
curl -s localhost:8081/v1/predict -H "Authorization: Bearer $SIMPLE_LOCAL_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"company": "CAHP", "kva": 75, "phase": "3ph"}'
# {"predicted_cost": ..., "std_dev": ..., "confidence": "high", "method_estimates": {...}, ...}
```

Streamed batch (one NDJSON row per spec, as they complete):

```bash
curl -sN localhost:8081/v1/predict -H "Authorization: Bearer $SIMPLE_LOCAL_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"specs": [{"kva": 75, "phase": "3ph"}, {"kva": 150, "company": "USHP"}]}'
```

Force a re-resolve of remote sources (e.g. after uploading new artifacts or
moving the `active.json` pointer) without restarting:

```bash
curl -s -X POST localhost:8081/v1/reload -H "Authorization: Bearer $SIMPLE_LOCAL_API_KEY" \
  -H "Content-Type: application/json" -d '{"model": "cost-ensemble"}'
```
