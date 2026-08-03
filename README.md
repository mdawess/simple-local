# Simple Local

A small local inference server for fine-tuned small models, CPU-friendly.

One YAML config, one server, any number of models:

- **`llm`**: GGUF language models via llama.cpp, exposed as an OpenAI-compatible chat endpoint (streaming supported). Serve LoRA fine-tune adapters on top of a shared base model, each under its own model name.
- **`predictor`**: scikit-learn / xgboost / lightgbm models saved as `.joblib`, exposed as a simple `/predict` endpoint.
- **`custom`**: your own Python runtime class behind the same server — arbitrary JSON in/out on `/predict`, streamed NDJSON for batches. See `examples/custom/`.

Model artifacts can come from Hugging Face, local files, or S3 — including a
versioned S3 layout (`{prefix}/{version}/…`) with `version: latest`, an explicit
version, or `active` resolved through a `{prefix}/active.json` pointer for
deploy-free rollbacks. `POST /v1/reload` (optionally `{"model": "name"}`)
re-resolves sources and swaps models blue/green without a restart.

## Credits

Inspired by https://github.com/basetenlabs/truss

## Prerequisites

- llama.cpp should also be installed on your computer
- Optionally, whisper.cpp if adding voice

## Setup

```bash
cp config.llm.example.yml config.yml           # LLM, or use config.predictor.example.yml
export SIMPLE_LOCAL_API_KEY=$(openssl rand -hex 16)
```
Note that the contents of the API key are irrelevant, it is just to comply with
openai's api.

The config file is the interface: models, sources, and server settings all live
there. `models:` is a list — mix llms and predictors in one server and route by
the `model` field in the request.

## Running

```bash
make serve      # Will download the models if not already on first run
```
Point at a different config with `-c path/to/config.yml`. Add `--watch` to
hot-reload models when the config or a model artifact changes — llm swaps are
blue/green (the new llama-server is health-checked before the old one stops),
and a broken artifact or config never takes down the currently serving model.

If a llama-server subprocess crashes, it is restarted automatically with
backoff; requests for that model return 503 until it recovers.

## Usage - LLM (`kind: llm`)

```bash
uv pip install openai
```

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["SIMPLE_LOCAL_API_KEY"],
    base_url="http://localhost:8081/v1",
)

response = client.chat.completions.create(
    model="Qwen-2.5-3B",
    messages=[{"role": "user", "content": "What is machine learning?"}],
)
print(response.choices[0].message.content)
```

The `api_key` must match `server.api_key` in your config. Pass `stream=True` for
token streaming. The OpenAI client always requires *some* key; if you leave
`server.api_key` empty, auth is disabled and any placeholder value works.

The old `/environments/{env}/sync/v1` base path still works as an alias for `/v1`.

### Fine-tune adapters

Point `adapters:` at LoRA GGUFs (convert PEFT output with llama.cpp's
`convert_lora_to_gguf.py`). Each adapter is served under its own model name on
the shared base-model process, so switching between the base and any adapter is
per-request and free:

```yaml
models:
  - name: Qwen-2.5-3B
    source: { provider: huggingface, repo: Qwen/Qwen2.5-3B-Instruct-GGUF, file: qwen2.5-3b-instruct-q4_k_m.gguf }
    adapters:
      - name: Qwen-2.5-3B-sql
        source: { provider: local, file: ~/adapters/sql-lora.gguf }
        scale: 1.0
```

Requests for `Qwen-2.5-3B` run with adapters disabled; requests for
`Qwen-2.5-3B-sql` apply the adapter at its configured scale. `GET /v1/models`
lists everything that's servable.

### Chat templates

Serving-time chat templates must match what the fine-tune was trained with, or
quality quietly degrades. Set `chat_template:` (a built-in llama.cpp template
name) or `chat_template_file:` (a jinja file). The resolved template source is
logged at startup; if neither is set, the model's embedded default is used.

### Embeddings

Set `embeddings: true` on an llm model to serve it on the OpenAI-compatible
`/v1/embeddings` endpoint instead of chat (llama.cpp runs the two modes with
different pooling, so an embedding model is its own `models:` entry and
process). GGUF conversions exist for most sentence-transformer models
(all-MiniLM, bge, nomic-embed, ...):

```yaml
models:
  - name: minilm-l6
    kind: llm
    embeddings: true
    source:
      provider: huggingface
      repo: second-state/All-MiniLM-L6-v2-Embedding-GGUF
      file: all-MiniLM-L6-v2-Q4_K_M.gguf
```

```python
client.embeddings.create(model="minilm-l6", input=["machine learning", "deep learning"])
```

Pass pooling overrides via `extra_args: ["--pooling", "mean"]` if a model needs
them. For a PyTorch sentence-transformers model that has no GGUF, wrap it in a
`kind: custom` runtime instead.

### Performance

Under `inference:`: `parallel` (concurrent slots with continuous batching),
`cache_type_k`/`cache_type_v` (KV-cache quantization), `mlock`, `no_mmap`,
`draft:` (speculative decoding with a small draft model), and `extra_args` for
anything else llama-server accepts (e.g. `["--flash-attn", "on"]`).

### Observability

- `GET /health` — per-model status; 503 while any llama-server is restarting
- `GET /v1/metrics` — llama-server's Prometheus metrics (TTFT, tokens/sec, slot
  usage); pass `?model=` when serving multiple llms
- One log line per request: model, status, duration, prompt/completion tokens

## Usage - predictor (`kind: predictor`)

Send rows of features to `/v1/predict` (include `"model": "<name>"` when more
than one predictor is configured). Two input modes, chosen by whether you
define a `schema` in the config:

**Raw mode** (no schema), numeric feature vectors:

```bash
curl http://localhost:8081/v1/predict \
  -H "Authorization: Bearer $SIMPLE_LOCAL_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"inputs": [[30, 50000, 1], [45, 80000, 0]]}'
# {"predictions": [0, 1], "probabilities": [[0.8, 0.2], [0.3, 0.7]]}
```

**Typed mode** (schema in config): named, validated objects. The `features`
order in the config defines the vector order, and a pydantic model is built from
it, so bad input returns a 422:

```bash
curl http://localhost:8081/v1/predict \
  -H "Authorization: Bearer $SIMPLE_LOCAL_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"inputs": [{"age": 30, "income": 50000, "region": 1}]}'
```

`probabilities` is included when `predictor.task: classification` and the model
supports `predict_proba`.

## Tests

```bash
uv run pytest
```
