# Simple Local

A small local inference server for small models, CPU-friendly. Once llama.cpp is installed, any open source
model can by served via the yml config. Also supports traditional ml models (e.g. regression, classification)
and remote runtimes (e.g. Modal or Azure).

Model artifacts can come from Hugging Face, local files, or S3 (no Azure blob yet) including a versioned S3 layout (`{prefix}/{version}/…`) with `version: latest`, an explicit version, or `active` resolved through a `{prefix}/active.json` pointer for deploy-free rollbacks. `POST /v1/reload` (optionally `{"model": "name"}`)
re-resolves sources and swaps models blue/green without a restart.

## Credits

Inspired by https://github.com/basetenlabs/truss

## Prerequisites

- llama.cpp should also be installed on your computer
- Optionally, whisper.cpp if adding voice

## Setup

```bash
cp examples/chat/config.yml config.yml
cp .env.example .env # then fill in SIMPLE_LOCAL_API_KEY
```
`make serve` and `make run` load `.env` automatically; it also holds optional HF and AWS credentials for huggingface/s3 model sources. Note that the contents of the API key are irrelevant, it is just to comply with openai's api.

## Running

```bash
make serve      # Will download the models if not already on first run
```

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

The `api_key` must match `server.api_key` in your config. Pass `stream=True` for token streaming. The OpenAI client always requires *some* key; if you leave `server.api_key` empty, auth is disabled and any placeholder value works.

### Embeddings

Set `embeddings: true` on an llm model to serve it on the OpenAI-compatible `/v1/embeddings` endpoint instead of chat (llama.cpp runs the two modes with different pooling, so an embedding model is its own `models:` entry and process). GGUF conversions exist for most sentence-transformer models (all-MiniLM, bge, nomic-embed, ...):

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

### Offloading a model (`kind: remote`)

I pushed my macbook to far and didn't like the tok/s so added this less local config. It splits the runtimes
allowing big models to run on a remote GPU while the small, latency-sensitive ones stay local (e.g. embedding models), but all called from the same local endpoint (e.g. `http://localhost:8000/v1`)

```yaml
models:
  - name: Qwen3-Embedding-0.6B     # local: small, hot, instant
    kind: llm
    embeddings: true
    source: { provider: huggingface, repo: Qwen/Qwen3-Embedding-0.6B-GGUF, file: Qwen3-Embedding-0.6B-Q8_0.gguf }

  - name: Qwen3.6-27B              # offloaded: no heat, no battery drain
    kind: remote
    remote:
      url: https://<workspace>--llm.modal.run/v1
      api_key: ${MODAL_PROXY_KEY}   # bearer token sent upstream
      # model: qwen3.6-27b-awq      # if the upstream name differs
      timeout: 300
```

Add `embeddings: true` to route a remote to `/v1/embeddings` instead of chat.
`/health` reports remotes as `remote` (reachability is proven per request, not
by a supervisor) and an unreachable one returns 503 rather than hanging.

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
