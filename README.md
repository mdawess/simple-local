# Simple Local

A very bare bones local inference setup that runs on CPU.

One YAML config, one endpoint that can serve 2 kinds of models:

- **`llm`**: GGUF language models via llama.cpp, exposed as an OpenAI-compatible chat endpoint (streaming supported). Pulled from Hugging Face.
- **`predictor`**: scikit-learn / xgboost / lightgbm models saved as `.joblib`, exposed as a simple `/predict` endpoint. Loaded from a local file.

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

The config file is the interface: model, source, and server settings all live there.

## Running

```bash
make serve      # Will download the model if not already on first run
```
Point at a different config with `-c path/to/config.yml`.

## Usage - LLM (`kind: llm`)

```bash
uv pip install openai
```

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["SIMPLE_LOCAL_API_KEY"],
    base_url="http://localhost:8080/environments/development/sync/v1",
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

## Usage - predictor (`kind: predictor`)

Send rows of features to `/predict`. Two input modes, chosen by whether you
define a `schema` in the config:

**Raw mode** (no schema), numeric feature vectors:

```bash
curl http://localhost:8080/environments/development/sync/v1/predict \
  -H "Authorization: Bearer $SIMPLE_LOCAL_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"inputs": [[30, 50000, 1], [45, 80000, 0]]}'
# {"predictions": [0, 1], "probabilities": [[0.8, 0.2], [0.3, 0.7]]}
```

**Typed mode** (schema in config): named, validated objects. The `features`
order in the config defines the vector order, and a pydantic model is built from
it, so bad input returns a 422:

```bash
curl http://localhost:8080/environments/development/sync/v1/predict \
  -H "Authorization: Bearer $SIMPLE_LOCAL_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"inputs": [{"age": 30, "income": 50000, "region": 1}]}'
```

`probabilities` is included when `predictor.task: classification` and the model
supports `predict_proba`.
