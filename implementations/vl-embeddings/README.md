# Multimodal embeddings (text + images)

`Qwen3-VL-Embedding-2B` served through a `kind: custom` runtime since it isn't openai-compatible, putting text and images in one 2048-dim space so you can retrieve images with text queries.

## Setup

Needs the optional torch stack (~2.5GB), plus a 4GB model download on first run:

```bash
uv sync --extra vl
make serve CONFIG=implementations/vl-embeddings/config.yml
```

## Usage

`POST /v1/predict` — each `input` item is a string, `{"image": ...}`,
`{"text": ...}`, or both. Images may be an http(s) URL, a local path, or a
`data:` URL:

```bash
curl -s http://localhost:8084/v1/predict \
  -H "Authorization: Bearer $SIMPLE_LOCAL_API_KEY" -H "Content-Type: application/json" \
  -d '{"input": ["a golden retriever on a beach", {"image": "/tmp/dog.jpg"}]}'
```

```python
import json, urllib.request

def embed(items, **opts):
    body = {"input": items, **opts}
    req = urllib.request.Request(
        "http://localhost:8084/v1/predict", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    return [e["embedding"] for e in json.load(urllib.request.urlopen(req))["data"]]

photo = embed([{"image": "receipt.png"}])[0]
query = embed(["an invoice with a total over $500"])[0]   # compare with cosine
```
