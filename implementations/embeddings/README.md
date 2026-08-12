# Embeddings + chat

Two models served by one `simple-local` server, both llama.cpp processes behind
the same OpenAI-compatible API:

| model | role | endpoint |
| --- | --- | --- |
| `Qwen3-Embedding-0.6B` | retrieval (`embeddings: true`) | `/v1/embeddings` |
| `Qwen2.5-3B` | answering | `/v1/chat/completions` |

## Run

```bash
make serve CONFIG=implementations/embeddings/config.yml
```

## Using it

The server binds the Tailscale IP in `config.yml`, so any device on the tailnet
can use it (set `host: localhost` to keep it on this machine).

```python
from openai import OpenAI

client = OpenAI(api_key=os.environ["SIMPLE_LOCAL_API_KEY"],
                base_url="http://<host>:8083/v1")

vectors = client.embeddings.create(model="Qwen3-Embedding-0.6B", input=["hello", "world"])
reply = client.chat.completions.create(model="Qwen2.5-3B",
                                       messages=[{"role": "user", "content": "hi"}])
```
