# Remote LLM on Modal

Moving heavier model runs to gpu.

| file | role |
| --- | --- |
| `deploy.py` | the Modal app: vLLM behind an OpenAI-compatible endpoint |
| `config.yml` | a `kind: remote` model pointing simple-local at it |

## Deploy

```bash
uv run modal secret create simple-local-proxy MODAL_PROXY_KEY=$(openssl rand -hex 16)

uv run modal run implementations/llm-remote/deploy.py::prefetch

uv run modal deploy implementations/llm-remote/deploy.py
```

Put the same key in `.env` as `MODAL_PROXY_KEY`, then:

```bash
make serve CONFIG=implementations/llm-remote/config.yml
curl -s http://<host>:8083/v1/chat/completions \
  -H "Authorization: Bearer $SIMPLE_LOCAL_API_KEY" -H "Content-Type: application/json" \
  -d '{"model": "Qwen3.6-27B", "messages": [{"role": "user", "content": "hi"}]}'
```
