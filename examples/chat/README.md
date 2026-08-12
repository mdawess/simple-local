# Chat example

Talks to a running `simple-local` LLM server using the OpenAI client.

## Run

Start the server in one terminal:

```bash
cp config.llm.example.yml config.yml
export SIMPLE_LOCAL_API_KEY=$(openssl rand -hex 16)
uv run simple-local serve
```

Then, in another terminal (same `SIMPLE_LOCAL_API_KEY` exported):

```bash
uv run --with openai examples/chat/chat.py
```
