# Browser use example

Gives the local model a **dedicated browser** to drive. The LLM runs an agent
loop over three tools — `search_web`, `open_url`, `finish` — and a single
long-lived Playwright page executes each action against the real web. This is the
pattern you'd use to let an agent test code changes in a browser; here it just
does a web search.

```
task ──▶ LLM ──tool_call──▶ Playwright page ──result──▶ LLM ──▶ ... ──▶ finish(answer)
```

| File | Role |
|---|---|
| `browser.py` | `Browser` — the persistent Playwright page + the tool actions |
| `agent.py` | tool schemas + the OpenAI tool-calling loop (`run_agent`) |
| `run.py` | entry point — takes a task string, wires up the client and browser |

## Setup

Install the Chromium binary Playwright drives (one time):

```bash
uv run --with playwright playwright install chromium
```

Start the `simple-local` LLM server (from the repo root, in its own terminal):

```bash
cp config.llm.example.yml config.yml
export SIMPLE_LOCAL_API_KEY=$(openssl rand -hex 16)
make serve                                   # serves Qwen-2.5-3B on :8081
```

The served model must support tool calling — Qwen2.5-3B-Instruct (the default
config) does.

## Run

```bash
# from the repo root (same SIMPLE_LOCAL_API_KEY exported)
make run EXAMPLE=browser

# or directly, with your own task
cd examples/browser
uv run --with openai --with playwright python run.py "Search for the latest stable Python version and tell me the number."

# watch it work in a real window
uv run --with openai --with playwright python run.py --headed
```

Each step prints the tool call the model chose, and the grounded answer prints at
the end. Flags: `--headed` (show the window), `--model`, `--max-steps`.

## Notes

- The browser instance persists across the whole run, so the agent can search,
  then open a result, then read it — state carries between tool calls.
- `search_web` scrapes DuckDuckGo's HTML endpoint — a real general web search
  that works from an ordinary machine. Google serves a bot wall to automated
  traffic, and DuckDuckGo does the same for some datacenter/CI IPs; when that
  happens `search_web` falls back to Wikipedia search so the agent always gets
  real results. `_dismiss_consent` best-effort clicks through consent dialogs.
- To point the agent at a local dev server instead of the web, just have it
  `open_url http://localhost:3000` — the same loop works for testing your own UI.
