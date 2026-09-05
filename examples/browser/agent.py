import json

from browser import Browser
from openai import OpenAI

SYSTEM_PROMPT = """You are a research agent driving a real web browser to answer the user's request.

Work in small steps:
- Use `search_web` to find relevant pages, then `open_url` to read the promising ones.
- Read each tool result before deciding the next action.
- Ground your answer in what you actually saw. If a page had nothing useful, search again.
- When you can answer, call `finish` with a concise answer. Do not guess."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web and return the top results (title, url, snippet).",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "the search query"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_url",
            "description": "Open a URL in the browser and return the page title and visible text.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string", "description": "the URL to open"}},
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Return the final answer to the user and stop.",
            "parameters": {
                "type": "object",
                "properties": {"answer": {"type": "string", "description": "the final answer"}},
                "required": ["answer"],
            },
        },
    },
]


def run_agent(
    client: OpenAI,
    model: str,
    browser: Browser,
    task: str,
    max_steps: int = 8,
    verbose: bool = True,
) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]
    for step in range(max_steps):
        response = client.chat.completions.create(
            model=model, messages=messages, tools=TOOLS, temperature=0.2
        )
        message = response.choices[0].message

        if not message.tool_calls:
            return message.content or ""

        messages.append(
            {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.function.name, "arguments": call.function.arguments},
                    }
                    for call in message.tool_calls
                ],
            }
        )

        for call in message.tool_calls:
            name = call.function.name
            args = _parse_args(call.function.arguments)
            if verbose:
                print(f"[step {step + 1}] {name}({', '.join(f'{k}={v!r}' for k, v in args.items())})")
            if name == "finish":
                return args.get("answer", "")
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": _dispatch(browser, name, args)}
            )

    return "Stopped: reached the step limit without a final answer."


def _parse_args(raw: str) -> dict:
    try:
        return json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}


def _dispatch(browser: Browser, name: str, args: dict) -> str:
    try:
        if name == "search_web":
            return browser.search_web(args["query"])
        if name == "open_url":
            return browser.open_url(args["url"])
        return f"Unknown tool: {name}"
    except Exception as e:
        return f"Tool error: {e}"
