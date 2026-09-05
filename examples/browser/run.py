import argparse
import os

from agent import run_agent
from browser import Browser
from openai import OpenAI

DEFAULT_TASK = (
    "Search the web for what the Python 'with' statement does, "
    "then give me a one-sentence explanation."
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Local-LLM agent that drives a real browser.")
    parser.add_argument("task", nargs="?", default=DEFAULT_TASK, help="what to research")
    parser.add_argument("--headed", action="store_true", help="show the browser window instead of running headless")
    parser.add_argument("--model", default="Qwen-2.5-3B", help="model name served by simple-local")
    parser.add_argument("--max-steps", type=int, default=8, help="max agent turns before giving up")
    args = parser.parse_args()

    client = OpenAI(
        api_key=os.environ.get("SIMPLE_LOCAL_API_KEY", "not-needed"),
        base_url="http://localhost:8081/environments/development/sync/v1",
    )

    print(f"Task: {args.task}\n")
    with Browser(headless=not args.headed) as browser:
        answer = run_agent(client, args.model, browser, args.task, max_steps=args.max_steps)

    print("\n=== Answer ===")
    print(answer)


if __name__ == "__main__":
    main()
