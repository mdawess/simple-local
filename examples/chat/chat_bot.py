import os

from openai import OpenAI

SIMPLE_LOCAL_API_KEY = os.environ.get("SIMPLE_LOCAL_API_KEY", "sk_89902u301i2j3o1h324iu234gb5r")

client = OpenAI(
    api_key=SIMPLE_LOCAL_API_KEY,
    base_url="http://localhost:8085/environments/development/sync/v1",
)

while True:
    user_input = input("> ")

    response = client.chat.completions.create(
        model="Qwen3.6-27B",
        messages=[{"role": "user", "content": user_input}],
        stream=True,
    )

    for chunk in response:
        print(chunk.choices[0].delta.content or "", end="", flush=True)
    print()
