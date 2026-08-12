import os

from openai import OpenAI

client = OpenAI(
    api_key=os.environ["SIMPLE_LOCAL_API_KEY"],
    base_url="http://localhost:8081/environments/development/sync/v1",
)

response = client.chat.completions.create(
    model="Qwen-2.5-3B",
    messages=[
        {"role": "user", "content": "In one sentence, what is machine learning?"}
    ],
)
print(response.choices[0].message.content)
