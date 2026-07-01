# Simple Local

A very bare bones local inference setup that only runs on cpu.

## Credits

This design was inspired by https://github.com/basetenlabs/truss and https://github.com/ollama/ollama

## Usage

```bash
uv pip install openai
```

Create a chat completion
```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["RANDOM_HASH_STRING"],
    base_url="http://localhost:8080/environments/development/sync/v1",
)

response = client.chat.completions.create(
    model="Qwen-2.5-3B",
    messages=[
        {"role": "user", "content": "What is machine learning?"}
    ],
)

print(response.choices[0].message.content)
```
You should see a response like:
```bash
Machine learning is a branch of artificial intelligence where systems learn
patterns from data to make predictions or decisions without being explicitly
programmed for each task...
```
