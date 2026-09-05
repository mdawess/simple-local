import json
import os
import urllib.error
import urllib.request

URL = os.environ.get(
    "PREDICT_URL",
    "http://localhost:8081/environments/development/sync/v1/predict",
)

# Typed inputs — keys and order match the schema in config.yml.
ROWS = [
    {"sqft": 1500, "bedrooms": 3, "age": 20},
    {"sqft": 3200, "bedrooms": 5, "age": 5},
    {"sqft": 800, "bedrooms": 1, "age": 80},
]


def main() -> None:
    request = urllib.request.Request(
        URL,
        data=json.dumps({"inputs": ROWS}).encode(),
        headers={"Content-Type": "application/json"},
    )
    key = os.environ.get("SIMPLE_LOCAL_API_KEY")
    if key:
        request.add_header("Authorization", f"Bearer {key}")

    try:
        with urllib.request.urlopen(request) as response:
            result = json.load(response)
    except urllib.error.URLError as e:
        raise SystemExit(f"predict request failed ({e}). Is the predictor served on :8081?")

    for row, prediction in zip(ROWS, result["predictions"]):
        print(f"{row}  ->  ${prediction:,.0f}")


if __name__ == "__main__":
    main()
