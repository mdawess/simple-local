# Regression example

Serves a scikit-learn regression model (`.joblib`) through simple-local's
**predictor** endpoint (`kind: predictor`) and calls it over HTTP. Predicts a
house price from `sqft`, `bedrooms`, and `age`.

This is the non-LLM side of simple-local: same YAML config, same one-endpoint
shape, but `/predict` instead of `/chat/completions`.

## Run

```bash
# 1. Train the model (uses the project env — scikit-learn is already a dep)
uv run python examples/regression/train.py        # writes model.joblib

# 2. Serve the predictor (from the repo root)
make serve CONFIG=examples/regression/config.yml   # port 8081

# 3. Call it (another terminal)
make run EXAMPLE=regression
```

Expected output:

```
{'sqft': 1500, 'bedrooms': 3, 'age': 20}  ->  $2xx,xxx
{'sqft': 3200, 'bedrooms': 5, 'age': 5}   ->  $5xx,xxx
{'sqft': 800, 'bedrooms': 1, 'age': 80}   ->  $1xx,xxx
```

## Input modes

`config.yml` defines a `schema`, so inputs are **named objects**:

```json
{"inputs": [{"sqft": 1500, "bedrooms": 3, "age": 20}]}
```

Remove the `schema` block to use **raw mode** — plain feature vectors in schema
order:

```json
{"inputs": [[1500, 3, 20]]}
```

For classification models, set `predictor.task: classification` and the response
also includes `probabilities`.

## Hot reload

Serve with `--watch` to reload the model in place when you retrain — no restart:

```bash
make serve CONFIG=examples/regression/config.yml WATCH=--watch
```

A background watcher polls `config.yml` and the `.joblib`. When either changes it
rebuilds the runtime and swaps it atomically (blue/green). Two things make it
safe:

- `train.py` writes the model atomically (temp file + rename), so the watcher
  never loads a half-written file.
- If a new model fails to load, the error is logged and the **current model keeps
  serving** — a bad artifact never takes the server down.

So the loop becomes: `uv run python train.py` (or edit `config.yml`) → save → live.

## Files

| File | Role |
|---|---|
| `train.py` | trains a `GradientBoostingRegressor` on synthetic data → `model.joblib` |
| `config.yml` | predictor config (local joblib + feature schema) |
| `run.py` | HTTP client that POSTs feature rows to `/predict` |
