import json
import os
import time
from pathlib import Path


def cache_dir(kind: str) -> Path:
    root = os.environ.get("SIMPLE_LOCAL_CACHE") or os.environ.get("XDG_CACHE_HOME")
    base = Path(root) if root else Path.home() / ".cache"
    return base / "simple-local" / kind


def write(kind: str, key: str, payload: dict) -> None:
    path = cache_dir(kind) / f"{key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"fetched_at": time.time(), **payload}, indent=2))


def read(kind: str, key: str) -> tuple[dict, float] | None:
    path = cache_dir(kind) / f"{key}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return data, float(data["fetched_at"])
    except (ValueError, KeyError, TypeError):
        return None


def age_days(fetched_at: float | None) -> float | None:
    return None if fetched_at is None else (time.time() - fetched_at) / 86400
