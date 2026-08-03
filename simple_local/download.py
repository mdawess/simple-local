import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import boto3
from huggingface_hub import hf_hub_download

from .config import ModelSpec, Source


def _models_dir() -> Path:
    return Path.home() / "simple-local" / "models"


@dataclass
class ModelPaths:
    model: Path | None  # artifact file, or directory for s3 prefix sources
    adapters: dict[str, Path] = field(default_factory=dict)  # adapter name -> path
    draft: Path | None = None
    version: str | None = None  # resolved s3 version, when applicable

    def all_paths(self) -> list[Path]:
        paths = [p for p in (self.model, self.draft) if p is not None]
        paths.extend(self.adapters.values())
        return paths


def _s3_client():
    return boto3.client("s3")


_VERSION_SPLIT = re.compile(r"(\d+)")


def _version_key(version: str) -> list:
    return [int(t) if t.isdigit() else t for t in _VERSION_SPLIT.split(version)]


def _resolve_s3_version(client, source: Source) -> str:
    prefix = source.prefix.rstrip("/") + "/"
    if source.version == "active":
        pointer = client.get_object(Bucket=source.bucket, Key=prefix + "active.json")
        return json.loads(pointer["Body"].read())["version"]
    if source.version == "latest":
        resp = client.list_objects_v2(Bucket=source.bucket, Prefix=prefix, Delimiter="/")
        versions = [
            p["Prefix"][len(prefix):].strip("/") for p in resp.get("CommonPrefixes", [])
        ]
        if not versions:
            raise FileNotFoundError(f"no versions under s3://{source.bucket}/{prefix}")
        return max(versions, key=_version_key)
    return source.version


def _download_s3_tree(client, bucket: str, remote_prefix: str, target: Path) -> None:
    resp = client.list_objects_v2(Bucket=bucket, Prefix=remote_prefix)
    contents = [o for o in resp.get("Contents", []) if o["Key"] != remote_prefix]
    if not contents:
        raise FileNotFoundError(f"no objects under s3://{bucket}/{remote_prefix}")
    for obj in contents:
        dest = target / obj["Key"][len(remote_prefix):]
        if dest.exists() and dest.stat().st_size == obj["Size"]:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"Fetching s3://{bucket}/{obj['Key']}")
        client.download_file(bucket, obj["Key"], str(dest))


def _ensure_s3(source: Source) -> tuple[Path, str | None]:
    client = _s3_client()
    if source.key:
        target = _models_dir() / "s3" / source.bucket / source.key
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            print(f"Fetching s3://{source.bucket}/{source.key}")
            client.download_file(source.bucket, source.key, str(target))
        return target, None
    version = _resolve_s3_version(client, source)
    remote_prefix = source.prefix.rstrip("/") + "/" + version + "/"
    target = _models_dir() / "s3" / source.bucket / source.prefix.rstrip("/") / version
    _download_s3_tree(client, source.bucket, remote_prefix, target)
    return target, version


def ensure_source(source: Source) -> tuple[Path, str | None]:
    """Return (local path, resolved version) for a source, downloading if needed.
    The path is a file, or a directory for s3 prefix sources."""
    if source.provider == "local":
        path = Path(source.file).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"model file not found: {path}")
        return path, None
    if source.provider == "s3":
        return _ensure_s3(source)

    target = _models_dir() / source.repo.replace("/", "_")
    print(f"Fetching {source.file}\n  from {source.repo}")
    downloaded = hf_hub_download(
        repo_id=source.repo,
        filename=source.file,
        local_dir=str(target),
    )
    return Path(downloaded), None


def ensure_file(source: Source) -> Path:
    return ensure_source(source)[0]


def _sole_file(path: Path, pattern: str, what: str) -> Path:
    matches = sorted(path.glob(pattern))
    if len(matches) != 1:
        raise ValueError(
            f"{what}: expected exactly one {pattern} file in {path}, found {len(matches)}"
        )
    return matches[0]


def ensure_model_files(spec: ModelSpec) -> ModelPaths:
    if spec.source is None:
        return ModelPaths(model=None)
    model, version = ensure_source(spec.source)
    if model.is_dir() and spec.kind == "llm":
        model = _sole_file(model, "*.gguf", f"model '{spec.name}'")
    if model.is_dir() and spec.kind == "predictor":
        model = _sole_file(model, "*.joblib", f"model '{spec.name}'")
    paths = ModelPaths(model=model, version=version)
    for adapter in spec.adapters:
        path = ensure_file(adapter.source)
        if path.suffix != ".gguf":
            raise ValueError(
                f"adapter '{adapter.name}' must be a GGUF file, got {path.name}; "
                "convert PEFT output with llama.cpp's convert_lora_to_gguf.py"
            )
        paths.adapters[adapter.name] = path
    if spec.inference.draft:
        paths.draft = ensure_file(spec.inference.draft.source)
    return paths
