from pathlib import Path

from huggingface_hub import hf_hub_download

from .config import Config


def _models_dir() -> Path:
    return Path.home() / "simple-local" / "models"


def ensure_model(cfg: Config) -> Path:
    """Return the local model path, downloading from Hugging Face if needed."""
    if cfg.source.provider == "local":
        path = Path(cfg.source.file).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"model file not found: {path}")
        return path

    target = _models_dir() / cfg.source.repo.replace("/", "_")
    print(f"Downloading {cfg.source.file}\n  from {cfg.source.repo}")
    downloaded = hf_hub_download(
        repo_id=cfg.source.repo,
        filename=cfg.source.file,
        local_dir=str(target),
    )
    print(f"Model ready: {downloaded}")
    return Path(downloaded)
