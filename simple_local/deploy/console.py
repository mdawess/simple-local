"""The flat config the console writes, and how it becomes a model directory.

The console should not have to know about `models:` lists, runtime kinds or
llama.cpp flags. It writes six fields; this expands them into a directory the
rest of the tooling already understands, so there is exactly one deployment path
whether a human or the console started it.

    model_name: Qwen/Qwen3-Embedding-0.6B
    gpu: NC8as-T4
    cpu: "4.0"
    memory_gi: 16
    min_replicas: 1
    max_replicas: 4
"""

import re
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator

from .config import DEFAULT_WORKSPACE, MODALITIES
from .units import parse_memory_gi

# Embedding models are named unambiguously enough to route on, and guessing
# wrong is cheap to correct — modality is overridable.
_EMBEDDING_HINT = re.compile(r"embed|bge|gte|e5|minilm", re.IGNORECASE)
_VISION_HINT = re.compile(r"\bvl\b|vision|clip|siglip|multimodal", re.IGNORECASE)


class ModelRequest(BaseModel):
    """What the console sends, or what sits in the config file it writes."""

    model_name: str  # HF repo id
    workspace: str = DEFAULT_WORKSPACE
    id: str | None = None  # route name; derived from model_name when absent
    modality: str | None = None
    gpu: str | None = None
    cpu: float | None = None
    memory_gi: float | None = None
    min_replicas: int | None = None
    max_replicas: int | None = None
    zone: str | None = None  # custom domain zone, when there is one
    env: dict[str, str] = Field(default_factory=dict)

    @field_validator("memory_gi", mode="before")
    @classmethod
    def _memory(cls, v):
        return None if v is None else parse_memory_gi(v)

    @field_validator("modality")
    @classmethod
    def _known_modality(cls, v):
        if v is not None and v not in MODALITIES:
            raise ValueError(f"modality must be one of {', '.join(MODALITIES)}")
        return v

    @property
    def route(self) -> str:
        return self.id or slug(self.model_name)

    @property
    def inferred_modality(self) -> str:
        if self.modality:
            return self.modality
        if _VISION_HINT.search(self.model_name):
            return "vision-language"
        if _EMBEDDING_HINT.search(self.model_name):
            return "embeddings"
        return "text"


def slug(hf_name: str) -> str:
    """`Qwen/Qwen3-Embedding-0.6B` -> `qwen3-embedding-0-6b`. The result is a
    hostname label, so it has to survive DNS: lowercase, alphanumeric and
    hyphens, no leading or trailing hyphen."""
    tail = hf_name.split("/")[-1]
    cleaned = re.sub(r"[^a-z0-9]+", "-", tail.lower()).strip("-")
    return cleaned or "model"


def load_request(path: str | Path) -> ModelRequest:
    data = yaml.safe_load(Path(path).read_text()) or {}
    return ModelRequest.model_validate(data)


def _served_model(request: ModelRequest) -> dict:
    """A vision-language model needs a runtime that knows how to decode images,
    so it cannot be expressed as a plain llama.cpp source. Everything else can."""
    if request.inferred_modality == "vision-language":
        return {
            "name": request.route,
            "kind": "custom",
            "runtime": "runtime.py:Embedder",
            "config": {"model": request.model_name, "batch_size": 8},
        }
    model = {
        "name": request.route,
        "kind": "vllm" if request.gpu else "llm",
        "embeddings": request.inferred_modality == "embeddings",
    }
    if request.gpu:
        model["vllm"] = {"model": request.model_name, "max_model_len": 4096}
    else:
        # llama.cpp needs a specific GGUF file rather than a repo, and which
        # quantisation to take is a judgement call — so it is left for a human
        # rather than guessed.
        model["source"] = {
            "provider": "huggingface",
            "repo": f"{request.model_name}-GGUF",
            "file": "CHOOSE-A-QUANT.gguf",
        }
        # Sized together with CPU_LLM_REPLICA below: the KV cache is
        # preallocated for the whole context, so these two have to agree or the
        # server loads cleanly and dies on the first large input.
        model["inference"] = {
            "context_length": 32768,
            "parallel": 8,
            "batch_size": 4096,
            "ubatch_size": 4096,
            "max_input_tokens": 4096,
        }
    return model


# What the generated inference block above is sized against — the ceiling of a
# Consumption replica.
CPU_LLM_REPLICA = {"cpu": 4.0, "memory_gi": 8.0}


def build_config(request: ModelRequest) -> dict:
    defaults = {} if request.gpu else CPU_LLM_REPLICA
    deploy: dict = {
        "name": request.route,
        "workspace": request.workspace,
        "hf_name": request.model_name,
        "modality": request.inferred_modality,
    }
    for key, value in (
        ("gpu", request.gpu),
        ("cpu", request.cpu if request.cpu is not None else defaults.get("cpu")),
        ("memory_gi", request.memory_gi if request.memory_gi is not None else defaults.get("memory_gi")),
        ("min_replicas", request.min_replicas),
        ("max_replicas", request.max_replicas),
    ):
        if value is not None:
            deploy[key] = value
    if request.env:
        deploy["env"] = request.env
    if request.zone:
        deploy["domain"] = {"zone": request.zone, "certificate": "managed"}

    return {
        "models": [_served_model(request)],
        "server": {
            "host": "${SIMPLE_LOCAL_HOST}",
            "port": "${SIMPLE_LOCAL_PORT}",
            "api_key": "${SIMPLE_LOCAL_API_KEY}",
        },
        "logging": {
            "mysql": {
                "host": "${MYSQL_HOST}",
                "port": 3306,
                "user": "${MYSQL_USER}",
                "password": "${MYSQL_PASSWORD}",
                "database": "${MYSQL_DATABASE}",
                "ssl": True,
            }
        },
        "deploy": deploy,
    }


HEADER = """# Generated by `simple-local new` from {hf_name}.
# The app name is the public route, so it names what this serves rather than the
# model serving it — swapping models later is a config change, not a URL change.
"""


def scaffold(request: ModelRequest, root: str | Path) -> Path:
    """Writes the model directory. Returns its path.

    Refuses to overwrite: regenerating over a directory someone has since tuned
    would silently discard that work, and the config is the only record of it.
    """
    directory = Path(root) / request.workspace / request.route
    config = directory / "config.yml"
    if config.exists():
        raise FileExistsError(f"{config} already exists")

    directory.mkdir(parents=True, exist_ok=True)
    body = yaml.safe_dump(build_config(request), sort_keys=False, width=100)
    config.write_text(HEADER.format(hf_name=request.model_name) + body)

    if request.inferred_modality == "vision-language":
        (directory / "runtime.py").write_text(VL_RUNTIME_STUB)
    return directory


VL_RUNTIME_STUB = '''"""Vision-language embedder. Generated as a starting point — the pooling and
prompt handling here decide what the vectors mean, so check them against the
model card before indexing anything with it."""

from sentence_transformers import SentenceTransformer

from simple_local.runtimes.custom import ModelContext


class Embedder:
    def load(self, ctx: ModelContext) -> None:
        settings = ctx.config
        self.batch_size = settings.get("batch_size", 8)
        self.model = SentenceTransformer(
            settings["model"],
            device=settings.get("device"),
            model_kwargs={"dtype": settings.get("dtype", "auto")},
        )

    def predict(self, request: dict) -> dict:
        inputs = request.get("input")
        if not isinstance(inputs, list):
            inputs = [inputs]
        vectors = self.model.encode(
            inputs,
            batch_size=request.get("batch_size", self.batch_size),
            normalize_embeddings=request.get("normalize", True),
        )
        return {
            "object": "list",
            "data": [
                {"object": "embedding", "index": i, "embedding": v.tolist()}
                for i, v in enumerate(vectors)
            ],
        }
'''
