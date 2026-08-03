import os
import re
import sys
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Source(BaseModel):
    provider: Literal["huggingface", "local", "s3"] = "huggingface"
    repo: str | None = None  # huggingface
    file: str | None = None  # huggingface + local
    bucket: str | None = None  # s3
    key: str | None = None  # s3: exact object key
    prefix: str | None = None  # s3: versioned layout {prefix}/{version}/<artifacts>
    version: str = "latest"  # s3 prefix: "latest" | "active" (pointer file) | explicit

    @model_validator(mode="after")
    def _check(self) -> "Source":
        if self.provider == "huggingface" and not (self.repo and self.file):
            raise ValueError("huggingface source requires repo and file")
        if self.provider == "local" and not self.file:
            raise ValueError("local source requires file")
        if self.provider == "s3" and not (self.bucket and (self.key or self.prefix)):
            raise ValueError("s3 source requires bucket and one of key or prefix")
        return self


class Adapter(BaseModel):
    name: str
    source: Source
    scale: float = 1.0


class Draft(BaseModel):
    source: Source
    max_tokens: int = Field(default=16, alias="max")
    min_tokens: int = Field(default=1, alias="min")

    model_config = ConfigDict(populate_by_name=True)


class Inference(BaseModel):
    context_length: int = 4096
    threads: int = 0  # 0 = all cores
    parallel: int = 1  # concurrent request slots (continuous batching)
    mlock: bool = False
    no_mmap: bool = False
    cache_type_k: str | None = None  # e.g. q8_0 to quantize the KV cache
    cache_type_v: str | None = None
    draft: Draft | None = None  # speculative decoding draft model
    extra_args: list[str] = []  # passed to llama-server verbatim
    llama_server_path: str = "llama-server"  # binary on PATH, or absolute path


class Feature(BaseModel):
    name: str
    type: Literal["float", "int", "bool", "str"] = "float"


class FeatureSchema(BaseModel):
    features: list[Feature] | None = None


class Predictor(BaseModel):
    task: Literal["regression", "classification"] = "regression"
    # 'schema' shadows a BaseModel attribute, so store under a safe name.
    input_schema: FeatureSchema | None = Field(default=None, alias="schema")

    model_config = ConfigDict(populate_by_name=True)


class ModelSpec(BaseModel):
    name: str
    kind: Literal["llm", "predictor", "custom"] = "llm"
    source: Source | None = None
    runtime: str | None = None  # custom: "module:Class" or "path/to/file.py:Class"
    config: dict = {}  # custom: opaque passthrough to the runtime's load()
    embeddings: bool = False  # llm: serve /v1/embeddings instead of chat
    chat_template: str | None = None  # built-in llama.cpp template name
    chat_template_file: str | None = None  # path to a .jinja file (implies --jinja)
    adapters: list[Adapter] = []
    inference: Inference = Inference()
    predictor: Predictor = Predictor()

    @model_validator(mode="after")
    def _check(self) -> "ModelSpec":
        if self.kind != "custom" and self.source is None:
            raise ValueError(f"model '{self.name}': source is required for kind: {self.kind}")
        if self.kind == "custom" and not self.runtime:
            raise ValueError(f"model '{self.name}': runtime is required for kind: custom")
        if self.kind != "custom" and self.runtime:
            raise ValueError(f"model '{self.name}': runtime is only supported for kind: custom")
        if self.kind != "llm" and self.adapters:
            raise ValueError(f"model '{self.name}': adapters are only supported for kind: llm")
        if self.kind != "llm" and self.embeddings:
            raise ValueError(f"model '{self.name}': embeddings is only supported for kind: llm")
        if self.chat_template and self.chat_template_file:
            raise ValueError(
                f"model '{self.name}': set chat_template or chat_template_file, not both"
            )
        return self


class Server(BaseModel):
    host: str = "localhost"
    port: int = 8081
    api_key: str = ""

    @field_validator("api_key", mode="before")
    @classmethod
    def _none_to_empty(cls, v):
        return v or ""


class Config(BaseModel):
    models: list[ModelSpec] = Field(min_length=1)
    server: Server = Server()

    @model_validator(mode="after")
    def _unique_names(self) -> "Config":
        seen: set[str] = set()
        for name in self.served_names():
            if name in seen:
                raise ValueError(f"duplicate model/adapter name: '{name}'")
            seen.add(name)
        return self

    def served_names(self) -> list[str]:
        names = []
        for m in self.models:
            names.append(m.name)
            names.extend(a.name for a in m.adapters)
        return names


_VAR = re.compile(r"\$\{(\w+)\}|\$(\w+)")


def _expand_env(text: str) -> str:
    return _VAR.sub(lambda m: os.environ.get(m.group(1) or m.group(2), ""), text)


def _upgrade_legacy(data: dict) -> dict:
    if "models" in data or "model_name" not in data:
        return data
    print(
        "note: single-model config format is deprecated; move model settings under a 'models:' list",
        file=sys.stderr,
    )
    model = {"name": data["model_name"]}
    for key in ("kind", "source", "inference", "predictor"):
        if key in data:
            model[key] = data[key]
    upgraded = {"models": [model]}
    if "server" in data:
        upgraded["server"] = data["server"]
    return upgraded


def load(path: str) -> Config:
    raw = Path(path).read_text()
    data = yaml.safe_load(_expand_env(raw))
    return Config.model_validate(_upgrade_legacy(data))
