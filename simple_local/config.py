import os
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Source(BaseModel):
    provider: Literal["huggingface", "local"] = "huggingface"
    repo: str | None = None  # required for huggingface
    file: str

    @model_validator(mode="after")
    def _check(self) -> "Source":
        if self.provider == "huggingface" and not self.repo:
            raise ValueError("source.repo is required when provider is 'huggingface'")
        return self


class Inference(BaseModel):
    context_length: int = 4096
    threads: int = 0  # 0 = all cores
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


class Server(BaseModel):
    host: str = "localhost"
    port: int = 8081
    api_key: str = ""

    @field_validator("api_key", mode="before")
    @classmethod
    def _none_to_empty(cls, v):
        return v or ""


class Config(BaseModel):
    kind: Literal["llm", "predictor"] = "llm"
    model_name: str
    source: Source
    inference: Inference = Inference()
    predictor: Predictor = Predictor()
    server: Server = Server()

    model_config = ConfigDict(protected_namespaces=())


_VAR = re.compile(r"\$\{(\w+)\}|\$(\w+)")


def _expand_env(text: str) -> str:
    return _VAR.sub(lambda m: os.environ.get(m.group(1) or m.group(2), ""), text)


def load(path: str) -> Config:
    raw = Path(path).read_text()
    data = yaml.safe_load(_expand_env(raw))
    return Config.model_validate(data)
