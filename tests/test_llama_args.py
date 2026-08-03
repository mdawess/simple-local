from pathlib import Path

from simple_local.config import ModelSpec
from simple_local.download import ModelPaths
from simple_local.runtimes.llm import build_llama_args


def spec(**overrides) -> ModelSpec:
    data = {"name": "base", "source": {"provider": "local", "file": "m.gguf"}}
    data.update(overrides)
    return ModelSpec.model_validate(data)


def test_default_args():
    args = build_llama_args(spec(), ModelPaths(model=Path("/m.gguf")), 9999)
    assert args[0] == "llama-server"
    assert ("--model", "/m.gguf") in zip(args, args[1:])
    assert ("--alias", "base") in zip(args, args[1:])
    assert ("--port", "9999") in zip(args, args[1:])
    assert "--metrics" in args
    assert "--parallel" not in args
    assert "--lora" not in args


def test_all_the_flags():
    s = spec(
        chat_template_file="t.jinja",
        adapters=[
            {"name": "a1", "source": {"provider": "local", "file": "a1.gguf"}},
            {"name": "a2", "source": {"provider": "local", "file": "a2.gguf"}},
        ],
        inference={
            "parallel": 4,
            "mlock": True,
            "no_mmap": True,
            "cache_type_k": "q8_0",
            "cache_type_v": "q8_0",
            "draft": {"source": {"provider": "local", "file": "d.gguf"}, "max": 24, "min": 2},
            "extra_args": ["--flash-attn", "on"],
        },
    )
    paths = ModelPaths(
        model=Path("/m.gguf"),
        adapters={"a1": Path("/a1.gguf"), "a2": Path("/a2.gguf")},
        draft=Path("/d.gguf"),
    )
    args = build_llama_args(s, paths, 9999)
    pairs = list(zip(args, args[1:]))
    assert ("--parallel", "4") in pairs
    assert "--mlock" in args and "--no-mmap" in args
    assert ("--cache-type-k", "q8_0") in pairs and ("--cache-type-v", "q8_0") in pairs
    assert "--jinja" in args and ("--chat-template-file", "t.jinja") in pairs
    # adapter order defines the per-request lora ids
    lora_paths = [b for a, b in pairs if a == "--lora"]
    assert lora_paths == ["/a1.gguf", "/a2.gguf"]
    assert ("--model-draft", "/d.gguf") in pairs
    assert ("--draft-max", "24") in pairs and ("--draft-min", "2") in pairs
    assert args[-2:] == ["--flash-attn", "on"]


def test_embeddings_flag():
    args = build_llama_args(spec(embeddings=True), ModelPaths(model=Path("/m.gguf")), 1)
    assert "--embeddings" in args
    assert "--embeddings" not in build_llama_args(spec(), ModelPaths(model=Path("/m.gguf")), 1)


def test_builtin_chat_template():
    args = build_llama_args(
        spec(chat_template="chatml"), ModelPaths(model=Path("/m.gguf")), 1
    )
    assert ("--chat-template", "chatml") in zip(args, args[1:])
    assert "--jinja" not in args
