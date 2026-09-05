import pytest
from pydantic import ValidationError

from simple_local.config import ModelSpec
from simple_local.download import ModelPaths
from simple_local.runtimes.vllm import build_vllm_args


def spec(**overrides) -> ModelSpec:
    data = {"name": "embed", "kind": "vllm", "embeddings": True,
            "vllm": {"model": "Qwen/Qwen3-Embedding-0.6B"}}
    data.update(overrides)
    return ModelSpec.model_validate(data)


def args(**overrides) -> list[str]:
    return build_vllm_args(spec(**overrides), ModelPaths(model=None), 9001)


def flag(argv, name):
    return argv[argv.index(name) + 1] if name in argv else None


def test_the_model_name_the_proxy_sees_is_ours_not_the_repo_id():
    argv = args()
    assert flag(argv, "--served-model-name") == "embed"
    assert "Qwen/Qwen3-Embedding-0.6B" in argv


def test_embeddings_ask_for_the_pooling_task():
    assert "--task" in args() and flag(args(), "--task") == "embed"


def test_pooling_is_passed_through_because_the_default_would_change_the_vectors():
    argv = args(vllm={"model": "m", "pooling": "last"})
    assert '"pooling_type": "LAST"' in flag(argv, "--override-pooler-config")


def test_pooling_is_only_sent_when_stated():
    assert "--override-pooler-config" not in args(vllm={"model": "m"})


def test_generation_models_do_not_get_the_embed_task():
    assert "--task" not in args(embeddings=False)


def test_sizing_flags_reach_the_command():
    argv = args(vllm={"model": "m", "max_model_len": 4096, "max_num_seqs": 8,
                      "dtype": "float16", "gpu_memory_utilization": 0.8,
                      "tensor_parallel_size": 2})
    assert flag(argv, "--max-model-len") == "4096"
    assert flag(argv, "--max-num-seqs") == "8"
    assert flag(argv, "--dtype") == "float16"
    assert flag(argv, "--gpu-memory-utilization") == "0.8"
    assert flag(argv, "--tensor-parallel-size") == "2"


def test_extra_args_are_passed_verbatim_and_last():
    argv = args(vllm={"model": "m", "extra_args": ["--enforce-eager"]})
    assert argv[-1] == "--enforce-eager"


def test_a_downloaded_source_is_used_when_no_model_id_is_given(tmp_path):
    weights = tmp_path / "model"
    weights.mkdir()
    target = ModelSpec.model_validate(
        {"name": "embed", "kind": "vllm",
         "source": {"provider": "local", "file": str(weights)}}
    )
    assert str(weights) in build_vllm_args(target, ModelPaths(model=weights), 9001)


def test_vllm_needs_something_to_serve():
    with pytest.raises(ValidationError, match="vllm.model or a source"):
        ModelSpec.model_validate({"name": "embed", "kind": "vllm"})


def test_adapters_are_still_llama_only():
    with pytest.raises(ValidationError, match="adapters are only supported"):
        ModelSpec.model_validate(
            {"name": "e", "kind": "vllm", "vllm": {"model": "m"},
             "adapters": [{"name": "a", "source": {"provider": "local", "file": "a.gguf"}}]}
        )


def test_vllm_weights_are_prefetched_so_they_are_not_a_cold_start(monkeypatch):
    from simple_local import download

    pulled = []
    monkeypatch.setattr(download, "snapshot_download", lambda repo: pulled.append(repo))
    assert download.snapshot_custom_model(spec()) == "Qwen/Qwen3-Embedding-0.6B"
    assert pulled == ["Qwen/Qwen3-Embedding-0.6B"]


def test_a_local_path_is_not_mistaken_for_a_repo_id(monkeypatch):
    from simple_local import download

    monkeypatch.setattr(download, "snapshot_download", lambda repo: pytest.fail("should not fetch"))
    assert download.snapshot_custom_model(spec(vllm={"model": "/models/local-copy"})) is None
