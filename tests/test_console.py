"""Scaffolding a model directory from what the console knows: a HuggingFace name
and a few numbers."""

import pytest

from simple_local import config as config_mod
from simple_local.deploy import check, resolve
from simple_local.deploy.console import ModelRequest, build_config, scaffold, slug


def request(**overrides) -> ModelRequest:
    data = {"model_name": "Qwen/Qwen3-Embedding-0.6B"}
    data.update(overrides)
    return ModelRequest.model_validate(data)


@pytest.mark.parametrize(
    "hf_name, expected",
    [
        ("Qwen/Qwen3-Embedding-0.6B", "qwen3-embedding-0-6b"),
        ("BAAI/bge-large-en-v1.5", "bge-large-en-v1-5"),
        ("meta-llama/Llama-3.1-8B-Instruct", "llama-3-1-8b-instruct"),
        ("org/___", "model"),
    ],
)
def test_route_names_survive_dns(hf_name, expected):
    # The route name becomes a hostname label, so it has to be lowercase
    # alphanumeric with no leading or trailing hyphen.
    assert slug(hf_name) == expected
    assert expected == expected.lower().strip("-")


@pytest.mark.parametrize(
    "hf_name, modality",
    [
        ("Qwen/Qwen3-Embedding-0.6B", "embeddings"),
        ("BAAI/bge-large-en", "embeddings"),
        ("Qwen/Qwen3-VL-Embedding-2B", "vision-language"),
        ("openai/clip-vit-large", "vision-language"),
        ("meta-llama/Llama-3.1-8B-Instruct", "text"),
    ],
)
def test_modality_is_inferred_from_the_name(hf_name, modality):
    assert request(model_name=hf_name).inferred_modality == modality


def test_an_explicit_modality_beats_the_guess():
    assert request(modality="text").inferred_modality == "text"


def test_an_unknown_modality_is_rejected_rather_than_passed_through():
    with pytest.raises(ValueError, match="modality must be one of"):
        request(modality="audio")


def test_a_scaffolded_config_passes_its_own_validation(tmp_path):
    # The generated inference block and the generated replica size have to agree
    # — a scaffold that fails `validate` is worse than no scaffold.
    directory = scaffold(request(), tmp_path)
    deploy = resolve(directory / "config.yml")
    served = config_mod.load(str(directory / "config.yml"))
    assert [f for f in check(deploy, served) if f.level == "error"] == []


def test_a_vision_model_gets_a_runtime_to_edit(tmp_path):
    directory = scaffold(request(model_name="Qwen/Qwen3-VL-Embedding-2B"), tmp_path)
    assert (directory / "runtime.py").exists()
    assert "runtime.py:Embedder" in (directory / "config.yml").read_text()


def test_scaffolding_never_overwrites_a_tuned_config(tmp_path):
    scaffold(request(), tmp_path)
    with pytest.raises(FileExistsError):
        scaffold(request(), tmp_path)


def test_workspaces_separate_directories(tmp_path):
    shared = scaffold(request(), tmp_path)
    siloed = scaffold(request(workspace="hps"), tmp_path)
    assert shared.parent.name == "shared" and siloed.parent.name == "hps"
    assert shared != siloed


def test_identity_reaches_the_deploy_block_and_the_tags(tmp_path):
    directory = scaffold(request(workspace="hps"), tmp_path)
    deploy = resolve(directory / "config.yml")
    assert deploy.workspace == "hps"
    assert deploy.tags == {
        "project": "aerium",
        "managed-by": "simple-local",
        "workspace": "hps",
        "model": "qwen3-embedding-0-6b",
        "hf-name": "Qwen/Qwen3-Embedding-0.6B",
        "modality": "embeddings",
    }


def test_a_gpu_request_serves_through_vllm_not_llama():
    built = build_config(request(gpu="Consumption-GPU-NC8as-T4"))
    assert built["models"][0]["kind"] == "vllm"
    assert built["deploy"]["gpu"] == "Consumption-GPU-NC8as-T4"


def test_a_cpu_request_leaves_the_quantisation_to_a_human():
    # Which GGUF quant to take is a judgement call with real memory
    # consequences, so it is marked rather than guessed.
    built = build_config(request())
    assert "CHOOSE-A-QUANT" in built["models"][0]["source"]["file"]


def test_explicit_sizing_wins_over_the_defaults():
    built = build_config(request(cpu=2.0, memory_gi=4.0, min_replicas=2))
    assert built["deploy"]["cpu"] == 2.0
    assert built["deploy"]["memory_gi"] == 4.0
    assert built["deploy"]["min_replicas"] == 2
