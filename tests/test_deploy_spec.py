import pytest

from simple_local import config as config_mod
from simple_local.deploy import spec as spec_mod
from simple_local.deploy.build import image_digest, render_dockerfile, stage
from simple_local.deploy import profiles as profiles_mod
from simple_local.deploy.profiles import Catalog, Profile
from simple_local.deploy.config import LLAMA_BASE_IMAGE, PYTHON_BASE_IMAGE
from simple_local.deploy.spec import Deploy, check, resolve

# An explicit catalog, so no test depends on a live Azure call or on whatever
# happens to be cached on the machine running them. Mirrors what
# `az containerapp env workload-profile list-supported` returns.
CATALOG = Catalog(
    location="test",
    source="azure",
    profiles={
        p.name: p
        for p in [
            Profile(name="Consumption", cores=4, memory_gi=8),
            Profile(name="Flex", cores=32, memory_gi=128),
            Profile(name="D4", category="GeneralPurpose", cores=4, memory_gi=16),
            Profile(
                name="Consumption-GPU-NC8as-T4",
                category="Consumption-GPU-T4",
                cores=8,
                memory_gi=56,
                gpus=1,
                quota=2,
            ),
            Profile(name="NC96-A100", category="GPU-NC-A100", cores=96, memory_gi=880, gpus=4),
        ]
    },
)


def check_with(target, served=None, **kwargs):
    catalog = kwargs.pop("catalog", CATALOG)
    return check(target, served, catalog=catalog, **kwargs)


def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return path


def deploy(**overrides) -> Deploy:
    data = {"name": "app", "directory": ".", "container_config": "config.yml"}
    data.update(overrides)
    return Deploy.model_validate(data)


def levels(findings, field):
    return [f.level for f in findings if f.field == field]


def test_memory_accepts_both_the_repo_and_console_spellings():
    assert deploy(memory="16.0Gi").memory_gi == 16.0
    assert deploy(memory_gi=16).memory_gi == 16.0
    assert deploy(memory="56Gi").memory_arg == "56Gi"


def test_defaults_follow_the_presence_of_a_gpu():
    cpu_only = deploy()
    assert (cpu_only.cpu, cpu_only.memory_gi, cpu_only.max_replicas) == (2.0, 4.0, 3)
    gpu = deploy(gpu="NC8as-T4")
    assert (gpu.cpu, gpu.memory_gi, gpu.max_replicas) == (8.0, 56.0, 1)


def test_resolve_defaults_name_and_container_config(tmp_path):
    directory = tmp_path / "embeddings"
    directory.mkdir()
    write(directory, "config.container.yml", "models: []")
    config = write(directory, "config.yml", "deploy:\n  cpu: 3.0\n")

    resolved = resolve(config)
    assert resolved.name == "simple-local-embeddings"
    assert resolved.container_config.endswith("embeddings/config.container.yml")
    assert resolved.cpu == 3.0


def test_resolve_without_a_deploy_block(tmp_path):
    config = write(tmp_path, "config.yml", "models: []")
    resolved = resolve(config)
    assert resolved.directory == str(tmp_path.resolve())
    assert resolved.container_config == str(config.resolve())


def test_consumption_cap_is_an_error():
    findings = check_with(deploy(cpu=8.0, memory_gi=16.0))
    assert levels(findings, "cpu") == ["error"]
    assert levels(findings, "memory") == ["error"]


def test_requesting_less_than_the_gpu_node_is_fine():
    # vl-embeddings runs 4 vCPU / 16Gi on a T4 node that provides 8 / 56.
    assert check_with(deploy(gpu="NC8as-T4", cpu=4.0, memory_gi=16.0, max_replicas=1)) == []


def test_requesting_more_than_the_gpu_node_is_an_error():
    findings = check_with(deploy(gpu="NC8as-T4", cpu=16.0, memory_gi=64.0, max_replicas=1))
    assert levels(findings, "cpu") == ["error"]
    assert levels(findings, "memory") == ["error"]


def test_replicas_past_the_gpu_quota_warn_rather_than_block():
    findings = check_with(deploy(gpu="NC8as-T4", max_replicas=4))
    assert levels(findings, "max_replicas") == ["warning"]
    assert not [f for f in findings if f.level == "error"]


def test_hardware_azure_does_not_offer_is_an_error_that_names_what_it_does():
    findings = check_with(deploy(gpu="NC40-H100", cpu=40.0, memory_gi=320.0))
    assert levels(findings, "gpu") == ["error"]
    message = next(f.message for f in findings if f.field == "gpu")
    assert "Consumption-GPU-NC8as-T4" in message
    assert "deploy.profiles" in message


def test_a_config_can_add_hardware_the_catalog_does_not_have():
    target = deploy(
        gpu="NC40-H100",
        cpu=40.0,
        memory_gi=320.0,
        max_replicas=1,
        profiles={"NC40-H100": {"category": "GPUAccelerated", "cores": 40, "memory_gi": 320, "gpus": 1}},
    )
    extended = CATALOG.model_copy(
        update={"profiles": {**CATALOG.profiles, **target.profiles}}
    )
    assert check_with(target, catalog=extended) == []


def test_multi_gpu_profiles_are_selectable():
    # westus3 offers dedicated NC48/NC96-A100 with 2 and 4 GPUs per replica.
    target = deploy(gpu="NC96-A100", cpu=96.0, memory_gi=880.0, max_replicas=1)
    assert check_with(target) == []
    profile, azure_name = CATALOG.resolve("NC96-A100")
    assert (profile.gpus, azure_name) == (4, "NC96-A100")


def test_the_short_and_azure_spellings_both_resolve():
    short = deploy(gpu="NC8as-T4", cpu=4.0, memory_gi=16.0, max_replicas=1)
    full = deploy(gpu="Consumption-GPU-NC8as-T4", cpu=4.0, memory_gi=16.0, max_replicas=1)
    assert CATALOG.resolve("NC8as-T4")[1] == CATALOG.resolve("Consumption-GPU-NC8as-T4")[1]
    assert check_with(short) == check_with(full) == []


def test_consumption_is_a_profile_like_any_other():
    # No special case: the 4 vCPU / 8Gi ceiling is just that profile's node size.
    target = deploy(cpu=8.0, memory_gi=16.0)
    findings = check_with(target)
    assert levels(findings, "cpu") == ["error"]
    assert "Consumption provides 4 vCPU" in next(f.message for f in findings if f.field == "cpu")


def test_sizing_constants_are_overridable_per_config(tmp_path):
    config = served(tmp_path, context=65536, ubatch=4096, max_input=4096)
    assert levels(check_with(deploy(cpu=4.0, memory_gi=8.0), config), "models.embed.inference.context_length") == ["error"]

    # A model family with a smaller KV cache per token fits where the default does not.
    lean = deploy(cpu=4.0, memory_gi=8.0, sizing={"kv_kib_per_token": 24, "source": "measured elsewhere"})
    assert check_with(lean, config) == []


def test_replica_bounds():
    assert levels(check_with(deploy(max_replicas=0)), "max_replicas") == ["error"]
    assert levels(check_with(deploy(min_replicas=5, max_replicas=2)), "min_replicas") == ["error"]


def test_quota_is_declared_separately_from_size(tmp_path, monkeypatch):
    # Azure has no quota API, so a config records it without restating cores
    # and memory. Cache dir is redirected so this never touches a real one.
    monkeypatch.setenv("SIMPLE_LOCAL_CACHE", str(tmp_path))
    target = deploy(gpu="NC8as-T4", cpu=4.0, memory_gi=16.0, max_replicas=4, quotas={"NC8as-T4": 2})
    catalog = profiles_mod.load(
        "test",
        overrides={"Consumption-GPU-NC8as-T4": CATALOG.profiles["Consumption-GPU-NC8as-T4"]},
        quotas=target.quotas,
        live=False,
    )
    assert catalog.profiles["Consumption-GPU-NC8as-T4"].quota == 2
    assert levels(check_with(target, catalog=catalog), "max_replicas") == ["warning"]


def test_no_catalog_warns_rather_than_passing_silently(tmp_path, monkeypatch):
    monkeypatch.setenv("SIMPLE_LOCAL_CACHE", str(tmp_path))
    empty = profiles_mod.load("nowhere", live=False)
    assert empty.profiles == {}

    findings = check_with(deploy(cpu=999.0), catalog=empty)
    assert levels(findings, "gpu") == ["warning"]
    assert "not checked" in next(f.message for f in findings if f.field == "gpu")


def test_a_cached_catalog_round_trips(tmp_path, monkeypatch):
    monkeypatch.setenv("SIMPLE_LOCAL_CACHE", str(tmp_path))
    rows = [
        {"name": "Consumption", "properties": {"category": "Consumption", "cores": 4, "memoryGiB": 8}},
        {
            "name": "Consumption-GPU-NC8as-T4",
            "properties": {"category": "Consumption-GPU-T4", "cores": 8, "memoryGiB": 56, "gpus": 1},
        },
    ]
    monkeypatch.setattr(profiles_mod.azure, "json_out", lambda args: rows)

    live = profiles_mod.load("westus3", live=True)
    assert live.source == "azure"
    assert live.profiles["Consumption-GPU-NC8as-T4"].gpus == 1

    offline = profiles_mod.load("westus3", live=False)
    assert offline.source == "cache"
    assert offline.profiles.keys() == live.profiles.keys()
    assert not offline.stale


SERVED = """
models:
  - name: embed
    kind: llm
    embeddings: true
    source: {{ provider: local, file: m.gguf }}
    inference:
      context_length: {context}
      ubatch_size: {ubatch}
      max_input_tokens: {max_input}
"""


def served(tmp_path, context, ubatch, max_input):
    path = write(
        tmp_path,
        "served.yml",
        SERVED.format(context=context, ubatch=ubatch, max_input=max_input),
    )
    return config_mod.load(str(path))


def test_the_shipped_embeddings_sizing_passes(tmp_path):
    # 32768 context -> ~3.5Gi KV, 4096 ubatch -> ~3.3Gi compute, under 8Gi.
    config = served(tmp_path, context=32768, ubatch=4096, max_input=4096)
    assert check_with(deploy(cpu=4.0, memory_gi=8.0), config) == []


def test_context_that_loads_and_then_dies_is_blocked(tmp_path):
    config = served(tmp_path, context=65536, ubatch=4096, max_input=4096)
    findings = check_with(deploy(cpu=4.0, memory_gi=8.0), config)
    assert levels(findings, "models.embed.inference.context_length") == ["error"]


def test_ubatch_below_max_input_is_blocked(tmp_path):
    config = served(tmp_path, context=8192, ubatch=512, max_input=4096)
    findings = check_with(deploy(memory_gi=8.0), config)
    assert levels(findings, "models.embed.inference.ubatch_size") == ["error"]


def test_weights_are_counted_when_known(tmp_path):
    config = served(tmp_path, context=32768, ubatch=4096, max_input=4096)
    findings = check_with(deploy(cpu=4.0, memory_gi=8.0), config, weights_gi={"embed": 2.0})
    assert levels(findings, "models.embed.inference.context_length") == ["error"]


def digest_fixture(tmp_path):
    container = write(tmp_path, "config.container.yml", "models: []")
    return container, deploy(directory=str(tmp_path), container_config=str(container))


def test_image_digest_is_stable_and_tracks_the_served_config(tmp_path):
    container, target = digest_fixture(tmp_path)
    before = image_digest(target)
    assert before == image_digest(target)

    container.write_text("models: [] # a different model")
    assert image_digest(target) != before


def test_image_digest_tracks_every_other_file_in_the_model_directory(tmp_path):
    _, target = digest_fixture(tmp_path)
    before = image_digest(target)

    runtime = write(tmp_path, "runtime.py", "class R: pass")
    assert image_digest(target) != before

    runtime.write_text("class R: pass  # edited")
    assert image_digest(target) not in (before, None)


def test_image_digest_ignores_pycache(tmp_path):
    _, target = digest_fixture(tmp_path)
    before = image_digest(target)
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "runtime.cpython-312.pyc").write_bytes(b"\x00\x01")
    assert image_digest(target) == before


def test_the_base_image_follows_what_the_config_serves(tmp_path):
    llm = served(tmp_path, context=8192, ubatch=4096, max_input=4096)
    assert LLAMA_BASE_IMAGE in render_dockerfile(deploy(), llm)
    assert PYTHON_BASE_IMAGE in render_dockerfile(deploy(), None)


def test_the_generated_dockerfile_keeps_the_venv_out_of_workdir():
    dockerfile = render_dockerfile(deploy(), None)
    assert "uv venv /opt/venv" in dockerfile
    assert "WORKDIR /srv" in dockerfile
    assert '["/opt/venv/bin/simple-local", "serve", "-c", "/srv/config.yml"]' in dockerfile


def test_build_options_reach_the_dockerfile():
    target = deploy(
        build={
            "base_image": "python:3.12-slim",
            "system_packages": ["build-essential"],
            "requirements": ["torch"],
            "extras": ["vl"],
            "env": {"VL_DEVICE": "cuda"},
            "package": "simple-local==0.2.0",
        }
    )
    dockerfile = render_dockerfile(target, None)
    assert "build-essential" in dockerfile
    assert "uv pip install --python /opt/venv torch" in dockerfile
    assert "simple-local==0.2.0[vl]" in dockerfile
    assert 'VL_DEVICE="cuda"' in dockerfile
    assert "COPY wheels" not in dockerfile


def test_prefetch_can_be_turned_off():
    assert "simple-local download" in render_dockerfile(deploy(), None)
    assert "simple-local download" not in render_dockerfile(deploy(build={"prefetch": False}), None)


@pytest.fixture
def model_dir(tmp_path):
    write(tmp_path, "config.yml", "models: []\ndeploy: {}\n")
    write(tmp_path, "config.container.yml", "models: []  # the one that ships\n")
    write(tmp_path, "runtime.py", "class R: pass")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "runtime.pyc").write_bytes(b"\x00")
    return deploy(
        directory=str(tmp_path),
        container_config=str(tmp_path / "config.container.yml"),
        build={"package": "simple-local==0.2.0"},
    )


def test_staging_is_self_contained(model_dir):
    context = stage(model_dir, None)
    staged = {p.name for p in context.rglob("*") if p.is_file()}
    assert staged == {"Dockerfile", ".dockerignore", "config.yml", "runtime.py"}


def test_staging_ships_the_container_config_as_the_only_config(model_dir):
    context = stage(model_dir, None)
    assert "the one that ships" in (context / "config.yml").read_text()
    assert not (context / "config.container.yml").exists()


def test_staging_leaves_pycache_behind(model_dir):
    context = stage(model_dir, None)
    assert not list(context.rglob("*.pyc"))


def test_staging_builds_a_wheel_when_no_version_is_pinned(tmp_path):
    write(tmp_path, "config.container.yml", "models: []")
    target = deploy(
        directory=str(tmp_path), container_config=str(tmp_path / "config.container.yml")
    )
    context = stage(target, None)
    assert list((context / "wheels").glob("simple_local-*.whl"))
    assert "COPY wheels /srv/wheels" in (context / "Dockerfile").read_text()


def test_memory_estimate_matches_the_documented_numbers():
    kv, compute, total = spec_mod.estimate_memory(32768, 4096, limit_gi=8.0)
    assert kv == pytest.approx(3.5, abs=0.05)
    assert compute == pytest.approx(3.28, abs=0.05)
    assert total == pytest.approx(kv + compute)


def test_the_venv_is_on_path_so_runtimes_can_spawn_siblings(tmp_path):
    # vllm serve is a subprocess found via PATH; the entrypoint's absolute path
    # does nothing for it. Both base images need it.
    llm = served(tmp_path, context=8192, ubatch=4096, max_input=4096)
    for config in (None, llm):
        dockerfile = render_dockerfile(deploy(), config)
        path_line = next(line for line in dockerfile.splitlines() if "PATH=" in line)
        assert "/opt/venv/bin" in path_line
