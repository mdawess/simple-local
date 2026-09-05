import pytest

from simple_local import cli
from simple_local.deploy import azure
from simple_local.deploy.apply import PlanStale, apply_plan
from simple_local.deploy.azure import Observed
from simple_local.deploy.plan import build_plan, plan_for
from simple_local.deploy.spec import Deploy, Finding

DIGEST = "abc123def456"


def deploy(**overrides) -> Deploy:
    data = {
        "name": "simple-local-vl",
        "directory": ".",
        "container_config": "config.container.yml",
        "gpu": "NC8as-T4",
        "cpu": 4.0,
        "memory_gi": 16.0,
        "min_replicas": 1,
        "max_replicas": 3,
    }
    data.update(overrides)
    return Deploy.model_validate(data)


def observed(**overrides) -> Observed:
    data = {
        "name": "simple-local-vl",
        "resource_group": "simple-local-ca",
        "exists": True,
        "image": f"reg.azurecr.io/simple-local-vl:20260822-120000-{DIGEST}",
        "config_digest": DIGEST,
        "cpu": 4.0,
        "memory_gi": 16.0,
        "gpu": "NC8as-T4",
        "min_replicas": 1,
        "max_replicas": 3,
        "fqdn": "vl.example.io",
        "active_revisions": 1,
    }
    data.update(overrides)
    return Observed.model_validate(data)


def plan(target=None, current=None, digest=DIGEST, findings=()):
    return build_plan(target or deploy(), current or observed(), digest, list(findings))


def fields(result):
    return {c.field: c.tier for c in result.changes}


def test_matching_state_is_a_noop():
    result = plan()
    assert result.tier == "noop"
    assert result.changes == []
    assert result.restarts is False


def test_missing_app_is_a_create():
    result = plan(current=observed(exists=False))
    assert result.tier == "create"
    assert result.estimated_seconds == 720


def test_replica_change_alone_is_the_cheap_tier():
    result = plan(target=deploy(max_replicas=2))
    assert fields(result) == {"replicas": "scale"}
    assert result.tier == "scale"
    assert result.restarts is False


def test_memory_change_updates_in_place_without_a_rebuild():
    result = plan(target=deploy(memory_gi=24.0))
    assert fields(result) == {"memory": "update"}
    assert result.restarts is True
    assert result.estimated_seconds == 90


def test_a_stale_image_forces_a_rebuild():
    result = plan(digest="999999999999")
    assert fields(result) == {"image": "rebuild"}
    assert result.tier == "rebuild"


def test_an_image_without_a_digest_suffix_is_treated_as_stale():
    result = plan(current=observed(config_digest=None))
    assert result.tier == "rebuild"
    assert "predates digest tagging" in result.changes[0].effect


@pytest.mark.parametrize(
    "image, expected",
    [
        (f"reg.azurecr.io/app:20260822-120000-{DIGEST}", DIGEST),
        ("reg.azurecr.io/app:20260115-012521", None),  # tag from before digest stamping
        ("reg.azurecr.io/app:latest", None),
        ("reg.azurecr.io/app", None),
    ],
)
def test_digest_is_only_read_from_a_real_digest_suffix(image, expected):
    assert azure._digest_from_image(image) == expected


def test_the_default_workload_profile_is_not_a_gpu():
    # Azure reports "Consumption" for a CPU app; a config just says nothing.
    assert azure._workload_profile("Consumption") is None
    assert azure._workload_profile("NC8as-T4") == "NC8as-T4"


def test_a_cpu_app_matching_its_config_is_a_noop():
    cpu_only = deploy(gpu=None, cpu=4.0, memory_gi=8.0)
    current = observed(gpu=None, cpu=4.0, memory_gi=8.0)
    assert build_plan(cpu_only, current, DIGEST, []).tier == "noop"


def test_the_plan_takes_the_most_expensive_tier():
    result = plan(target=deploy(memory_gi=24.0, max_replicas=2), digest="999999999999")
    assert set(fields(result)) == {"image", "memory", "replicas"}
    assert result.tier == "rebuild"


def test_env_changes_are_an_update():
    result = plan(target=deploy(env={"VL_DEVICE": "cuda"}))
    assert fields(result) == {"env": "update"}


def test_errors_block_the_plan():
    result = plan(findings=[Finding(level="error", field="cpu", message="too big")])
    assert result.blocked is True


def test_warnings_do_not_block():
    result = plan(findings=[Finding(level="warning", field="max_replicas", message="quota")])
    assert result.blocked is False


def test_a_stopped_app_is_not_running():
    assert plan(current=observed(active_revisions=0)).running is False


def test_the_hash_moves_when_the_desired_state_moves():
    assert plan().hash != plan(target=deploy(memory_gi=24.0)).hash


def test_the_hash_moves_when_the_deployment_drifts_underneath():
    approved = plan(target=deploy(max_replicas=2))
    drifted = plan(target=deploy(max_replicas=2), current=observed(cpu=8.0))
    assert approved.hash != drifted.hash


def test_apply_refuses_a_hash_it_was_not_approved_with():
    result = plan(target=deploy(max_replicas=2))
    with pytest.raises(PlanStale, match="stale"):
        apply_plan(
            "config.yml", deploy(), observed(), result, lambda _: None, expected_hash="stale00"
        )


def test_apply_short_circuits_a_noop():
    result = plan()
    events = []
    outcome = apply_plan(
        "config.yml", deploy(), observed(), result, events.append, expected_hash=result.hash
    )
    assert outcome == {"ok": True, "app": "simple-local-vl", "tier": "noop", "url": "https://vl.example.io"}
    assert events == []


CONFIG = """
models:
  - name: embed
    kind: llm
    source: {{ provider: local, file: m.gguf }}
    inference: {{ context_length: 8192, ubatch_size: 4096, max_input_tokens: 4096 }}

deploy:
  name: simple-local-vl
  cpu: 4.0
  memory: {memory}Gi
  min_replicas: 1
  max_replicas: 3
"""


@pytest.fixture
def project(tmp_path):
    (tmp_path / "Dockerfile").write_text("FROM scratch")

    def build(memory="8.0"):
        path = tmp_path / "config.yml"
        path.write_text(CONFIG.format(memory=memory))
        return path

    return build


def test_plan_for_resolves_checks_and_diffs_in_one_call(project, monkeypatch):
    monkeypatch.chdir(project().parent)
    target, _, result = plan_for(
        "config.yml", observe_fn=lambda name, group: observed(exists=False, name=name)
    )
    assert target.name == "simple-local-vl"
    assert result.tier == "create"
    assert result.blocked is False
    assert len(result.image_digest) == 12


def test_plan_for_blocks_on_an_oversized_replica(project, monkeypatch):
    monkeypatch.chdir(project(memory="16.0").parent)
    _, _, result = plan_for(
        "config.yml", observe_fn=lambda name, group: observed(exists=False, name=name)
    )
    assert result.blocked is True
    assert [f.field for f in result.errors] == ["memory"]


def test_render_names_the_cost_and_the_approval_hash(capsys):
    result = plan(target=deploy(memory_gi=24.0))
    cli._render_plan(result)
    out = capsys.readouterr().out
    assert "update: new revision from the same image; replicas restart" in out
    assert "memory: '16Gi' -> '24Gi'" in out
    assert f"--plan-hash {result.hash}" in out


def test_render_says_blocked_instead_of_offering_apply(capsys):
    result = plan(findings=[Finding(level="error", field="cpu", message="too big")])
    cli._render_plan(result)
    out = capsys.readouterr().out
    assert "blocked" in out
    assert "--plan-hash" not in out


def test_a_resize_shows_what_it_adds_per_month():
    from simple_local.deploy.plan import cost_for
    from simple_local.deploy.pricing import Prices
    from simple_local.deploy.profiles import Catalog, Profile
    from tests.test_deploy_cost import METERS

    t4 = Profile(
        name="Consumption-GPU-NC8as-T4",
        category="Consumption-GPU-T4",
        cores=8,
        memory_gi=56,
        gpus=1,
    )
    catalog = Catalog(location="test", source="azure", profiles={t4.name: t4})
    prices = Prices(location="test", meters=METERS, source="azure")

    cost = cost_for(deploy(memory_gi=32.0), observed(memory_gi=16.0), catalog, prices)
    assert cost.before is not None
    assert cost.delta_ceiling > 0
    assert "+USD" in cost.summary()


def test_an_unchanged_config_reports_no_cost_movement():
    from simple_local.deploy.plan import cost_for
    from simple_local.deploy.pricing import Prices
    from simple_local.deploy.profiles import Catalog, Profile
    from tests.test_deploy_cost import METERS

    t4 = Profile(
        name="Consumption-GPU-NC8as-T4",
        category="Consumption-GPU-T4",
        cores=8,
        memory_gi=56,
        gpus=1,
    )
    catalog = Catalog(location="test", source="azure", profiles={t4.name: t4})
    cost = cost_for(deploy(), observed(), catalog, Prices(location="test", meters=METERS, source="azure"))
    assert cost.delta_ceiling == 0
    assert "unchanged" in cost.summary()


def test_cost_is_not_part_of_the_approval_hash():
    # Prices move on Azure's schedule; that must not invalidate an approved plan.
    assert "cost" not in plan().approval_payload()


def test_apply_refuses_to_call_a_placeholder_image_a_success():
    # Container Apps reports the revision provisioned even when the pull failed
    # and it substituted mcr.microsoft.com/k8se/quickstart. az exits 0 either way.
    from simple_local.deploy.apply import _verify_image

    result = plan(digest="aaaaaaaaaaaa")
    placeholder = observed(image="mcr.microsoft.com/k8se/quickstart:latest", config_digest=None)
    with pytest.raises(azure.AzureError, match="fell back to a placeholder"):
        _verify_image(placeholder, result)


def test_apply_accepts_the_image_it_actually_built():
    from simple_local.deploy.apply import _verify_image

    result = plan(digest=DIGEST)
    _verify_image(observed(config_digest=DIGEST), result)


def test_cheap_tiers_are_not_image_checked():
    from simple_local.deploy.apply import _verify_image

    _verify_image(observed(config_digest=None), plan(target=deploy(max_replicas=2)))
