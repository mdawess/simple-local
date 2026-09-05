import shutil
from pathlib import Path
from typing import Callable

from . import azure, operations
from .azure import AzureError, Observed
from .build import stage
from .config import BUILDING_TIERS, DEFAULT_WORKLOAD_PROFILE
from .plan import Plan
from .spec import Deploy, load_served

Emit = Callable[[dict], None]


class PlanStale(RuntimeError):
    pass


def _update_in_place(deploy: Deploy, observed: Observed, plan: Plan, emit: Emit) -> None:
    fields = {c.field for c in plan.changes}
    args = ["containerapp", "update", "--name", deploy.name, "--resource-group", deploy.resource_group]

    if "gpu" in fields:
        if deploy.gpu:
            operations.ensure_workload_profile(deploy, emit)
            args += ["--workload-profile-name", deploy.gpu]
        else:
            args += ["--workload-profile-name", DEFAULT_WORKLOAD_PROFILE]
    if "cpu" in fields or "memory" in fields:
        args += ["--cpu", str(deploy.cpu), "--memory", deploy.memory_arg]
    if "replicas" in fields:
        args += ["--min-replicas", str(deploy.min_replicas), "--max-replicas", str(deploy.max_replicas)]
    if "env" in fields:
        for key, value in deploy.env.items():
            args += ["--set-env-vars", f"{key}={value}"]
        for key in observed.env.keys() - deploy.env.keys():
            args += ["--remove-env-vars", key]

    emit({"event": "step", "name": plan.tier, "status": "start"})
    azure.run([*args, "--output", "none"])
    emit({"event": "step", "name": plan.tier, "status": "ok"})


def _build_and_deploy(deploy: Deploy, plan: Plan, emit: Emit) -> str | None:
    registry = operations.ensure_infrastructure(deploy, emit)
    context = stage(deploy, load_served(deploy).config)
    try:
        image = operations.build_image(
            deploy, registry, context, operations.image_tag(plan.image_digest), emit
        )
    finally:
        shutil.rmtree(context, ignore_errors=True)

    if plan.exists:
        operations.update_app(deploy, image, emit, registry=registry)
        _label_new_revision(deploy, emit)
        return None
    key = operations.create_app(deploy, registry, image, emit)
    _label_new_revision(deploy, emit)
    return key


def _label_new_revision(deploy: Deploy, emit: Emit) -> None:
    """In multiple-revision mode a new image takes no traffic until it is
    promoted, so it gets the candidate label and the live label stays put. The
    route keeps serving the model it was serving."""
    if not deploy.revisions.multiple:
        return
    latest = operations.latest_revision(deploy)
    if latest is None:
        return
    labels = deploy.revisions
    live = operations.labelled(deploy, labels.live_label)
    label = labels.live_label if live is None else labels.candidate_label
    if live == latest:
        return
    operations.set_label(deploy, label, latest, 100 if live is None else 0, emit)


def _verify_image(observed: Observed, plan: Plan) -> None:
    """Container Apps reports a revision as provisioned even when it could not
    pull the image and quietly substituted its own placeholder, so `az` exiting
    zero proves nothing. The digest in the tag is what proves it."""
    if plan.tier not in ("create", "rebuild"):
        return
    if observed.config_digest == plan.image_digest:
        return
    raise AzureError(
        f"{observed.name} is running {observed.image or 'no image'}, not the image just built "
        f"({plan.image_digest}). Container Apps could not pull from the registry and fell back "
        "to a placeholder — check `az containerapp show --query properties.configuration.registries`"
    )


def apply_plan(
    config_path: str | Path,
    deploy: Deploy,
    observed: Observed,
    plan: Plan,
    emit: Emit,
    expected_hash: str | None = None,
) -> dict:
    """Runs an approved plan. `expected_hash` is what the operator clicked Deploy
    on; a mismatch means the config or the deployment moved underneath them and
    the plan they read is no longer the plan that would run."""
    if expected_hash is not None and expected_hash != plan.hash:
        raise PlanStale(
            f"plan is stale: approved {expected_hash}, current {plan.hash} — re-plan and review again"
        )
    if plan.blocked:
        raise AzureError("plan has blocking errors; refusing to apply")

    if plan.tier == "noop":
        return {"ok": True, "app": deploy.name, "tier": "noop", "url": plan.url}

    api_key = None
    if plan.tier in BUILDING_TIERS:
        api_key = _build_and_deploy(deploy, plan, emit)
    else:
        _update_in_place(deploy, observed, plan, emit)

    after = azure.observe(deploy.name, deploy.resource_group)
    _verify_image(after, plan)
    result = {
        "route": deploy.route,
        "ok": True,
        "app": deploy.name,
        "tier": plan.tier,
        "url": after.url,
        "revision": after.latest_revision,
        "replicas": {"min": after.min_replicas, "max": after.max_replicas},
    }
    if api_key:
        result["api_key"] = api_key
        result["api_key_note"] = (
            f"stored as the '{operations.MANAGED_SECRET}' secret — "
            "az containerapp secret show reads it back"
        )
    return result
