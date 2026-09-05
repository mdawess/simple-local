import hashlib
import json
from pathlib import Path
from typing import Callable

from pydantic import BaseModel, Field

from .azure import Observed, observe
from . import pricing
from .build import image_digest
from .config import RESTARTING_TIERS, TIER_ORDER, TIER_SECONDS, TIER_SUMMARY, Tier
from .pricing import Estimate, Prices
from .spec import Deploy, Finding, check, load_served, resolve


def _rank(tier: Tier) -> int:
    return TIER_ORDER.index(tier)


class Change(BaseModel):
    field: str
    before: object | None
    after: object | None
    tier: Tier
    effect: str


class Cost(BaseModel):
    currency: str = "USD"
    before: Estimate | None = None  # what the live deployment costs today
    after: Estimate
    delta_floor: float = 0.0
    delta_ceiling: float = 0.0

    def summary(self) -> str:
        if not self.after.available:
            return "cost unknown"
        line = self.after.summary()
        if self.before is None or not self.before.available:
            return line
        if abs(self.delta_floor) < 0.01 and abs(self.delta_ceiling) < 0.01:
            return f"{line} (unchanged)"
        sign = "+" if self.delta_ceiling >= 0 else "-"
        return f"{line} ({sign}{self.currency} {abs(self.delta_ceiling):,.0f} at max)"


class Plan(BaseModel):
    version: int = 1
    app: str
    resource_group: str
    exists: bool
    running: bool
    url: str | None = None
    tier: Tier
    summary: str
    changes: list[Change] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    blocked: bool = False
    restarts: bool = False
    estimated_seconds: int = 0
    image_digest: str
    # Derived from the config, which the hash already covers, and from prices
    # that move on Azure's schedule — so it is deliberately not part of the
    # approval payload. A price change must not invalidate an approved plan.
    cost: Cost | None = None
    hash: str = ""

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.level == "error"]

    def approval_payload(self) -> dict:
        # Anything that would change what apply does belongs in here: the hash is
        # what ties the Deploy button to the plan the operator actually read.
        return {
            "version": self.version,
            "app": self.app,
            "resource_group": self.resource_group,
            "exists": self.exists,
            "tier": self.tier,
            "image_digest": self.image_digest,
            "changes": [c.model_dump() for c in self.changes],
        }


def _hash(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _scale_effect(deploy: Deploy) -> str:
    # A new replica appears in ~30s but needs ~90s more to reach full throughput,
    # so raising max_replicas is not capacity you have during a burst.
    return (
        f"replica bounds become {deploy.min_replicas}-{deploy.max_replicas}; "
        "a new replica takes about two minutes to reach full throughput"
    )


def diff(deploy: Deploy, observed: Observed, digest: str, force_rebuild: bool = False) -> list[Change]:
    if not observed.exists:
        return [
            Change(
                field="app",
                before=None,
                after=deploy.name,
                tier="create",
                effect=f"creates {deploy.name} in {deploy.resource_group}",
            )
        ]

    changes: list[Change] = []

    if force_rebuild or observed.config_digest != digest:
        if force_rebuild and observed.config_digest == digest:
            effect = "forced: the image matches the config, but the package source may have moved"
        elif observed.config_digest:
            effect = "the model directory or served config changed, so the running image is stale"
        else:
            effect = "the running image predates digest tagging, so it cannot be matched to this config"
        changes.append(
            Change(field="image", before=observed.config_digest, after=digest, tier="rebuild", effect=effect)
        )

    if observed.gpu != deploy.gpu:
        changes.append(
            Change(
                field="gpu",
                before=observed.gpu,
                after=deploy.gpu,
                tier="update",
                effect="moves the app to a different workload profile, adding it to the environment if missing",
            )
        )
    if observed.cpu != deploy.cpu:
        changes.append(
            Change(
                field="cpu",
                before=observed.cpu,
                after=deploy.cpu,
                tier="update",
                effect="resizes the replica; the image is reused",
            )
        )
    if observed.memory_gi != deploy.memory_gi:
        changes.append(
            Change(
                field="memory",
                before=f"{observed.memory_gi:g}Gi" if observed.memory_gi else None,
                after=deploy.memory_arg,
                tier="update",
                effect="resizes the replica; the image is reused",
            )
        )

    if observed.env != deploy.env:
        changes.append(
            Change(
                field="env",
                before=observed.env,
                after=deploy.env,
                tier="update",
                effect="replaces environment variables; secrets are unaffected",
            )
        )

    if (observed.min_replicas, observed.max_replicas) != (
        deploy.min_replicas,
        deploy.max_replicas,
    ):
        changes.append(
            Change(
                field="replicas",
                before=f"{observed.min_replicas}-{observed.max_replicas}",
                after=f"{deploy.min_replicas}-{deploy.max_replicas}",
                tier="scale",
                effect=_scale_effect(deploy),
            )
        )

    return changes


def build_plan(
    deploy: Deploy,
    observed: Observed,
    digest: str,
    findings: list[Finding],
    force_rebuild: bool = False,
    cost: Cost | None = None,
) -> Plan:
    changes = diff(deploy, observed, digest, force_rebuild)
    tier: Tier = max((c.tier for c in changes), key=_rank, default="noop")
    blocked = any(f.level == "error" for f in findings)

    plan = Plan(
        app=deploy.name,
        resource_group=deploy.resource_group,
        exists=observed.exists,
        running=observed.running,
        url=observed.url,
        tier=tier,
        summary=TIER_SUMMARY[tier],
        changes=changes,
        findings=findings,
        blocked=blocked,
        restarts=tier in RESTARTING_TIERS,
        estimated_seconds=TIER_SECONDS[tier],
        image_digest=digest,
        cost=cost,
    )
    plan.hash = _hash(plan.approval_payload())
    return plan


def cost_for(deploy: Deploy, observed: Observed, catalog, prices: Prices) -> Cost:
    after = pricing.estimate_for(deploy, catalog, prices)
    before = None
    if observed.exists and observed.cpu is not None and observed.memory_gi is not None:
        profile, _ = catalog.resolve(observed.gpu)
        if profile is not None:
            before = pricing.estimate(
                observed.cpu,
                observed.memory_gi,
                observed.min_replicas or 0,
                observed.max_replicas or 1,
                profile,
                prices,
            )
    return Cost(
        currency=after.currency,
        before=before,
        after=after,
        delta_floor=after.floor - (before.floor if before else 0.0),
        delta_ceiling=after.ceiling - (before.ceiling if before else 0.0),
    )


def plan_for(
    config_path: str | Path,
    observe_fn: Callable[[str, str], Observed] = observe,
    force_rebuild: bool = False,
) -> tuple[Deploy, Observed, Plan]:
    deploy = resolve(config_path)
    served = load_served(deploy)
    digest = image_digest(deploy, served.config)
    catalog = deploy.catalog(live=True)
    findings = check(deploy, served.config, served_error=served.error, catalog=catalog)
    observed = observe_fn(deploy.name, deploy.resource_group)
    cost = cost_for(deploy, observed, catalog, pricing.load(deploy.location))
    return deploy, observed, build_plan(
        deploy, observed, digest, findings, force_rebuild, cost=cost
    )
