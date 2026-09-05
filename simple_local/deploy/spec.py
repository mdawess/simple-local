from pathlib import Path
from typing import Literal, NamedTuple

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from .. import config as config_mod
from . import profiles as profiles_mod
from .config import (
    COMPUTE_BUFFER_GI_PER_4K_TOKENS,
    COMPUTE_BUFFER_REFERENCE_TOKENS,
    DEFAULT_CPU,
    DEFAULT_ENVIRONMENT,
    DEFAULT_LOCATION,
    DEFAULT_MAX_REPLICAS,
    DEFAULT_MEMORY_GI,
    DEFAULT_MIN_REPLICAS,
    CANDIDATE_LABEL,
    DEFAULT_RESOURCE_GROUP,
    DEFAULT_TARGET_PORT,
    DEFAULT_WORKSPACE,
    TAG_MANAGED_BY,
    TAG_PROJECT,
    LABEL_SEPARATOR,
    LIVE_LABEL,
    MULTIPLE_REVISION,
    DEFAULT_UBATCH_TOKENS,
    KV_KIB_PER_TOKEN,
    LLAMA_BASE_IMAGE,
    MEMORY_HEADROOM_WARN_RATIO,
    PYTHON_BASE_IMAGE,
    SIZING_SOURCE,
)
from .profiles import Catalog, Profile
from .units import parse_memory_gi


class Sizing(BaseModel):
    """The llama.cpp memory model — see config.py for where the numbers came
    from. Overridable because they are measurements of one model family, not
    constants: a different architecture has different numbers, and a guardrail
    built on a silently wrong constant is worse than no guardrail. `source` is
    there so an override says where its numbers came from.
    """

    kv_kib_per_token: float = KV_KIB_PER_TOKEN
    compute_buffer_gi_per_4k_tokens: float = COMPUTE_BUFFER_GI_PER_4K_TOKENS
    source: str = SIZING_SOURCE


class Build(BaseModel):
    base_image: str | None = None  # defaults to the runtime the served config needs
    system_packages: list[str] = Field(default_factory=list)
    requirements: list[str] = Field(default_factory=list)  # pip installs beyond simple-local
    extras: list[str] = Field(default_factory=list)  # simple-local extras, e.g. ["vl"]
    env: dict[str, str] = Field(default_factory=dict)  # baked into the image
    prefetch: bool = True  # cache weights at build time so a cold start is model-load only
    package: str | None = None  # pip spec for simple-local; None stages a locally built wheel

    @field_validator("env", mode="before")
    @classmethod
    def _env(cls, v):
        return {} if v is None else {str(k): str(val) for k, val in v.items()}


class Domain(BaseModel):
    """Where a route lives. Either way the app name is a public hostname, so name
    apps after the route they serve rather than the model serving it.

    `managed` binds the hostname to this app and lets Azure issue and renew a
    free certificate. Two DNS records per app, and they can live in any zone —
    no delegation needed.

    `wildcard` sets a DNS suffix on the environment instead, so every app in it
    serves at <name>.<zone> with no per-app DNS at all. That needs a wildcard
    certificate, which Azure will not issue: you supply and rotate one every 90
    days. Worth it once something is creating models without a human.
    """

    zone: str  # e.g. model.aeriumhq.com
    certificate: Literal["managed", "wildcard"] = "managed"
    zone_resource_group: str | None = None  # only used when Azure DNS holds the zone
    hostname: str | None = None  # label under the zone; defaults to the app name
    # The zone a record is actually typed into, which is not the same thing as
    # the zone the route lives under. Without delegation the registrar holds
    # aeriumhq.com, so a route at text-embed.model.aeriumhq.com is a record named
    # `text-embed.model` there — getting this wrong is the easiest way to add a
    # record that silently never resolves.
    registrar_zone: str | None = None

    @property
    def managed(self) -> bool:
        return self.certificate == "managed"

    @property
    def record_zone(self) -> str:
        if self.registrar_zone:
            return self.registrar_zone
        labels = self.zone.split(".")
        return ".".join(labels[-2:]) if len(labels) > 2 else self.zone

    def record_name(self, fqdn: str) -> str:
        suffix = f".{self.record_zone}"
        return fqdn[: -len(suffix)] if fqdn.endswith(suffix) else fqdn


class Revisions(BaseModel):
    mode: Literal["Single", "Multiple"] = MULTIPLE_REVISION
    live_label: str = LIVE_LABEL
    candidate_label: str = CANDIDATE_LABEL

    @property
    def multiple(self) -> bool:
        return self.mode == MULTIPLE_REVISION


class Deploy(BaseModel):
    name: str
    directory: str  # build context: the model directory the config lives in
    workspace: str = DEFAULT_WORKSPACE  # "shared", or a customer id when siloed
    hf_name: str | None = None  # what the weights were pulled from, for the grid
    modality: Literal["embeddings", "vision-language", "text"] | None = None
    version: str | None = None
    container_config: str
    resource_group: str = DEFAULT_RESOURCE_GROUP
    location: str = DEFAULT_LOCATION
    environment: str = DEFAULT_ENVIRONMENT
    registry: str = ""
    target_port: int = DEFAULT_TARGET_PORT
    gpu: str | None = None
    cpu: float = DEFAULT_CPU["cpu"]
    memory_gi: float = DEFAULT_MEMORY_GI["cpu"]
    min_replicas: int = DEFAULT_MIN_REPLICAS
    max_replicas: int = DEFAULT_MAX_REPLICAS["cpu"]
    env: dict[str, str] = Field(default_factory=dict)
    build: Build = Field(default_factory=Build)
    sizing: Sizing = Field(default_factory=Sizing)
    domain: Domain | None = None
    revisions: Revisions = Field(default_factory=Revisions)
    # Adds hardware Azure has not surfaced yet, corrects a size, or records a
    # quota no API will tell you. Keyed by profile name.
    profiles: dict[str, Profile] = Field(default_factory=dict)
    # Replica ceilings per workload profile. Azure exposes no quota API for
    # Container Apps, and past the limit it refuses the pod while the app keeps
    # reporting healthy — so this is worth recording even though nothing can
    # verify it.
    quotas: dict[str, int] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _gpu_flavoured_defaults(cls, data):
        if not isinstance(data, dict):
            return data
        data = dict(data)
        if "memory" in data and "memory_gi" not in data:
            data["memory_gi"] = data.pop("memory")
        if isinstance(data.get("profiles"), dict):
            data["profiles"] = {
                name: {"name": name, **entry} if isinstance(entry, dict) and "name" not in entry else entry
                for name, entry in data["profiles"].items()
            }
        kind = "gpu" if data.get("gpu") else "cpu"
        for key, table in (
            ("cpu", DEFAULT_CPU),
            ("memory_gi", DEFAULT_MEMORY_GI),
            ("max_replicas", DEFAULT_MAX_REPLICAS),
        ):
            if data.get(key) is None:
                data[key] = table[kind]
        if data.get("min_replicas") is None:
            data["min_replicas"] = DEFAULT_MIN_REPLICAS
        return data

    @field_validator("memory_gi", mode="before")
    @classmethod
    def _memory(cls, v):
        return parse_memory_gi(v)

    @field_validator("env", mode="before")
    @classmethod
    def _env(cls, v):
        return {} if v is None else {str(k): str(val) for k, val in v.items()}

    @property
    def memory_arg(self) -> str:
        return f"{self.memory_gi:g}Gi"

    @property
    def tags(self) -> dict[str, str]:
        """Applied to every resource created for this model. Cost Management
        groups by these, so a resource created without them is spend nobody can
        attribute later."""
        tags = {
            "project": TAG_PROJECT,
            "managed-by": TAG_MANAGED_BY,
            "workspace": self.workspace,
            "model": self.name,
        }
        # Identity the console reads back from `list`, so a row survives without
        # anyone holding a name-to-model mapping on the side.
        if self.hf_name:
            tags["hf-name"] = self.hf_name
        if self.modality:
            tags["modality"] = self.modality
        return tags

    def resolved_modality(self, served: config_mod.Config | None) -> str | None:
        """Declared wins; otherwise inferred from what the config serves, since
        the console renders a different test panel per modality."""
        if self.modality:
            return self.modality
        if served is None:
            return None
        for model in served.models:
            if model.kind == "custom":
                return "vision-language"
            if model.embeddings:
                return "embeddings"
        return "text" if served.models else None

    @property
    def dns_label(self) -> str:
        return (self.domain.hostname or self.name) if self.domain else self.name

    @property
    def route(self) -> str | None:
        """The stable public hostname. Callers depend on this, not on which
        revision or model is behind it."""
        return f"{self.dns_label}.{self.domain.zone}" if self.domain else None

    def label_route(self, label: str) -> str | None:
        if self.domain is None:
            return None
        return f"{self.dns_label}{LABEL_SEPARATOR}{label}.{self.domain.zone}"

    @property
    def dns_zone_group(self) -> str:
        if self.domain and self.domain.zone_resource_group:
            return self.domain.zone_resource_group
        return self.resource_group

    def catalog(self, live: bool = True) -> Catalog:
        return profiles_mod.load(self.location, self.profiles, self.quotas, live=live)

    def profile(self, live: bool = True) -> tuple[Profile | None, str | None]:
        return self.catalog(live=live).resolve(self.gpu)

    @property
    def gpu_profile_type(self) -> str | None:
        if self.gpu is None:
            return None
        _, azure_name = self.profile()
        return azure_name


def resolve(config_path: str | Path) -> Deploy:
    """A model directory is the unit of deployment: the config names the models,
    the directory beside it holds whatever else the image needs."""
    path = Path(config_path).resolve()
    data = yaml.safe_load(path.read_text()) or {}
    block = dict(data.get("deploy") or {})

    directory = path.parent
    block["directory"] = str(directory)
    if block.get("container_config") is None:
        sibling = directory / "config.container.yml"
        block["container_config"] = str(sibling if sibling.exists() else path)
    if block.get("name") is None:
        block["name"] = f"simple-local-{directory.name}"
    return Deploy.model_validate(block)


class Finding(BaseModel):
    level: Literal["error", "warning"]
    field: str
    message: str


class MemoryEstimate(BaseModel):
    model: str
    kv_gi: float
    compute_gi: float
    weights_gi: float
    total_gi: float
    limit_gi: float
    fits: bool
    counts_weights: bool


def estimate_memory(
    context_length: int,
    ubatch_tokens: int,
    limit_gi: float,
    weights_gi: float = 0.0,
    sizing: "Sizing | None" = None,
) -> tuple[float, float, float]:
    sizing = sizing or Sizing()
    kv_gi = context_length * sizing.kv_kib_per_token / (1024 * 1024)
    compute_gi = (
        sizing.compute_buffer_gi_per_4k_tokens * ubatch_tokens / COMPUTE_BUFFER_REFERENCE_TOKENS
    )
    return kv_gi, compute_gi, kv_gi + compute_gi + weights_gi


def llm_memory_estimates(
    deploy: Deploy, served: config_mod.Config, weights_gi: dict[str, float] | None = None
) -> list[MemoryEstimate]:
    weights_gi = weights_gi or {}
    estimates = []
    for spec in served.models:
        if spec.kind != "llm":
            continue
        inference = spec.inference
        ubatch = inference.ubatch_size or inference.max_input_tokens
        if ubatch is None:
            ubatch = min(inference.context_length, DEFAULT_UBATCH_TOKENS)
        weights = weights_gi.get(spec.name, 0.0)
        kv, compute, total = estimate_memory(
            inference.context_length, ubatch, deploy.memory_gi, weights, deploy.sizing
        )
        estimates.append(
            MemoryEstimate(
                model=spec.name,
                kv_gi=round(kv, 2),
                compute_gi=round(compute, 2),
                weights_gi=round(weights, 2),
                total_gi=round(total, 2),
                limit_gi=deploy.memory_gi,
                fits=total <= deploy.memory_gi,
                counts_weights=spec.name in weights_gi,
            )
        )
    return estimates


def _profile_findings(deploy: Deploy, catalog: Catalog) -> list[Finding]:
    """One rule for every profile: a replica may request less than the node
    provides — vl-embeddings runs 4 vCPU / 16Gi on a T4 node of 8 / 56 — but
    never more. Consumption is a profile like any other, so its 4 vCPU / 8GiB
    ceiling needs no special case."""
    findings: list[Finding] = []

    if not catalog.profiles:
        # Never silent: "not verified" must not read as "fine".
        return [
            Finding(
                level="warning",
                field="gpu",
                message=(
                    f"replica sizing not checked: no workload profile catalog for {deploy.location}. "
                    f"Run `simple-local profiles -l {deploy.location}` with Azure access to cache one."
                ),
            )
        ]
    if catalog.stale:
        findings.append(
            Finding(
                level="warning",
                field="gpu",
                message=(
                    f"workload profile catalog is {catalog.age_days:.0f} days old; "
                    f"refresh with `simple-local profiles -l {deploy.location}`"
                ),
            )
        )

    profile, azure_name = catalog.resolve(deploy.gpu)
    if profile is None:
        gpus = ", ".join(p.name for p in catalog.gpus()) or "none"
        return findings + [
            Finding(
                level="error",
                field="gpu",
                message=(
                    f"unknown workload profile '{deploy.gpu}' in {deploy.location} "
                    f"({catalog.provenance}). GPU profiles available: {gpus}. "
                    "Add it under deploy.profiles if Azure offers it and the catalog is behind."
                ),
            )
        ]

    if deploy.cpu > profile.cores:
        findings.append(
            Finding(
                level="error",
                field="cpu",
                message=f"{azure_name} provides {profile.cores:g} vCPU per replica, requested {deploy.cpu:g}",
            )
        )
    if deploy.memory_gi > profile.memory_gi:
        findings.append(
            Finding(
                level="error",
                field="memory",
                message=f"{azure_name} provides {profile.memory_gi:g}Gi per replica, requested {deploy.memory_arg}",
            )
        )
    # Past quota the platform refuses the pod and the app keeps reporting
    # healthy on the replicas it already has, so this is worth saying out loud
    # even though it is only a ceiling.
    if profile.quota and deploy.max_replicas > profile.quota:
        findings.append(
            Finding(
                level="warning",
                field="max_replicas",
                message=(
                    f"{azure_name} quota is {profile.quota} per subscription; "
                    f"max_replicas {deploy.max_replicas} is unreachable without an increase"
                ),
            )
        )
    return findings


def check(
    deploy: Deploy,
    served: config_mod.Config | None = None,
    weights_gi: dict[str, float] | None = None,
    served_error: str | None = None,
    catalog: Catalog | None = None,
) -> list[Finding]:
    # Offline unless the caller hands over a live catalog: `validate` is the CI
    # gate, and a gate that needs an Azure login (or waits on one to time out)
    # is not a gate. `plan` is already talking to Azure, so it passes one in.
    findings: list[Finding] = []
    catalog = deploy.catalog(live=False) if catalog is None else catalog

    if served_error:
        findings.append(Finding(level="error", field="container_config", message=served_error))

    if deploy.max_replicas < 1:
        findings.append(
            Finding(
                level="error",
                field="max_replicas",
                message="max_replicas has a floor of 1; use --stop to take a deployment out of service",
            )
        )
    if deploy.min_replicas > deploy.max_replicas:
        findings.append(
            Finding(
                level="error",
                field="min_replicas",
                message=f"min_replicas {deploy.min_replicas} exceeds max_replicas {deploy.max_replicas}",
            )
        )

    findings.extend(_profile_findings(deploy, catalog))

    if served is not None:
        findings.extend(_served_findings(deploy, served, weights_gi))
    return findings


def _served_findings(
    deploy: Deploy, served: config_mod.Config, weights_gi: dict[str, float] | None
) -> list[Finding]:
    findings: list[Finding] = []
    for spec in served.models:
        if spec.kind != "llm":
            continue
        inference = spec.inference
        # An embedding input longer than one ubatch hangs rather than erroring.
        if (
            inference.ubatch_size is not None
            and inference.max_input_tokens is not None
            and inference.ubatch_size < inference.max_input_tokens
        ):
            findings.append(
                Finding(
                    level="error",
                    field=f"models.{spec.name}.inference.ubatch_size",
                    message=(
                        f"ubatch_size {inference.ubatch_size} is below max_input_tokens "
                        f"{inference.max_input_tokens}; an input past one ubatch hangs instead of erroring"
                    ),
                )
            )

    for estimate in llm_memory_estimates(deploy, served, weights_gi):
        unaccounted = "" if estimate.counts_weights else ", excluding model weights"
        detail = (
            f"KV {estimate.kv_gi:g}Gi + compute {estimate.compute_gi:g}Gi"
            + (f" + weights {estimate.weights_gi:g}Gi" if estimate.counts_weights else "")
            + f" = {estimate.total_gi:g}Gi against a {estimate.limit_gi:g}Gi limit{unaccounted}"
        )
        if not estimate.fits:
            findings.append(
                Finding(
                    level="error",
                    field=f"models.{estimate.model}.inference.context_length",
                    message=(
                        f"estimated peak memory does not fit: {detail}. "
                        "The server loads fine and then dies on the first large input."
                    ),
                )
            )
        elif estimate.total_gi > MEMORY_HEADROOM_WARN_RATIO * estimate.limit_gi:
            findings.append(
                Finding(
                    level="warning",
                    field=f"models.{estimate.model}.inference.context_length",
                    message=f"little memory headroom: {detail}",
                )
            )
    return findings


class Served(NamedTuple):
    config: config_mod.Config | None
    error: str | None


def load_served(deploy: Deploy) -> Served:
    """A container config that will not load is a blocking problem, not a reason
    to quietly skip every check that depends on reading it."""
    target = Path(deploy.container_config)
    if not target.exists():
        return Served(None, f"container config not found: {target}")
    try:
        return Served(config_mod.load(str(target)), None)
    except Exception as e:
        return Served(None, f"{target.name} does not load: {e}")


def default_base_image(served: config_mod.Config | None) -> str:
    if served and any(spec.kind == "llm" for spec in served.models):
        return LLAMA_BASE_IMAGE
    return PYTHON_BASE_IMAGE
