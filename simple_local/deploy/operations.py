import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from . import azure
from .azure import AzureError
from .config import DNS_TTL, DOMAIN_VERIFICATION_PREFIX, MANAGED_SECRET
from .spec import Deploy

Emit = Callable[[dict], None]


def _step(emit: Emit, name: str, status: str, **extra) -> None:
    emit({"event": "step", "name": name, "status": status, **extra})


def image_tag(digest: str) -> str:
    # A unique tag per build: pushing over a tag the app already runs produces no
    # new revision, so the deploy would appear to succeed and change nothing.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{digest}"


def find_registry(deploy: Deploy) -> str | None:
    if deploy.registry:
        return deploy.registry
    found = azure.run(
        ["acr", "list", "-g", deploy.resource_group, "--query", "[0].name", "-o", "tsv"]
    ).strip()
    return found or None


def ensure_infrastructure(deploy: Deploy, emit: Emit) -> str:
    try:
        azure.run(["group", "show", "-n", deploy.resource_group, "--output", "none"])
    except AzureError:
        _step(emit, "resource-group", "start", name=deploy.resource_group)
        azure.run(
            ["group", "create", "-n", deploy.resource_group, "-l", deploy.location,
             *_tag_args(deploy), "--output", "none"]
        )
        _step(emit, "resource-group", "ok", name=deploy.resource_group)

    registry = find_registry(deploy)
    if registry is None:
        registry = f"sl{secrets.token_hex(5)}"
        _step(emit, "registry", "start", name=registry)
        azure.run(
            [
                "acr", "create", "-g", deploy.resource_group, "-n", registry,
                "--sku", "Basic", "--admin-enabled", "true",
                *_tag_args(deploy), "--output", "none",
            ]
        )
        _step(emit, "registry", "ok", name=registry)

    try:
        azure.run(
            ["containerapp", "env", "show", "-n", deploy.environment,
             "-g", deploy.resource_group, "--output", "none"]
        )
    except AzureError:
        _step(emit, "environment", "start", name=deploy.environment)
        azure.run(
            ["containerapp", "env", "create", "-n", deploy.environment,
             "-g", deploy.resource_group, "-l", deploy.location,
             *_tag_args(deploy), "--output", "none"]
        )
        _step(emit, "environment", "ok", name=deploy.environment)

    ensure_workload_profile(deploy, emit)
    return registry


def ensure_workload_profile(deploy: Deploy, emit: Emit) -> None:
    # GPU replicas need a workload profile on the environment; the default
    # Consumption profile has no GPU to hand out.
    if deploy.gpu is None:
        return
    existing = azure.run(
        [
            "containerapp", "env", "workload-profile", "list",
            "-n", deploy.environment, "-g", deploy.resource_group,
            "--query", f"[?name=='{deploy.gpu}'] | length(@)", "-o", "tsv",
        ]
    ).strip()
    if existing not in ("", "0"):
        return
    _step(emit, "workload-profile", "start", profile=deploy.gpu)
    azure.run(
        [
            "containerapp", "env", "workload-profile", "add",
            "--name", deploy.environment, "--resource-group", deploy.resource_group,
            "--workload-profile-name", deploy.gpu,
            "--workload-profile-type", deploy.gpu_profile_type,
            "--output", "none",
        ]
    )
    _step(emit, "workload-profile", "ok", profile=deploy.gpu)


def build_image(deploy: Deploy, registry: str, context: Path, tag: str, emit: Emit) -> str:
    image = f"{registry}.azurecr.io/{deploy.name}:{tag}"
    _step(emit, "build", "start", image=image, context=str(context))
    azure.stream(
        [
            "acr", "build", "--registry", registry, "--platform", "linux/amd64",
            "-t", f"{deploy.name}:{tag}", "-t", f"{deploy.name}:latest",
            str(context),
        ],
        lambda line: emit({"event": "output", "line": line}),
    )
    _step(emit, "build", "ok", image=image)
    return image


def registry_credentials(registry: str) -> tuple[str, str]:
    """Admin credentials for the ACR. Passing `--registry-server` alone makes the
    CLI attempt managed-identity auth, which fails without a role assignment —
    and rather than erroring, Container Apps silently substitutes its own
    placeholder image and reports the revision as provisioned."""
    creds = azure.json_out(["acr", "credential", "show", "-n", registry])
    return creds["username"], creds["passwords"][0]["value"]


def _tag_args(deploy: Deploy) -> list[str]:
    return ["--tags", *(f"{key}={value}" for key, value in deploy.tags.items())]


def _env_args(deploy: Deploy) -> list[str]:
    pairs = [f"SIMPLE_LOCAL_API_KEY=secretref:{MANAGED_SECRET}"]
    pairs += [f"{key}={value}" for key, value in deploy.env.items()]
    return pairs


def create_app(deploy: Deploy, registry: str, image: str, emit: Emit) -> str:
    key = secrets.token_hex(16)
    username, password = registry_credentials(registry)
    _step(emit, "create", "start", app=deploy.name)
    args = [
        "containerapp", "create",
        "--name", deploy.name, "--resource-group", deploy.resource_group,
        "--environment", deploy.environment,
        "--image", image,
        "--registry-server", f"{registry}.azurecr.io",
        "--registry-username", username, "--registry-password", password,
        "--target-port", str(deploy.target_port), "--ingress", "external",
        "--cpu", str(deploy.cpu), "--memory", deploy.memory_arg,
        "--min-replicas", str(deploy.min_replicas),
        "--max-replicas", str(deploy.max_replicas),
        "--secrets", f"{MANAGED_SECRET}={key}",
        "--env-vars", *_env_args(deploy),
        *_tag_args(deploy),
        "--output", "none",
    ]
    if deploy.gpu:
        args += ["--workload-profile-name", deploy.gpu]
    if deploy.revisions.multiple:
        args += ["--revisions-mode", "multiple"]
    azure.run(args)
    _step(emit, "create", "ok", app=deploy.name)
    return key


def update_app(deploy: Deploy, image: str | None, emit: Emit, registry: str | None = None) -> None:
    _step(emit, "update", "start", app=deploy.name)
    args = [
        "containerapp", "update",
        "--name", deploy.name, "--resource-group", deploy.resource_group,
        "--cpu", str(deploy.cpu), "--memory", deploy.memory_arg,
        "--min-replicas", str(deploy.min_replicas),
        "--max-replicas", str(deploy.max_replicas),
        "--set-env-vars", *_env_args(deploy),
        "--output", "none",
    ]
    if image:
        args += ["--image", image]
    if deploy.gpu:
        args += ["--workload-profile-name", deploy.gpu]
    azure.run(args)
    if image and registry:
        username, password = registry_credentials(registry)
        azure.run(
            ["containerapp", "registry", "set", "-n", deploy.name, "-g", deploy.resource_group,
             "--server", f"{registry}.azurecr.io", "--username", username,
             "--password", password, "--output", "none"]
        )
    _step(emit, "update", "ok", app=deploy.name)


def latest_revision(deploy: Deploy) -> str | None:
    return (
        azure.run(
            ["containerapp", "show", "-n", deploy.name, "-g", deploy.resource_group,
             "--query", "properties.latestRevisionName", "-o", "tsv"]
        ).strip()
        or None
    )


def stop(deploy: Deploy) -> str:
    # Deactivating the revision is the only real stop: max-replicas has a floor
    # of 1, so scaling down cannot get to zero capacity on its own.
    revision = latest_revision(deploy)
    if revision is None:
        raise AzureError(f"{deploy.name} has no revision to deactivate")
    azure.run(
        ["containerapp", "revision", "deactivate", "--name", deploy.name,
         "--resource-group", deploy.resource_group, "--revision", revision, "--output", "none"]
    )
    return revision


def start(deploy: Deploy) -> str:
    revision = latest_revision(deploy)
    if revision is None:
        raise AzureError(f"{deploy.name} has no revision to activate")
    active = azure.run(
        ["containerapp", "revision", "show", "-n", deploy.name, "-g", deploy.resource_group,
         "--revision", revision, "--query", "properties.active", "-o", "tsv"]
    ).strip()
    # Activating an already-active revision is an error, so on a running app
    # this is just how replica counts get pushed without a rebuild.
    if active != "true":
        azure.run(
            ["containerapp", "revision", "activate", "--name", deploy.name,
             "--resource-group", deploy.resource_group, "--revision", revision, "--output", "none"]
        )
    azure.run(
        ["containerapp", "update", "--name", deploy.name, "--resource-group", deploy.resource_group,
         "--min-replicas", str(deploy.min_replicas),
         "--max-replicas", str(deploy.max_replicas), "--output", "none"]
    )
    return revision


def teardown(deploy: Deploy) -> None:
    azure.run(
        ["containerapp", "delete", "--name", deploy.name,
         "--resource-group", deploy.resource_group, "--yes", "--output", "none"]
    )


def list_apps(resource_group: str) -> list[dict]:
    """One row per deployed model. Identity comes from the resource tags rather
    than from parsing names, so it survives a rename and matches exactly what
    Cost Management groups spend by."""
    apps = azure.json_out(["containerapp", "list", "-g", resource_group])
    rows = []
    for app in apps:
        name = app.get("name")
        properties = app.get("properties") or {}
        template = properties.get("template") or {}
        scale = template.get("scale") or {}
        ingress = (properties.get("configuration") or {}).get("ingress") or {}
        tags = app.get("tags") or {}
        custom = [d.get("name") for d in (ingress.get("customDomains") or []) if d.get("name")]
        fqdn = ingress.get("fqdn")

        rows.append(
            {
                "id": name,
                "workspace": tags.get("workspace"),
                "hf_name": tags.get("hf-name"),
                "modality": tags.get("modality"),
                "azure_region": app.get("location"),
                "resource_group": resource_group,
                "endpoint": f"https://{custom[0]}" if custom else (f"https://{fqdn}" if fqdn else None),
                "status": _status(properties, scale),
                "replicas": {
                    "current": sum(r.get("properties", {}).get("replicas") or 0
                                   for r in _active_revisions(name, resource_group)),
                    "min": scale.get("minReplicas"),
                    "max": scale.get("maxReplicas"),
                },
                "last_updated_at": properties.get("latestRevisionName") and app.get("systemData", {}).get("lastModifiedAt"),
            }
        )
    return rows


def _active_revisions(name: str, resource_group: str) -> list[dict]:
    try:
        rows = azure.json_out(
            ["containerapp", "revision", "list", "-n", name, "-g", resource_group, "--all"]
        )
    except AzureError:
        return []
    return [r for r in rows if (r.get("properties") or {}).get("active")]


def _status(properties: dict, scale: dict) -> str:
    """An app whose revisions are all deactivated still reports runningStatus
    "Running", so stopped has to come from the revisions. Scaled to zero is a
    different thing from stopped and the console should not conflate them."""
    if properties.get("runningStatus") == "Stopped":
        return "stopped"
    if (scale.get("minReplicas") or 0) == 0:
        return "idle"
    return "healthy"


# --- Revisions ------------------------------------------------------------


def revision_mode(deploy: Deploy) -> str:
    return azure.run(
        ["containerapp", "show", "-n", deploy.name, "-g", deploy.resource_group,
         "--query", "properties.configuration.activeRevisionsMode", "-o", "tsv"]
    ).strip()


def set_revision_mode(deploy: Deploy, mode: str, emit: Emit) -> None:
    _step(emit, "revision-mode", "start", mode=mode)
    azure.run(
        ["containerapp", "revision", "set-mode", "-n", deploy.name,
         "-g", deploy.resource_group, "--mode", mode.lower(), "--output", "none"]
    )
    _step(emit, "revision-mode", "ok", mode=mode)


def revisions(deploy: Deploy) -> list[dict]:
    """Every revision with the traffic it takes and the labels pointing at it.
    Weights and labels live on the app's ingress, not on the revision, so the
    two have to be stitched together."""
    rows = azure.json_out(
        ["containerapp", "revision", "list", "-n", deploy.name, "-g", deploy.resource_group, "--all"]
    )
    traffic = azure.json_out(
        ["containerapp", "show", "-n", deploy.name, "-g", deploy.resource_group,
         "--query", "properties.configuration.ingress.traffic"]
    ) or []
    by_revision = {t.get("revisionName"): t for t in traffic if t.get("revisionName")}
    latest = next((t for t in traffic if t.get("latestRevision")), None)

    out = []
    for row in rows:
        properties = row.get("properties") or {}
        name = row.get("name")
        share = by_revision.get(name) or (latest if properties.get("active") and latest else None)
        containers = (properties.get("template") or {}).get("containers") or [{}]
        out.append(
            {
                "name": name,
                "active": bool(properties.get("active")),
                "created": properties.get("createdTime"),
                "image": containers[0].get("image"),
                "digest": _digest(containers[0].get("image")),
                "replicas": properties.get("replicas"),
                "weight": (share or {}).get("weight", 0),
                "label": (share or {}).get("label"),
                "healthy": properties.get("healthState"),
            }
        )
    return sorted(out, key=lambda r: r.get("created") or "", reverse=True)


def _digest(image: str | None) -> str | None:
    from .azure import _digest_from_image

    return _digest_from_image(image)


def labelled(deploy: Deploy, label: str) -> str | None:
    for row in revisions(deploy):
        if row["label"] == label:
            return row["name"]
    return None


def set_label(deploy: Deploy, label: str, revision: str, weight: int, emit: Emit) -> None:
    _step(emit, "label", "start", label=label, revision=revision, weight=weight)
    azure.run(
        ["containerapp", "revision", "label", "add", "-n", deploy.name,
         "-g", deploy.resource_group, "--label", label, "--revision", revision,
         "--yes", "--output", "none"]
    )
    _step(emit, "label", "ok", label=label, revision=revision)


def swap_labels(deploy: Deploy, source: str, target: str, emit: Emit) -> None:
    """Atomic: the route keeps serving throughout, and swapping back is the
    same call with the arguments reversed."""
    _step(emit, "swap", "start", source=source, target=target)
    azure.run(
        ["containerapp", "revision", "label", "swap", "-n", deploy.name,
         "-g", deploy.resource_group, "--source", source, "--target", target, "--output", "none"]
    )
    _step(emit, "swap", "ok", source=source, target=target)


def set_traffic(deploy: Deploy, weights: dict[str, int], emit: Emit) -> None:
    args = [f"{name}={weight}" for name, weight in weights.items()]
    _step(emit, "traffic", "start", weights=weights)
    azure.run(
        ["containerapp", "ingress", "traffic", "set", "-n", deploy.name,
         "-g", deploy.resource_group, "--label-weight", *args, "--output", "none"]
    )
    _step(emit, "traffic", "ok", weights=weights)


# --- Custom domain --------------------------------------------------------


def verification_id(deploy: Deploy) -> str | None:
    """The app's ownership token — but it is scoped to the subscription, not the
    app, so the environment's is the same value. Falling back to it means the
    DNS records can be prepared before the app exists, which is the order you
    actually want: records first, then deploy, then bind."""
    try:
        found = azure.run(
            ["containerapp", "show", "-n", deploy.name, "-g", deploy.resource_group,
             "--query", "properties.customDomainVerificationId", "-o", "tsv"]
        ).strip()
        if found:
            return found
    except AzureError:
        pass
    return environment_verification_id(deploy)


def app_exists(deploy: Deploy) -> bool:
    try:
        azure.run(
            ["containerapp", "show", "-n", deploy.name, "-g", deploy.resource_group, "--output", "none"]
        )
        return True
    except AzureError:
        return False


def environment_suffix(deploy: Deploy) -> str | None:
    return (
        azure.run(
            ["containerapp", "env", "show", "-n", deploy.environment, "-g", deploy.resource_group,
             "--query", "properties.customDomainConfiguration.dnsSuffix", "-o", "tsv"]
        ).strip()
        or None
    )


def environment_verification_id(deploy: Deploy) -> str | None:
    return (
        azure.run(
            ["containerapp", "env", "show", "-n", deploy.environment, "-g", deploy.resource_group,
             "--query", "properties.customDomainConfiguration.customDomainVerificationId", "-o", "tsv"]
        ).strip()
        or None
    )


def environment_static_ip(deploy: Deploy) -> str | None:
    return (
        azure.run(
            ["containerapp", "env", "show", "-n", deploy.environment, "-g", deploy.resource_group,
             "--query", "properties.staticIp", "-o", "tsv"]
        ).strip()
        or None
    )


def dns_zone_exists(deploy: Deploy) -> bool:
    if deploy.domain is None:
        return False
    try:
        azure.run(
            ["network", "dns", "zone", "show", "-g", deploy.dns_zone_group,
             "-n", deploy.domain.zone, "--output", "none"]
        )
        return True
    except AzureError:
        return False


def dns_nameservers(deploy: Deploy) -> list[str]:
    if deploy.domain is None:
        return []
    return azure.run(
        ["network", "dns", "zone", "show", "-g", deploy.dns_zone_group,
         "-n", deploy.domain.zone, "--query", "nameServers", "-o", "tsv"]
    ).split()


def required_dns_records(
    deploy: Deploy,
    verification: str | None,
    static_ip: str | None,
    default_domain: str | None = None,
) -> list[dict]:
    """What has to exist in DNS for the route to answer.

    With a managed certificate that is a CNAME and a TXT per app, in whatever
    zone already holds the domain — no delegation, and only record types every
    registrar supports. With a wildcard it is one pair for the whole
    environment, and adding a model later needs no DNS change at all.
    """
    if deploy.domain is None:
        return []
    if deploy.domain.managed:
        domain = deploy.domain
        target = f"{deploy.name}.{default_domain}" if default_domain else "<app default FQDN>"
        verify_fqdn = f"{DOMAIN_VERIFICATION_PREFIX}.{deploy.route}"
        return [
            {
                "type": "CNAME",
                "name": domain.record_name(deploy.route),
                "fqdn": deploy.route,
                "value": target,
                "zone": domain.record_zone,
                "purpose": "points the route at this app",
            },
            {
                "type": "TXT",
                "name": domain.record_name(verify_fqdn),
                "fqdn": verify_fqdn,
                "value": verification or "<app customDomainVerificationId>",
                "zone": domain.record_zone,
                "purpose": "proves you control the hostname",
            },
        ]
    return [
        {
            "type": "A",
            "name": "*",
            "fqdn": f"*.{deploy.domain.zone}",
            "value": static_ip or "<environment static IP>",
            "purpose": "routes every app in the environment",
        },
        {
            "type": "TXT",
            "name": DOMAIN_VERIFICATION_PREFIX,
            "fqdn": f"{DOMAIN_VERIFICATION_PREFIX}.{deploy.domain.zone}",
            "value": verification or "<environment customDomainVerificationId>",
            "purpose": "proves ownership of the zone",
        },
    ]


def ensure_dns_records(deploy: Deploy, emit: Emit) -> list[dict]:
    if deploy.domain is None:
        raise AzureError("no deploy.domain block to publish")
    records = required_dns_records(
        deploy, environment_verification_id(deploy), environment_static_ip(deploy)
    )
    zone_args = ["-g", deploy.dns_zone_group, "-z", deploy.domain.zone]

    for record in records:
        _step(emit, "dns", "start", record=record["fqdn"], type=record["type"])
        kind = record["type"].lower()
        azure.run(
            ["network", "dns", "record-set", kind, "create", *zone_args,
             "-n", record["name"], "--ttl", str(DNS_TTL), "--output", "none"]
        )
        if kind == "a":
            azure.run(
                ["network", "dns", "record-set", "a", "add-record", *zone_args,
                 "-n", record["name"], "-a", record["value"], "--output", "none"]
            )
        else:
            azure.run(
                ["network", "dns", "record-set", "txt", "add-record", *zone_args,
                 "-n", record["name"], "-v", record["value"], "--output", "none"]
            )
        _step(emit, "dns", "ok", record=record["fqdn"])
    return records


def ensure_dns_zone(deploy: Deploy, emit: Emit) -> list[str]:
    """The zone gets its own resource group by default, because a domain outlives
    any particular deployment: tearing down the compute must not take the DNS
    records — and the delegation at the registrar — with it."""
    if deploy.domain is None:
        raise AzureError("no deploy.domain block")
    group = deploy.dns_zone_group
    try:
        azure.run(["group", "show", "-n", group, "--output", "none"])
    except AzureError:
        _step(emit, "resource-group", "start", name=group)
        azure.run(["group", "create", "-n", group, "-l", deploy.location, "--output", "none"])
        _step(emit, "resource-group", "ok", name=group)
    if not dns_zone_exists(deploy):
        _step(emit, "dns-zone", "start", zone=deploy.domain.zone)
        azure.run(
            ["network", "dns", "zone", "create", "-g", deploy.dns_zone_group,
             "-n", deploy.domain.zone, "--output", "none"]
        )
        _step(emit, "dns-zone", "ok", zone=deploy.domain.zone)
    return dns_nameservers(deploy)


def bind_hostname(deploy: Deploy, emit: Emit) -> str:
    """Binds the route to this app and has Azure issue a managed certificate.
    The DNS records have to resolve first — validation reads them, and the
    failure when they are missing does not say so."""
    if deploy.domain is None:
        raise AzureError("no deploy.domain block")
    route = deploy.route
    _step(emit, "hostname", "start", hostname=route)
    azure.run(
        ["containerapp", "hostname", "bind", "-n", deploy.name, "-g", deploy.resource_group,
         "--hostname", route, "--environment", deploy.environment,
         "--validation-method", "CNAME", "--output", "none"]
    )
    _step(emit, "hostname", "ok", hostname=route)
    return route


def bound_hostnames(deploy: Deploy) -> list[str]:
    try:
        rows = azure.json_out(
            ["containerapp", "hostname", "list", "-n", deploy.name, "-g", deploy.resource_group]
        )
    except AzureError:
        return []
    return [row.get("name") for row in rows if row.get("name")]


def environment_default_domain(deploy: Deploy) -> str | None:
    return (
        azure.run(
            ["containerapp", "env", "show", "-n", deploy.environment, "-g", deploy.resource_group,
             "--query", "properties.defaultDomain", "-o", "tsv"]
        ).strip()
        or None
    )


def set_environment_suffix(deploy: Deploy, certificate: str, password: str, emit: Emit) -> None:
    """ACA managed certificates cannot be wildcards, so the environment suffix
    needs one supplied. Everything under it then serves on the custom domain."""
    if deploy.domain is None:
        raise AzureError("no deploy.domain block")
    _step(emit, "dns-suffix", "start", suffix=deploy.domain.zone)
    azure.run(
        ["containerapp", "env", "update", "-n", deploy.environment, "-g", deploy.resource_group,
         "--dns-suffix", deploy.domain.zone,
         "--certificate-file", certificate, "--certificate-password", password,
         "--output", "none"]
    )
    _step(emit, "dns-suffix", "ok", suffix=deploy.domain.zone)
