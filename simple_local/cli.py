import argparse
import json
import logging
import sys
from pathlib import Path

from . import config as config_mod
from .download import prefetch
from .server import serve

# simple_local.deploy is imported lazily, per command. A serving replica has no
# business loading the code that creates and deletes cloud resources, and an
# image built without the deploy extra must still start. That is also why the
# argparse defaults below are None and resolved inside the commands, rather than
# copied here from simple_local.deploy.config.


def _deploy():
    try:
        from . import deploy
    except ImportError as e:  # pragma: no cover - only when packaged without it
        raise SystemExit(f"deploy commands need the deploy extra: pip install 'simple-local[deploy]' ({e})")
    return deploy


def _serve(args) -> None:
    cfg = config_mod.load(args.config)

    if not cfg.server.api_key:
        print("note: server.api_key not set — auth disabled", file=sys.stderr)

    base = f"http://{cfg.server.host}:{cfg.server.port}/v1"
    print(f"Serving {len(cfg.served_names())} model(s) at {base}:")
    for spec in cfg.models:
        if spec.kind == "llm":
            endpoint = "embeddings" if spec.embeddings else "chat/completions"
        else:
            endpoint = "predict"
        print(f"  {spec.name} ({spec.kind})  {base}/{endpoint}")
        for adapter in spec.adapters:
            print(f"  {adapter.name} (adapter of {spec.name}, scale {adapter.scale})")
    serve(args.config, watch=args.watch)


def _download(args) -> None:
    prefetch(config_mod.load(args.config))


def _emit(payload: dict) -> None:
    print(json.dumps(payload), flush=True)


def _fail(message: str, as_json: bool, code: int = 1) -> int:
    if as_json:
        _emit({"event": "error", "ok": False, "error": message})
    else:
        print(message, file=sys.stderr)
    return code


def _print_findings(findings) -> None:
    for finding in findings:
        marker = "ERROR" if finding.level == "error" else "warn "
        print(f"  {marker}  {finding.field}: {finding.message}")


def _validate(args) -> int:
    d = _deploy()
    deploy = d.resolve(args.config)
    served = d.load_served(deploy)
    findings = d.check(
        deploy,
        served.config,
        served_error=served.error,
        catalog=deploy.catalog(live=args.live),
    )
    blocked = any(f.level == "error" for f in findings)
    if args.json:
        _emit(
            {
                "event": "result",
                "ok": not blocked,
                "app": deploy.name,
                "findings": [f.model_dump() for f in findings],
            }
        )
    else:
        print(f"{deploy.name} ({deploy.container_config})")
        if findings:
            _print_findings(findings)
        else:
            print("  no findings")
    return 1 if blocked else 0


def _render_plan(plan) -> None:
    state = "not deployed" if not plan.exists else ("running" if plan.running else "stopped")
    print(f"{plan.app} in {plan.resource_group} — {state}")
    print(f"\n{plan.tier}: {plan.summary}")
    if plan.tier != "noop":
        print(f"estimated {plan.estimated_seconds}s, restarts replicas: {plan.restarts}")
    if plan.cost:
        print(f"cost: {plan.cost.summary()}")
    for change in plan.changes:
        print(f"\n  {change.field}: {change.before!r} -> {change.after!r}  [{change.tier}]")
        print(f"    {change.effect}")
    if plan.findings:
        print()
        _print_findings(plan.findings)
    print()
    if plan.blocked:
        print("blocked — fix the errors above before deploying")
    elif plan.tier != "noop":
        print(f"apply with: simple-local apply -c <config> --plan-hash {plan.hash}")


def _plan(args) -> int:
    d = _deploy()
    try:
        _, _, plan = d.plan_for(args.config, force_rebuild=args.rebuild)
    except d.AzureError as e:
        return _fail(str(e), args.json)
    if args.json:
        _emit({"event": "result", "ok": not plan.blocked, **plan.model_dump()})
    else:
        _render_plan(plan)
    return 1 if plan.blocked else 0


def _apply(args) -> int:
    d = _deploy()
    emit = _emit if args.json else (lambda payload: print(_human(payload)))
    try:
        deploy, observed, plan = d.plan_for(args.config, force_rebuild=args.rebuild)
        result = d.apply_plan(
            args.config, deploy, observed, plan, emit, expected_hash=args.plan_hash
        )
    except (d.AzureError, d.BuildError, d.PlanStale) as e:
        return _fail(str(e), args.json)
    if args.json:
        _emit({"event": "result", **result})
    else:
        if result.get("api_key"):
            print(f"\napi key: {result['api_key']}\n({result['api_key_note']})")
        print(f"\n{result.get('url') or result['app']}")
    return 0


def _human(payload: dict) -> str:
    if payload.get("event") == "output":
        return payload["line"]
    return f"==> {payload.get('name')} {payload.get('status')}"


def _render(args) -> int:
    d = _deploy()
    deploy = d.resolve(args.config)
    dockerfile = d.render_dockerfile(deploy, d.load_served(deploy).config)
    if args.json:
        _emit(
            {
                "event": "result",
                "ok": True,
                "app": deploy.name,
                "context": deploy.directory,
                "dockerfile": dockerfile,
            }
        )
    else:
        print(f"# build context: {deploy.directory}\n")
        print(dockerfile, end="")
    return 0


def _render_estimate(estimate, label: str) -> None:
    print(f"{label}: {estimate.summary()}")
    for line in estimate.lines:
        print(f"    {line.label:<34} {estimate.currency} {line.monthly:>12,.2f}")
    for note in estimate.assumptions:
        print(f"  - {note}")


def _cost(args) -> int:
    d = _deploy()
    deploy = d.resolve(args.config)
    catalog = deploy.catalog(live=not args.offline)
    profile, azure_name = catalog.resolve(deploy.gpu)
    if profile is None:
        return _fail(f"no workload profile '{deploy.gpu}' in {deploy.location}", args.json)

    prices = d.load_prices(deploy.location, live=not args.offline)
    estimate = d.estimate_cost(deploy, catalog, prices)
    if args.json:
        _emit({"event": "result", "ok": estimate.available, "app": deploy.name,
               "profile": azure_name, **estimate.model_dump()})
        return 0 if estimate.available else 1
    if not estimate.available:
        return _fail(estimate.provenance, args.json)

    print(f"{deploy.name} on {azure_name} ({deploy.cpu:g} vCPU, {deploy.memory_arg}, "
          f"{deploy.min_replicas}-{deploy.max_replicas} replicas)\n")
    print("  per replica, per month:")
    for line in estimate.lines:
        print(f"    {line.label:<34} {estimate.currency} {line.monthly:>12,.2f}")
    print(f"\n  {'floor (min_replicas, at rest)':<34} {estimate.currency} {estimate.floor:>12,.2f}")
    print(f"  {'ceiling (max_replicas, active)':<34} {estimate.currency} {estimate.ceiling:>12,.2f}")
    print()
    for note in estimate.assumptions:
        print(f"  - {note}")
    print(f"\n{estimate.provenance}")
    return 0


def _revisions(args) -> int:
    d = _deploy()
    deploy = d.resolve(args.config)
    try:
        rows = d.operations.revisions(deploy)
    except d.AzureError as e:
        return _fail(str(e), args.json)
    if args.json:
        _emit({"event": "result", "ok": True, "app": deploy.name,
               "route": deploy.route, "revisions": rows})
        return 0
    if deploy.route:
        print(f"{deploy.name}  ->  https://{deploy.route}\n")
    print(f"{'REVISION':<38}  {'LABEL':<10}  {'TRAFFIC':>7}  {'DIGEST':<13}  STATE")
    for row in rows:
        label = row["label"] or "-"
        state = "active" if row["active"] else "inactive"
        print(
            f"{row['name']:<38}  {label:<10}  {str(row['weight']) + '%':>7}  "
            f"{row['digest'] or '-':<13}  {state}"
        )
    return 0


def _promote(args) -> int:
    d = _deploy()
    deploy = d.resolve(args.config)
    labels = deploy.revisions
    emit = _emit if args.json else (lambda p: print(_human(p)))
    try:
        if not deploy.revisions.multiple:
            return _fail(
                "promote needs deploy.revisions.mode: Multiple — in Single mode every apply is "
                "already live",
                args.json,
            )
        live = d.operations.labelled(deploy, labels.live_label)
        candidate = args.revision or d.operations.labelled(deploy, labels.candidate_label)
        if candidate is None:
            return _fail(f"nothing labelled '{labels.candidate_label}' to promote", args.json)
        if args.weight is not None:
            d.operations.set_traffic(
                deploy,
                {labels.live_label: 100 - args.weight, labels.candidate_label: args.weight},
                emit,
            )
        elif live is None:
            d.operations.set_label(deploy, labels.live_label, candidate, 100, emit)
        else:
            d.operations.swap_labels(deploy, labels.candidate_label, labels.live_label, emit)
    except d.AzureError as e:
        return _fail(str(e), args.json)

    payload = {"event": "result", "ok": True, "app": deploy.name, "route": deploy.route,
               "promoted": candidate, "previous": live}
    if args.json:
        _emit(payload)
    else:
        print(f"\n{candidate} is live at https://{deploy.route or deploy.name}")
        if live:
            print(f"previous ({live}) keeps the '{labels.candidate_label}' label — "
                  f"`simple-local rollback` swaps it back")
    return 0


def _rollback(args) -> int:
    args.revision, args.weight = None, None
    d = _deploy()
    deploy = d.resolve(args.config)
    labels = deploy.revisions
    emit = _emit if args.json else (lambda p: print(_human(p)))
    try:
        d.operations.swap_labels(deploy, labels.candidate_label, labels.live_label, emit)
    except d.AzureError as e:
        return _fail(str(e), args.json)
    live = d.operations.labelled(deploy, labels.live_label)
    if args.json:
        _emit({"event": "result", "ok": True, "app": deploy.name, "rolled_back_to": live})
    else:
        print(f"\nrolled back — {live} is live at https://{deploy.route or deploy.name}")
    return 0


def _domain(args) -> int:
    d = _deploy()
    deploy = d.resolve(args.config)
    if deploy.domain is None:
        return _fail(f"{deploy.name} has no deploy.domain block", args.json)

    if args.bind:
        emit = _emit if args.json else (lambda p: print(_human(p)))
        try:
            route = d.operations.bind_hostname(deploy, emit)
        except d.AzureError as e:
            return _fail(f"{e}\n(the CNAME and TXT records must resolve before binding)", args.json)
        if args.json:
            _emit({"event": "result", "ok": True, "app": deploy.name, "route": route})
        else:
            print(f"\nhttps://{route}")
        return 0

    ops = d.operations
    managed = deploy.domain.managed
    try:
        default_domain = ops.environment_default_domain(deploy)
        if managed:
            suffix, static_ip, zone_exists, nameservers = None, None, False, []
            verification = ops.verification_id(deploy)
            bound = ops.bound_hostnames(deploy)
            deployed = ops.app_exists(deploy)
        else:
            suffix = ops.environment_suffix(deploy)
            verification = ops.environment_verification_id(deploy)
            static_ip = ops.environment_static_ip(deploy)
            zone_exists = ops.dns_zone_exists(deploy)
            nameservers = ops.dns_nameservers(deploy) if zone_exists else []
            bound, deployed = [], True
    except d.AzureError as e:
        return _fail(str(e), args.json)
    records = ops.required_dns_records(deploy, verification, static_ip, default_domain)

    live = deploy.route in bound if managed else suffix == deploy.domain.zone
    if args.json:
        _emit({"event": "result", "ok": live, "app": deploy.name, "route": deploy.route,
               "certificate": deploy.domain.certificate, "zone": deploy.domain.zone,
               "bound": bound, "zone_exists": zone_exists, "nameservers": nameservers,
               "environment_suffix": suffix, "records": records})
        return 0

    print(f"{deploy.name}  ->  https://{deploy.route}")
    print(f"  certificate: {deploy.domain.certificate}\n")
    if managed:
        if not deployed:
            print(f"  app not deployed yet — add these records now, then "
                  f"`make apply CONFIG={args.config}`")
        print(f"  hostname bound: {'yes' if live else 'no'}")
        print("  Azure issues and renews the certificate; these two records can live")
        print("  in whatever zone already holds the domain — no delegation needed.\n")
    else:
        print(f"  zone {deploy.domain.zone}: {'created' if zone_exists else 'NOT created'} "
              f"in {deploy.dns_zone_group}")
        if not zone_exists:
            print(f"    az group create -n {deploy.dns_zone_group} -l {deploy.location}")
            print(f"    az network dns zone create -g {deploy.dns_zone_group} "
                  f"-n {deploy.domain.zone}")
        print(f"  environment suffix: {suffix or 'not set'}\n")
        if nameservers:
            print("  delegate at your registrar — NS records for "
                  f"'{deploy.domain.zone.split('.')[0]}':")
            for ns in nameservers:
                print(f"    {ns}")
            print()

    zone = records[0].get("zone") if records else deploy.domain.zone
    print(f"  DNS records to add in the {zone} zone:")
    for record in records:
        print(f"    {record['type']:<6} {record['name']:<26} {record['value']}")
        print(f"           {record['fqdn']} — {record['purpose']}")
    if managed and not live:
        print("\n  once those resolve:")
        print(f"    uv run simple-local domain -c {args.config} --bind")
    return 0


def _new(args) -> int:
    d = _deploy()
    request = d.console.ModelRequest(
        model_name=args.model, workspace=args.workspace, id=args.id,
        modality=args.modality, gpu=args.gpu, cpu=args.cpu, memory_gi=args.memory,
        min_replicas=args.min_replicas, max_replicas=args.max_replicas, zone=args.zone,
    )
    try:
        directory = d.console.scaffold(request, args.root)
    except FileExistsError as e:
        return _fail(str(e), args.json)

    config = str(directory / "config.yml")
    payload = {
        "event": "result", "ok": True,
        "id": request.route, "workspace": request.workspace,
        "hf_name": request.model_name, "modality": request.inferred_modality,
        "directory": str(directory), "config": config,
    }
    if args.json:
        _emit(payload)
        return 0
    print(f"{request.workspace}/{request.route}  ({request.inferred_modality})")
    print(f"  {config}\n")
    for line in Path(config).read_text().splitlines():
        if "CHOOSE-A-QUANT" in line:
            print("  ! pick a GGUF quantisation before deploying — see the source: line")
    print(f"  make plan CONFIG={config}")
    return 0


def _lifecycle(name, verb):
    def run(args) -> int:
        d = _deploy()
        deploy = d.resolve(args.config)
        try:
            detail = getattr(d, name)(deploy)
        except d.AzureError as e:
            return _fail(str(e), args.json)
        payload = {"event": "result", "ok": True, "app": deploy.name, "action": verb}
        if isinstance(detail, str):
            payload["revision"] = detail
        if args.json:
            _emit(payload)
        else:
            print(f"{deploy.name} {verb}" + (f" ({detail})" if detail else ""))
        return 0

    return run


def _teardown(args) -> int:
    d = _deploy()
    deploy = d.resolve(args.config)
    if not args.yes and not args.json:
        print(f"about to delete {deploy.name} in {deploy.resource_group}")
        if input("type the app name to confirm: ").strip() != deploy.name:
            print("aborted", file=sys.stderr)
            return 1
    elif not args.yes:
        return _fail("teardown needs --yes in --json mode", args.json, 2)
    try:
        d.teardown(deploy)
    except d.AzureError as e:
        return _fail(str(e), args.json)
    if args.json:
        _emit({"event": "result", "ok": True, "app": deploy.name, "action": "deleted"})
    else:
        print(f"deleted {deploy.name} (environment, registry and images are left in place)")
    return 0


def _profiles(args) -> int:
    d = _deploy()
    location = args.location or d.config.DEFAULT_LOCATION
    catalog = d.load_catalog(location, live=not args.offline)
    rows = catalog.gpus() if args.gpu else catalog.sorted()
    if args.json:
        _emit(
            {
                "event": "result",
                "ok": bool(rows),
                "location": location,
                "source": catalog.source,
                "fetched_at": catalog.fetched_at,
                "profiles": [p.model_dump() for p in rows],
            }
        )
        return 0 if rows else 1
    if not rows:
        print(
            f"no catalog for {location} ({catalog.provenance}) — "
            "run without --offline, with Azure access",
            file=sys.stderr,
        )
        return 1
    print(f"{'NAME':<26}  {'CATEGORY':<22}  {'CORES':>5}  {'MEMORY':>7}  GPUS")
    for p in rows:
        print(
            f"{p.name:<26}  {p.category:<22}  {p.cores:>5g}  {p.memory_gi:>6g}Gi  {p.gpus:>4}"
        )
    print(f"\n{catalog.provenance}")
    return 0


def _list(args) -> int:
    d = _deploy()
    group = args.resource_group or d.config.DEFAULT_RESOURCE_GROUP
    try:
        rows = d.list_apps(group)
    except d.AzureError as e:
        return _fail(str(e), args.json)
    if args.workspace:
        rows = [r for r in rows if r.get("workspace") == args.workspace]
    if args.json:
        _emit({"event": "result", "ok": True, "version": 1,
               "resource_group": group, "models": rows})
        return 0
    if not rows:
        print(f"no deployments in {group}")
        return 0
    width = max(len(str(r["id"])) for r in rows + [{"id": "MODEL"}])
    ws = max(len(str(r.get("workspace") or "-")) for r in rows + [{"workspace": "WORKSPACE"}])
    print(f"{'MODEL':<{width}}  {'WORKSPACE':<{ws}}  {'MODALITY':<15}  {'STATUS':<8}  REPLICAS")
    for row in rows:
        replicas = row["replicas"]
        print(
            f"{row['id']:<{width}}  {str(row.get('workspace') or '-'):<{ws}}  "
            f"{str(row.get('modality') or '-'):<15}  {row['status']:<8}  "
            f"{replicas['current']} ({replicas['min']}-{replicas['max']})"
        )
    return 0


def _add_deploy_commands(sub) -> None:
    def config_command(name, func, help_text):
        parser = sub.add_parser(name, help=help_text)
        parser.add_argument("-c", "--config", default="config.yml", help="path to config file")
        parser.add_argument("--json", action="store_true", help="machine-readable output")
        parser.set_defaults(func=func)
        return parser

    validate = config_command("validate", _validate, "check a config against Azure's limits, without Azure")
    validate.add_argument(
        "--live", action="store_true", help="fetch the workload profile catalog from Azure first"
    )
    config_command("render", _render, "print the Dockerfile that would be generated for this config")

    for name, func, help_text in (
        ("plan", _plan, "diff the config against the live deployment and price the change"),
        ("apply", _apply, "run an approved plan"),
    ):
        parser = config_command(name, func, help_text)
        parser.add_argument(
            "--rebuild", action="store_true", help="force an image rebuild even if the digest matches"
        )
        if name == "apply":
            parser.add_argument(
                "--plan-hash",
                help="hash of the plan that was approved; apply refuses if the plan has moved since",
            )

    new = sub.add_parser("new", help="scaffold a model directory from a HuggingFace name")
    new.add_argument("-m", "--model", required=True, help="HuggingFace repo id")
    new.add_argument("-w", "--workspace", default="shared", help="shared, or a customer id")
    new.add_argument("--id", help="route name; derived from the model name by default")
    new.add_argument("--modality", choices=["embeddings", "vision-language", "text"])
    new.add_argument("--gpu", help="workload profile, e.g. Consumption-GPU-NC8as-T4")
    new.add_argument("--cpu", type=float)
    new.add_argument("--memory", help="e.g. 16Gi")
    new.add_argument("--min-replicas", type=int)
    new.add_argument("--max-replicas", type=int)
    new.add_argument("--zone", help="custom domain zone, e.g. model.aeriumhq.com")
    new.add_argument("--root", default="implementations", help="where model directories live")
    new.add_argument("--json", action="store_true")
    new.set_defaults(func=_new)

    config_command("revisions", _revisions, "revisions, their labels and traffic weights")
    domain = config_command("domain", _domain, "custom domain state and the DNS records it needs")
    domain.add_argument(
        "--bind", action="store_true",
        help="bind the hostname and have Azure issue a managed certificate",
    )
    config_command("rollback", _rollback, "swap the live label back to the previous revision")

    promote = config_command("promote", _promote, "make a candidate revision live on the route")
    promote.add_argument("--revision", help="promote this revision instead of the candidate label")
    promote.add_argument(
        "--weight", type=int, help="send only this percent to the candidate (canary) instead of swapping"
    )

    costing = config_command("cost", _cost, "estimated monthly cost at list price")
    costing.add_argument("--offline", action="store_true", help="use cached prices and profiles")

    config_command("stop", _lifecycle("stop", "stopped"), "deactivate the revision, keeping the deployment")
    config_command("start", _lifecycle("start", "started"), "reactivate a stopped deployment, no rebuild")
    teardown_parser = config_command("teardown", _teardown, "delete the deployment")
    teardown_parser.add_argument("--yes", action="store_true", help="skip the confirmation prompt")

    catalog = sub.add_parser("profiles", help="workload profiles this region offers")
    catalog.add_argument("-l", "--location", help="defaults to the deploy block default")
    catalog.add_argument("--gpu", action="store_true", help="only profiles with a GPU")
    catalog.add_argument("--offline", action="store_true", help="use the bundled snapshot, do not call Azure")
    catalog.add_argument("--json", action="store_true", help="machine-readable output")
    catalog.set_defaults(func=_profiles)

    listing = sub.add_parser("list", help="every deployment in a resource group and its URL")
    listing.add_argument("-g", "--resource-group", help="defaults to the deploy block default")
    listing.add_argument("-w", "--workspace", help="only models in this workspace")
    listing.add_argument("--json", action="store_true", help="machine-readable output")
    listing.set_defaults(func=_list)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    # Price and catalog lookups are an implementation detail, not progress.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(prog="simple-local", description="bare-bones local inference")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve")
    serve.add_argument("-c", "--config", default="config.yml", help="path to config file")
    serve.add_argument("--watch", action="store_true", help="reload models when the config or model artifacts change (blue/green for llms)")
    serve.set_defaults(func=_serve)

    download = sub.add_parser("download")
    download.add_argument("-c", "--config", default="config.yml", help="path to config file")
    download.set_defaults(func=_download)

    _add_deploy_commands(sub)

    args = parser.parse_args()
    raise SystemExit(args.func(args) or 0)
