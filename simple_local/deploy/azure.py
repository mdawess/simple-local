import json
import re
import shutil
import subprocess

from pydantic import BaseModel, Field

from .config import DEFAULT_WORKLOAD_PROFILE, MANAGED_ENV, TAG_DIGEST_PATTERN
from .units import parse_memory_gi


class AzureError(RuntimeError):
    pass


def _require_az() -> str:
    path = shutil.which("az")
    if path is None:
        raise AzureError("az CLI not found — brew install azure-cli")
    return path


def run(args: list[str]) -> str:
    proc = subprocess.run(
        [_require_az(), *args], capture_output=True, text=True, check=False
    )
    if proc.returncode != 0:
        raise AzureError((proc.stderr or proc.stdout).strip() or f"az {args[0]} failed")
    return proc.stdout


def stream(args: list[str], emit) -> None:
    proc = subprocess.Popen(
        [_require_az(), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in proc.stdout:
        emit(line.rstrip("\n"))
    if proc.wait() != 0:
        raise AzureError(f"az {' '.join(args[:2])} exited {proc.returncode}")


def json_out(args: list[str]):
    return json.loads(run([*args, "-o", "json"]))


_json = json_out


class Observed(BaseModel):
    name: str
    resource_group: str
    exists: bool
    image: str | None = None
    config_digest: str | None = None
    cpu: float | None = None
    memory_gi: float | None = None
    gpu: str | None = None
    min_replicas: int | None = None
    max_replicas: int | None = None
    env: dict[str, str] = Field(default_factory=dict)
    fqdn: str | None = None
    latest_revision: str | None = None
    active_revisions: int = 0

    @property
    def url(self) -> str | None:
        return f"https://{self.fqdn}" if self.fqdn else None

    # An app whose revisions are all deactivated still reports runningStatus
    # "Running", so whether anything serves has to come from the revisions.
    @property
    def running(self) -> bool:
        return self.exists and self.active_revisions > 0


_TAG_DIGEST = re.compile(TAG_DIGEST_PATTERN)


def _digest_from_image(image: str | None) -> str | None:
    if not image or ":" not in image:
        return None
    match = _TAG_DIGEST.search(image.rsplit(":", 1)[1])
    return match.group(1) if match else None


def _workload_profile(name: str | None) -> str | None:
    return None if not name or name == DEFAULT_WORKLOAD_PROFILE else name


def _env_map(entries) -> dict[str, str]:
    env = {}
    for entry in entries or []:
        name = entry.get("name")
        if name is None or name in MANAGED_ENV:
            continue
        secret = entry.get("secretRef")
        env[name] = f"secretref:{secret}" if secret else str(entry.get("value", ""))
    return env


def observe(name: str, resource_group: str) -> Observed:
    try:
        app = _json(["containerapp", "show", "-n", name, "-g", resource_group])
    except AzureError as e:
        if "was not found" in str(e) or "ResourceNotFound" in str(e):
            return Observed(name=name, resource_group=resource_group, exists=False)
        raise

    properties = app.get("properties", {})
    template = properties.get("template", {})
    containers = template.get("containers") or [{}]
    resources = containers[0].get("resources") or {}
    scale = template.get("scale") or {}
    ingress = (properties.get("configuration") or {}).get("ingress") or {}

    memory = resources.get("memory")
    revisions = _json(
        ["containerapp", "revision", "list", "-n", name, "-g", resource_group, "--all"]
    )
    active = sum(1 for r in revisions if (r.get("properties") or {}).get("active"))

    image = containers[0].get("image")
    return Observed(
        name=name,
        resource_group=resource_group,
        exists=True,
        image=image,
        config_digest=_digest_from_image(image),
        cpu=float(resources["cpu"]) if resources.get("cpu") is not None else None,
        memory_gi=parse_memory_gi(memory) if memory else None,
        gpu=_workload_profile(properties.get("workloadProfileName")),
        min_replicas=scale.get("minReplicas"),
        max_replicas=scale.get("maxReplicas"),
        env=_env_map(containers[0].get("env")),
        fqdn=ingress.get("fqdn"),
        latest_revision=properties.get("latestRevisionName"),
        active_revisions=active,
    )
