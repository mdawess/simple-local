from typing import Literal

# Azure defaults

DEFAULT_RESOURCE_GROUP = "simple-local-ca"
DEFAULT_LOCATION = "canadacentral"
DEFAULT_ENVIRONMENT = "simple-local-env"
DEFAULT_TARGET_PORT = 8080

# Default (non-GPU) profile on azure
DEFAULT_WORKLOAD_PROFILE = "Consumption"

# Created and injected on every deploy, so neither is ever a config diff.
MANAGED_SECRET = "api-key"
MANAGED_ENV = {"SIMPLE_LOCAL_API_KEY"}

# Image tags are "<timestamp>-<digest>". Tags predating digest stamping are
# plain timestamps, whose trailing segment would otherwise read as a digest.
TAG_DIGEST_PATTERN = r"-([0-9a-f]{12})$"
DIGEST_LENGTH = 12

# Replica defaults

# A GPU replica is pointless at CPU-app sizing, and a GPU app that scales wide
# hits quota rather than capacity, so the defaults differ by whether a workload
# profile was asked for.
DEFAULT_CPU = {"cpu": 2.0, "gpu": 8.0}
DEFAULT_MEMORY_GI = {"cpu": 4.0, "gpu": 56.0}
DEFAULT_MAX_REPLICAS = {"cpu": 3, "gpu": 1}
DEFAULT_MIN_REPLICAS = 0

# Image build
LLAMA_BASE_IMAGE = "ghcr.io/ggml-org/llama.cpp:server"
PYTHON_BASE_IMAGE = "python:3.12-slim"

# Container Apps runs a recursive chown over WORKDIR before starting a container
# and gives it roughly three minutes. Torch and CUDA wheels are enough files to
# blow that budget, and the replica dies with ContainerCreateFailure before the
# server ever runs. Keeping the venv out of WORKDIR is not optional.
VENV_PATH = "/opt/venv"
WORKDIR = "/srv"
IMAGE_CONFIG_PATH = f"{WORKDIR}/config.yml"
WHEELS_PATH = f"{WORKDIR}/wheels"
EXPOSED_PORT = 8080

BASE_IMAGE_ENV = {
    # The venv is on PATH because runtimes spawn siblings out of it — `vllm`
    # lives there, and the entrypoint's absolute path does not help a subprocess.
    "PATH": f"{VENV_PATH}/bin:/root/.local/bin:${{PATH}}",
    "UV_COMPILE_BYTECODE": "1",
    "UV_LINK_MODE": "copy",
    "HF_HOME": "/models",
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
    "SIMPLE_LOCAL_HOST": "0.0.0.0",
    "SIMPLE_LOCAL_PORT": "8080",
}

# The llama.cpp image keeps llama-server and its shared objects in /app.
LLAMA_IMAGE_ENV = {
    "PATH": f"/app:{VENV_PATH}/bin:/root/.local/bin:${{PATH}}",
    "LD_LIBRARY_PATH": "/app",
}

BASE_SYSTEM_PACKAGES = ["curl", "ca-certificates"]

# Never worth shipping into a build context, and __pycache__ in particular would
# make the image digest churn on every local run.
EXCLUDED_NAMES = {"__pycache__", ".venv", ".git", ".DS_Store", "Dockerfile", "wheels"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}
# Both configs are replaced by a single canonical one inside the image.
CONFIG_NAMES = ("config.yml", "config.container.yml")

DOCKERIGNORE = "__pycache__\n*.pyc\n.venv\n.git\n"

# --- Workspaces and identity ---

# A model is either shared across customers or siloed to one. The workspace is
# part of a model's identity, its resource group, and its cost attribution.
DEFAULT_WORKSPACE = "shared"
WORKSPACE_RESOURCE_GROUP = "aerium-models-{workspace}"

# Stamped on every resource so Cost Management can group spend by model without
# anyone maintaining a mapping. Untagged resources are unattributable after the
# fact, so this has to happen at creation.
TAG_PROJECT = "aerium"
TAG_MANAGED_BY = "simple-local"

MODALITIES = ("embeddings", "vision-language", "text")

# --- Revisions and routing ------------------------------------------------

# Multiple-revision mode is what makes a route stable: a new image arrives as a
# revision taking no traffic, gets tested at its own URL, and goes live by
# swapping a label. The app name — and therefore the public hostname — never
# moves, so swapping the model behind a route is a config change rather than a
# URL change for every caller.
SINGLE_REVISION = "Single"
MULTIPLE_REVISION = "Multiple"

LIVE_LABEL = "live"
CANDIDATE_LABEL = "candidate"

# Azure builds a labelled revision's hostname as <app>---<label>.<suffix>. That
# is still one DNS label, so a *.<suffix> wildcard certificate covers it.
LABEL_SEPARATOR = "---"

# --- Custom domain --------------------------------------------------------

# Ownership of a subdomain is proven with a TXT record at asuid.<label>, holding
# the resource's customDomainVerificationId.
DOMAIN_VERIFICATION_PREFIX = "asuid"
DNS_TTL = 3600

# --- Plan tiers -----------------------------------------------------------

Tier = Literal["noop", "promote", "scale", "update", "rebuild", "create"]

TIER_ORDER: list[Tier] = ["noop", "promote", "scale", "update", "rebuild", "create"]

TIER_SECONDS: dict[Tier, int] = {
    "noop": 0,
    "promote": 10,
    "scale": 20,
    "update": 90,
    "rebuild": 600,
    "create": 720,
}

TIER_SUMMARY: dict[Tier, str] = {
    "noop": "nothing to do",
    "promote": "moves a label between revisions; no restart, no rebuild",
    "scale": "adjusts replica bounds in place; running replicas are untouched",
    "update": "new revision from the same image; replicas restart",
    "rebuild": "builds and pushes a new image, then a new revision; replicas restart",
    "create": "provisions the app and everything it needs, then a first build",
}

RESTARTING_TIERS = ("update", "rebuild", "create")
BUILDING_TIERS = ("create", "rebuild")

# --- Model sizing ---------------------------------------------------------

# The llama.cpp memory model, measured on the CPU embedder (see DEPLOY.md). The
# KV cache is preallocated for the *total* context at startup; the compute
# buffer is per in-flight request and scales with the ubatch. These are
# measurements of one model family rather than constants, which is why a config
# can override them — see spec.Sizing.
KV_KIB_PER_TOKEN = 112
COMPUTE_BUFFER_GI_PER_4K_TOKENS = 3.2
COMPUTE_BUFFER_REFERENCE_TOKENS = 4000
SIZING_SOURCE = "measured on Qwen3-Embedding-0.6B Q8_0, llama.cpp"
DEFAULT_UBATCH_TOKENS = 4096

# Below this much headroom the sizing check warns rather than staying silent.
MEMORY_HEADROOM_WARN_RATIO = 0.95

# --- Catalog and pricing --------------------------------------------------

# Past this a cached workload profile catalog is still used, but says how old.
CATALOG_STALE_AFTER_DAYS = 30

PRICES_ENDPOINT = "https://prices.azure.com/api/retail/prices"
PRICES_SERVICE = "Azure Container Apps"
PRICES_TIMEOUT_SECONDS = 30
HOURS_PER_MONTH = 730  # what Azure's own pricing pages assume

# Consumption ("Standard") is metered per second of vCPU and memory, split into
# active and idle rates. Dedicated profiles bill the node by the hour whether it
# is doing anything or not.
VCPU_ACTIVE = "Standard vCPU Active Usage"
VCPU_IDLE = "Standard vCPU Idle Usage"
MEMORY_ACTIVE = "Standard Memory Active Usage"
MEMORY_IDLE = "Standard Memory Idle Usage"
REQUESTS = "Standard Requests"
DEDICATED_VCPU = "Dedicated vCPU Usage"
DEDICATED_MEMORY = "Dedicated Memory Usage"
DEDICATED_GPU = "Dedicated GPU Usage"
DEDICATED_PLAN = "Dedicated Plan Management"

# Consumption GPU is a per-second meter named after the silicon, so it is
# matched on a fragment of the profile name rather than an exact meter name.
CONSUMPTION_GPU_METERS = {
    "T4": "Standard NC T4 v3 GPU Usage",
    "A100": "Standard NC A100 v4 GPU Usage",
}

# --- Units ----------------------------------------------------------------

MEMORY_PATTERN = r"^\s*([0-9]*\.?[0-9]+)\s*(?:gi|gib|g)?\s*$"
