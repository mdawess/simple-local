"""Serves Qwen3.6-27B on a Modal GPU behind an OpenAI-compatible endpoint, so a
laptop doesn't have to. simple-local points a `kind: remote` model at the URL
this prints on deploy.

    uv run modal secret create simple-local-proxy MODAL_PROXY_KEY=$(openssl rand -hex 16)
    uv run modal deploy implementations/llm-remote/deploy.py

The container loads weights once on start and stays warm; requests hitting a
warm container pay nothing extra. After SCALEDOWN_WINDOW idle seconds it stops,
and the next request pays a cold start — weights come from the Volume, not from
Hugging Face.
"""

import subprocess

import modal

APP_NAME = "aerium-inference"
LABEL = "llm"
MODEL = "cyankiwi/Qwen3.6-27B-AWQ-INT4"
SERVED_NAME = "Qwen3.6-27B"
GPU = "A100-40GB"
VLLM_PORT = 8000
SCALEDOWN_WINDOW = 30 * 60
STARTUP_TIMEOUT = 15 * 60
MIN_CONTAINERS = 0 # always scale down when idle to save $$ since this is not a high-traffic service
MAX_CONCURRENT = 16


app = modal.App(APP_NAME)

cache = modal.Volume.from_name("simple-local-hf-cache", create_if_missing=True)
proxy_key = modal.Secret.from_name("simple-local-proxy", required_keys=["MODAL_PROXY_KEY"])

image = (
    modal.Image.from_registry("nvidia/cuda:13.1.2-devel-ubuntu22.04", add_python="3.12")
    .pip_install("vllm==0.26.0", "huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/cache"})
)


@app.function(
    image=image,
    gpu=GPU,
    volumes={"/cache": cache},
    secrets=[proxy_key],
    scaledown_window=SCALEDOWN_WINDOW,
    timeout=24 * 60 * 60,
    min_containers=MIN_CONTAINERS,
    max_containers=1,
)
@modal.concurrent(max_inputs=MAX_CONCURRENT)
@modal.web_server(port=VLLM_PORT, startup_timeout=STARTUP_TIMEOUT, label=LABEL)
def serve() -> None:
    import os

    subprocess.Popen(
        [
            "vllm",
            "serve",
            MODEL,
            "--served-model-name", SERVED_NAME,
            "--api-key", os.environ["MODAL_PROXY_KEY"],
            "--host", "0.0.0.0",
            "--port", str(VLLM_PORT),
            "--max-model-len", "8192",
            "--gpu-memory-utilization", "0.90",
        ]
    )


@app.function(image=image, volumes={"/cache": cache}, timeout=60 * 60)
def prefetch() -> None:
    """Pull weights into the Volume without booting a GPU, so the first real
    request isn't also the first download:
        uv run modal run implementations/llm-remote/deploy.py::prefetch
    """
    from huggingface_hub import snapshot_download

    snapshot_download(MODEL)
    cache.commit()
    print(f"cached {MODEL}")
