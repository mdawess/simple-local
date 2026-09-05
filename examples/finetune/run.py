import subprocess
import sys
from pathlib import Path

import build_dataset

# Mac-native LoRA fine-tune of Qwen-2.5-3B on the SDR dataset, then convert to
# GGUF so simple-local can serve it. Apple Silicon only (MLX uses Metal).
# Heavy + slow: pulls the ~6GB base model, trains on the GPU, and the GGUF
# conversion pulls torch/transformers. A worked example, not a turnkey black box.

HERE = Path(__file__).resolve().parent
BASE = "Qwen/Qwen2.5-3B-Instruct"          # HF safetensors base (not the GGUF)
ADAPTERS = HERE / "adapters"
MERGED = HERE / "merged"
GGUF = HERE / "qwen-sdr-lora.gguf"
LLAMA = Path.home() / "simple-local" / "llama.cpp"  # cloned for the convert script


def step(label: str, cmd: list) -> None:
    print(f"\n==> {label}\n    {' '.join(map(str, cmd))}", flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    if sys.platform != "darwin":
        print("warning: MLX training is Apple Silicon only — see the README for other hardware", file=sys.stderr)

    print("==> 1/4  build dataset from seeds + sdr transcripts")
    build_dataset.main()

    step("2/4  LoRA fine-tune (MLX, Apple Silicon GPU)", [
        "uv", "run", "--with", "mlx-lm", "mlx_lm.lora",
        "--model", BASE, "--train", "--data", HERE / "data",
        "--iters", "200", "--batch-size", "2", "--num-layers", "8",
        "--adapter-path", ADAPTERS,
    ])

    step("3/4  fuse adapter into the base weights", [
        "uv", "run", "--with", "mlx-lm", "mlx_lm.fuse",
        "--model", BASE, "--adapter-path", ADAPTERS, "--save-path", MERGED,
    ])

    if not LLAMA.exists():
        step("clone llama.cpp (for the convert script)", [
            "git", "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp", LLAMA,
        ])
    step("4/4  convert merged model to GGUF", [
        "uv", "run", "--with-requirements", LLAMA / "requirements.txt",
        "python", LLAMA / "convert_hf_to_gguf.py", MERGED,
        "--outfile", GGUF, "--outtype", "q8_0",
    ])

    print(f"\ndone -> {GGUF}")
    print("serve it:  make serve CONFIG=examples/finetune/config.yml")
    print("then chat: make run EXAMPLE=chat")


if __name__ == "__main__":
    main()
