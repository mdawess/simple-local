#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

need_brew() {
  local formula="$1" bin="$2"
  if command -v "$bin" >/dev/null 2>&1; then
    echo "  ok: $bin"
  elif command -v brew >/dev/null 2>&1; then
    echo "  installing $formula ..."
    brew install "$formula"
  else
    echo "  ! $bin missing and Homebrew not found — install $formula manually" >&2
  fi
}

echo "==> uv"
command -v uv >/dev/null 2>&1 || { echo "install uv first: https://docs.astral.sh/uv/" >&2; exit 1; }
echo "  ok: uv"

echo "==> python env (uv sync)"
uv sync

echo "==> llama.cpp (LLM runtime)"
need_brew llama.cpp llama-server

echo "==> whisper.cpp (STT for the sdr example)"
need_brew whisper-cpp whisper-cli

echo "==> whisper model"
WMODEL="$HOME/simple-local/models/whisper/ggml-base.en.bin"
if [ -f "$WMODEL" ]; then
  echo "  ok: $WMODEL"
else
  mkdir -p "$(dirname "$WMODEL")"
  curl -L --fail -o "$WMODEL" https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin
fi

echo "==> config.yml"
[ -f config.yml ] || cp config.llm.example.yml config.yml
echo "  ok: config.yml"

echo "==> LLM model (GGUF, downloads on first run)"
uv run simple-local download -c config.yml

echo
echo "done. next:"
echo "  make serve                 # start the local LLM server"
echo "  make run EXAMPLE=chat       # or: make run EXAMPLE=sdr ARGS=--dry-run"
