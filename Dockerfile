# simple-local as a deployable container: llama.cpp's official server image already
# ships llama-server and python3.12, so we only add uv and this package.
#
#   docker build --platform linux/amd64 -t simple-local:embeddings \
#     --build-arg CONFIG=implementations/embeddings/config.container.yml .
#
# Model weights are baked in at build time (see the download step), so a cold
# start doesn't wait on Hugging Face — the same reason the Modal deploy prefetches
# into a Volume.
FROM ghcr.io/ggml-org/llama.cpp:server

ARG CONFIG=implementations/embeddings/config.container.yml
ENV CONFIG=${CONFIG} \
    PATH="/app:/root/.local/bin:${PATH}" \
    LD_LIBRARY_PATH="/app" \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    SIMPLE_LOCAL_HOST=0.0.0.0 \
    SIMPLE_LOCAL_PORT=8080

WORKDIR /srv

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | sh

COPY pyproject.toml uv.lock README.md ./
COPY simple_local ./simple_local
RUN uv sync --frozen --no-dev

COPY implementations ./implementations

ARG PREFETCH=1
RUN if [ "$PREFETCH" = "1" ]; then uv run simple-local download -c "$CONFIG"; fi

EXPOSE 8080

ENTRYPOINT ["sh", "-c", "exec /srv/.venv/bin/simple-local serve -c \"$CONFIG\""]
