import hmac
import json
import logging
import re
import time
from contextlib import asynccontextmanager
from typing import Optional

import httpx
from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse
from starlette.concurrency import iterate_in_threadpool, run_in_threadpool

from . import config as config_mod
from .config import Config
from .registry import REBUILD_LOCK, ChatTarget, ModelEntry, Registry, build_entry

log = logging.getLogger("simple_local.requests")

LEGACY_BASE = "/environments/{env}/sync/v1"

# llama-server reports usage on non-stream responses; streams carry the counts
# in the final chunk's timings (prompt_n / predicted_n) instead.
_PROMPT_TOKENS = re.compile(rb'"(?:prompt_tokens|prompt_n)"\s*:\s*(\d+)')
_COMPLETION_TOKENS = re.compile(rb'"(?:completion_tokens|predicted_n)"\s*:\s*(\d+)')


def create_app(cfg: Config, registry: Registry, config_path: str | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            yield
        finally:
            await app.state.client.aclose()
            app.state.registry.stop_all()

    app = FastAPI(title="simple-local", lifespan=lifespan)
    app.state.registry = registry  # swapped in place by the --watch reloader
    app.state.client = httpx.AsyncClient(timeout=None)

    def auth(authorization: Optional[str] = Header(None)) -> None:
        if not cfg.server.api_key:
            return
        token = (authorization or "").removeprefix("Bearer ")
        if not hmac.compare_digest(token, cfg.server.api_key):
            raise HTTPException(status_code=401, detail="invalid api key")

    router = APIRouter(dependencies=[Depends(auth)])

    @router.get("/models")
    async def list_models(request: Request):
        return {"object": "list", "data": request.app.state.registry.model_cards()}

    @router.post("/chat/completions")
    async def chat_completions(request: Request):
        reg: Registry = request.app.state.registry
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="body must be JSON")
        name, target = _resolve_target(reg.chat_targets, body.get("model"), "chat")
        if not target.runtime.ready.is_set():
            raise HTTPException(status_code=503, detail=f"model '{name}' is restarting")
        lora = target.lora_payload()
        if lora is not None:
            body["lora"] = lora

        client: httpx.AsyncClient = request.app.state.client
        upstream = client.build_request(
            "POST", target.runtime.base_url + "/v1/chat/completions", json=body
        )
        started = time.monotonic()
        try:
            if body.get("stream"):
                resp = await client.send(upstream, stream=True)
                return StreamingResponse(
                    _relay_and_log(resp, name, started),
                    status_code=resp.status_code,
                    media_type=resp.headers.get("content-type"),
                )
            resp = await client.send(upstream)
        except httpx.TransportError as e:
            raise HTTPException(status_code=503, detail=f"model '{name}' is unavailable: {e}")
        _log_request(name, resp.status_code, started, resp.content)
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type"),
        )

    @router.post("/embeddings")
    async def embeddings(request: Request):
        reg: Registry = request.app.state.registry
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="body must be JSON")
        name, target = _resolve_target(reg.embedding_targets, body.get("model"), "embedding")
        if not target.runtime.ready.is_set():
            raise HTTPException(status_code=503, detail=f"model '{name}' is restarting")
        lora = target.lora_payload()
        if lora is not None:
            body["lora"] = lora

        client: httpx.AsyncClient = request.app.state.client
        started = time.monotonic()
        try:
            resp = await client.post(target.runtime.base_url + "/v1/embeddings", json=body)
        except httpx.TransportError as e:
            raise HTTPException(status_code=503, detail=f"model '{name}' is unavailable: {e}")
        _log_request(name, resp.status_code, started, resp.content)
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type"),
        )

    @router.post("/predict")
    async def predict(request: Request):
        reg: Registry = request.app.state.registry
        body = await request.json()
        runtime = _resolve_predictor(reg, body.get("model"))
        try:
            result = await run_in_threadpool(runtime.predict, body)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        if isinstance(result, dict):
            return result
        # iterator of rows from a custom runtime: stream as NDJSON
        async def ndjson():
            async for row in iterate_in_threadpool(iter(result)):
                yield json.dumps(row) + "\n"

        return StreamingResponse(ndjson(), media_type="application/x-ndjson")

    @router.post("/reload")
    async def reload_models(request: Request):
        raw = await request.body()
        only = json.loads(raw).get("model") if raw else None
        return await run_in_threadpool(_reload_registry, request.app, config_path, only)

    @router.get("/metrics")
    async def metrics(request: Request, model: Optional[str] = None):
        reg: Registry = request.app.state.registry
        llm_entries = {n: e for n, e in reg.entries.items() if e.spec.kind == "llm"}
        if not llm_entries:
            raise HTTPException(status_code=404, detail="no llm models configured")
        if model is None:
            if len(llm_entries) > 1:
                raise HTTPException(
                    status_code=422,
                    detail=f"multiple llm models; pass ?model= one of {sorted(llm_entries)}",
                )
            entry = next(iter(llm_entries.values()))
        else:
            entry = llm_entries.get(model)
            if entry is None:
                raise HTTPException(status_code=404, detail=f"unknown llm model '{model}'")
        if not entry.runtime.ready.is_set():
            raise HTTPException(status_code=503, detail=f"model '{entry.spec.name}' is restarting")
        try:
            resp = await request.app.state.client.get(entry.runtime.base_url + "/metrics")
        except httpx.TransportError as e:
            raise HTTPException(status_code=503, detail=str(e))
        return PlainTextResponse(resp.text, status_code=resp.status_code)

    @app.get("/health")
    async def health(request: Request):
        reg: Registry = request.app.state.registry
        statuses = {}
        for name, entry in reg.entries.items():
            if entry.spec.kind == "llm":
                statuses[name] = "ready" if entry.runtime.ready.is_set() else "restarting"
            else:
                statuses[name] = "ready"
        all_ready = all(s == "ready" for s in statuses.values())
        return JSONResponse(
            {"status": "ok" if all_ready else "degraded", "models": statuses},
            status_code=200 if all_ready else 503,
        )

    app.include_router(router, prefix="/v1")
    app.include_router(router, prefix=LEGACY_BASE)  # back-compat alias

    return app


def _reload_registry(app, config_path: str | None, only: str | None) -> dict:
    """Force-rebuild model entries (re-resolving remote sources) and swap the
    registry blue/green. With `only`, just that model; otherwise everything in
    the (re-read) config, including additions and removals."""
    if not REBUILD_LOCK.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="a reload is already in progress")
    try:
        old = app.state.registry.entries
        specs = config_mod.load(config_path).models if config_path else [
            e.spec for e in old.values()
        ]
        if only is not None and only not in {s.name for s in specs}:
            raise HTTPException(status_code=404, detail=f"unknown model '{only}'")

        new_entries: dict[str, ModelEntry] = {}
        reloaded: list[str] = []
        errors: dict[str, str] = {}
        for spec in specs:
            prev = old.get(spec.name)
            if only is not None and spec.name != only:
                if prev is not None:
                    new_entries[spec.name] = prev
                continue
            try:
                new_entries[spec.name] = build_entry(spec)
                reloaded.append(spec.name)
            except Exception as e:
                errors[spec.name] = str(e)
                if prev is not None:
                    new_entries[spec.name] = prev

        app.state.registry = Registry(new_entries)
        for name, entry in old.items():
            if new_entries.get(name) is not entry and hasattr(entry.runtime, "stop"):
                entry.runtime.stop()
        return {
            "reloaded": reloaded,
            "kept": sorted(set(new_entries) - set(reloaded)),
            "errors": errors,
        }
    finally:
        REBUILD_LOCK.release()


def _resolve_target(targets: dict[str, ChatTarget], model, what: str) -> tuple[str, ChatTarget]:
    if not targets:
        raise HTTPException(status_code=404, detail=f"no {what} models configured")
    if model is None:
        if len(targets) == 1:
            return next(iter(targets.items()))
        raise HTTPException(
            status_code=422,
            detail=f"specify 'model'; available: {sorted(targets)}",
        )
    target = targets.get(model)
    if target is None:
        raise HTTPException(
            status_code=404,
            detail=f"unknown {what} model '{model}'; available: {sorted(targets)}",
        )
    return model, target


def _resolve_predictor(reg: Registry, model):
    if not reg.predictors:
        raise HTTPException(status_code=404, detail="no predictor models configured")
    if model is None:
        if len(reg.predictors) == 1:
            return next(iter(reg.predictors.values()))
        raise HTTPException(
            status_code=422,
            detail=f"specify 'model'; available: {sorted(reg.predictors)}",
        )
    runtime = reg.predictors.get(model)
    if runtime is None:
        raise HTTPException(
            status_code=404,
            detail=f"unknown model '{model}'; available: {sorted(reg.predictors)}",
        )
    return runtime


async def _relay_and_log(resp: httpx.Response, model: str, started: float):
    # Keep a tail of the stream so token usage from the final SSE chunk can be logged.
    tail = b""
    try:
        async for chunk in resp.aiter_raw():
            tail = (tail + chunk)[-8192:]
            yield chunk
    finally:
        await resp.aclose()
        _log_request(model, resp.status_code, started, tail)


def _log_request(model: str, status: int, started: float, payload: bytes) -> None:
    prompt = _PROMPT_TOKENS.findall(payload)
    completion = _COMPLETION_TOKENS.findall(payload)
    log.info(
        "model=%s status=%s duration_ms=%d prompt_tokens=%s completion_tokens=%s",
        model,
        status,
        (time.monotonic() - started) * 1000,
        int(prompt[-1]) if prompt else "-",
        int(completion[-1]) if completion else "-",
    )
