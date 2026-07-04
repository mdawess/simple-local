import hmac
from typing import Optional

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool

from .config import Config
from .runtimes.llm import LLMRuntime
from .runtimes.predictor import PredictorRuntime

BASE = "/environments/{env}/sync/v1"


def create_app(cfg: Config, runtime) -> FastAPI:
    app = FastAPI(title="simple-local")
    app.state.runtime = runtime  # swapped in place by the --watch reloader

    def auth(authorization: Optional[str] = Header(None)) -> None:
        if not cfg.server.api_key:
            return
        token = (authorization or "").removeprefix("Bearer ")
        if not hmac.compare_digest(token, cfg.server.api_key):
            raise HTTPException(status_code=401, detail="invalid api key")

    if cfg.kind == "llm":
        _mount_llm(app, runtime, auth)
    else:
        _mount_predictor(app, runtime, auth)

    return app


def _mount_llm(app: FastAPI, runtime: LLMRuntime, auth) -> None:
    client = httpx.AsyncClient(base_url=runtime.base_url, timeout=None)

    @app.on_event("shutdown")
    async def _close():
        await client.aclose()

    @app.post(BASE + "/chat/completions", dependencies=[Depends(auth)])
    async def chat_completions(env: str, request: Request):
        upstream = client.build_request(
            "POST",
            "/v1/chat/completions",
            content=await request.body(),
            headers={"content-type": "application/json"},
        )
        resp = await client.send(upstream, stream=True)
        return StreamingResponse(
            resp.aiter_raw(),
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type"),
            background=BackgroundTask(resp.aclose),
        )


def _mount_predictor(app: FastAPI, runtime: PredictorRuntime, auth) -> None:
    @app.post(BASE + "/predict", dependencies=[Depends(auth)])
    async def predict(env: str, request: Request):
        runtime = request.app.state.runtime  # live runtime; may be hot-swapped
        body = await request.json()
        inputs = body.get("inputs")
        if not isinstance(inputs, list):
            raise HTTPException(
                status_code=422, detail="body must contain an 'inputs' list"
            )
        try:
            return await run_in_threadpool(runtime.predict, inputs)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
