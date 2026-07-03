import argparse
import sys

import uvicorn

from . import config as config_mod
from .download import ensure_model
from .runtimes.llm import LLMRuntime
from .runtimes.predictor import PredictorRuntime
from .server import create_app


def _build_runtime(cfg, model_path):
    if cfg.kind == "llm":
        return LLMRuntime(cfg, model_path)
    return PredictorRuntime(cfg, model_path)


def _serve(args) -> None:
    cfg = config_mod.load(args.config)
    model_path = ensure_model(cfg)
    runtime = _build_runtime(cfg, model_path)
    app = create_app(cfg, runtime)

    if not cfg.server.api_key:
        print("note: server.api_key not set — auth disabled", file=sys.stderr)

    endpoint = "chat/completions" if cfg.kind == "llm" else "predict"
    print(
        f"Serving {cfg.model_name} ({cfg.kind}) at "
        f"http://{cfg.server.host}:{cfg.server.port}/environments/development/sync/v1/{endpoint}"
    )
    try:
        uvicorn.run(app, host=cfg.server.host, port=cfg.server.port, log_level="info")
    finally:
        if hasattr(runtime, "stop"):
            runtime.stop()


def _download(args) -> None:
    cfg = config_mod.load(args.config)
    ensure_model(cfg)


def main() -> None:
    parser = argparse.ArgumentParser(prog="simple-local", description="bare-bones local inference")
    sub = parser.add_subparsers(dest="command", required=True)

    for name, fn in (("serve", _serve), ("download", _download)):
        p = sub.add_parser(name)
        p.add_argument("-c", "--config", default="config.yml", help="path to config file")
        p.set_defaults(func=fn)

    args = parser.parse_args()
    args.func(args)
