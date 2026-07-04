import argparse
import sys

import uvicorn

from . import config as config_mod
from .download import ensure_model
from .reload import ReloadWatcher, build_runtime
from .server import create_app


def _serve(args) -> None:
    cfg = config_mod.load(args.config)
    model_path = ensure_model(cfg)
    runtime = build_runtime(cfg, model_path)
    app = create_app(cfg, runtime)

    if not cfg.server.api_key:
        print("note: server.api_key not set — auth disabled", file=sys.stderr)

    watcher = None
    if args.watch:
        watcher = ReloadWatcher(app, args.config, cfg.kind, model_path)
        watcher.start()

    endpoint = "chat/completions" if cfg.kind == "llm" else "predict"
    print(
        f"Serving {cfg.model_name} ({cfg.kind}) at "
        f"http://{cfg.server.host}:{cfg.server.port}/environments/development/sync/v1/{endpoint}"
    )
    try:
        uvicorn.run(app, host=cfg.server.host, port=cfg.server.port, log_level="info")
    finally:
        if watcher:
            watcher.stop()
        live = getattr(app.state, "runtime", runtime)
        if hasattr(live, "stop"):
            live.stop()


def _download(args) -> None:
    cfg = config_mod.load(args.config)
    ensure_model(cfg)


def main() -> None:
    parser = argparse.ArgumentParser(prog="simple-local", description="bare-bones local inference")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve")
    serve.add_argument("-c", "--config", default="config.yml", help="path to config file")
    serve.add_argument("--watch", action="store_true", help="reload the model when the config or model file changes (predictor)")
    serve.set_defaults(func=_serve)

    download = sub.add_parser("download")
    download.add_argument("-c", "--config", default="config.yml", help="path to config file")
    download.set_defaults(func=_download)

    args = parser.parse_args()
    args.func(args)
