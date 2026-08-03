import argparse
import logging
import sys

import uvicorn

from . import config as config_mod
from .download import ensure_model_files
from .registry import build_registry
from .reload import ReloadWatcher
from .server import create_app


def _serve(args) -> None:
    cfg = config_mod.load(args.config)
    registry = build_registry(cfg)
    app = create_app(cfg, registry, config_path=args.config)

    if not cfg.server.api_key:
        print("note: server.api_key not set — auth disabled", file=sys.stderr)

    watcher = None
    if args.watch:
        watcher = ReloadWatcher(app, args.config)
        watcher.start()

    base = f"http://{cfg.server.host}:{cfg.server.port}/v1"
    print(f"Serving {len(cfg.served_names())} model(s) at {base}:")
    for spec in cfg.models:
        if spec.kind == "llm":
            endpoint = "embeddings" if spec.embeddings else "chat/completions"
        else:
            endpoint = "predict"
        print(f"  {spec.name} ({spec.kind})  {base}/{endpoint}")
        for adapter in spec.adapters:
            print(f"  {adapter.name} (adapter of {spec.name}, scale {adapter.scale})")
    try:
        uvicorn.run(app, host=cfg.server.host, port=cfg.server.port, log_level="info")
    finally:
        if watcher:
            watcher.stop()
        app.state.registry.stop_all()


def _download(args) -> None:
    cfg = config_mod.load(args.config)
    for spec in cfg.models:
        ensure_model_files(spec)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    parser = argparse.ArgumentParser(prog="simple-local", description="bare-bones local inference")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve")
    serve.add_argument("-c", "--config", default="config.yml", help="path to config file")
    serve.add_argument("--watch", action="store_true", help="reload models when the config or model artifacts change (blue/green for llms)")
    serve.set_defaults(func=_serve)

    download = sub.add_parser("download")
    download.add_argument("-c", "--config", default="config.yml", help="path to config file")
    download.set_defaults(func=_download)

    args = parser.parse_args()
    args.func(args)
