"""Command-line entrypoint.

    llm-proxy --config /etc/llm-proxy/prod.yaml
    llm-proxy --config prod.yaml --check        # validate and exit
"""
from __future__ import annotations

import argparse
import logging
import sys

import uvicorn

from .config import ConfigError, load_config


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    # quiet down noisy libs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="llm-proxy",
        description="Slot-aware reverse proxy for llama-server backends",
    )
    parser.add_argument(
        "-c", "--config",
        default="/etc/llm-proxy/prod.yaml",
        help="path to YAML config (default: %(default)s)",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="validate the configuration file and exit",
    )
    parser.add_argument(
        "--host", default=None,
        help="override server.host from the config",
    )
    parser.add_argument(
        "--port", type=int, default=None,
        help="override server.port from the config",
    )
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config)
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2

    if args.check:
        print(f"config OK: {len(cfg.backends)} backend(s), "
              f"{len(cfg.enabled_backends)} enabled, "
              f"strategy={cfg.routing.strategy}, full_policy={cfg.routing.full_policy}")
        for b in cfg.enabled_backends:
            print(f"  - {b.name}: {b.url} slots={b.slots} weight={b.weight}")
        return 0

    _setup_logging(cfg.server.log_level)
    if args.host:
        cfg.server.host = args.host
    if args.port:
        cfg.server.port = args.port

    from .app import create_app

    app = create_app(cfg)

    uvicorn.run(
        app,
        host=cfg.server.host,
        port=cfg.server.port,
        log_level=cfg.server.log_level,
        # NOTE: keep the event loop single-threaded; in-memory routing
        # state is per-process (see README "Running multiple instances").
        workers=1,
        timeout_graceful_shutdown=10,
        access_log=False,  # we log our own access lines with request ids
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
