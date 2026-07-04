"""Entry point for the hermes-acp-relay CLI."""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys

from aiohttp import web


def _setup_logging(level: str) -> None:
    # Logging always to stderr. The relay never uses stdout for anything.
    logging.basicConfig(
        stream=sys.stderr,
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _load_hermes_env() -> None:
    """Replicate acp_adapter/entry.py's env bootstrap."""
    from hermes_cli.env_loader import load_hermes_dotenv
    from hermes_constants import get_hermes_home

    load_hermes_dotenv(hermes_home=get_hermes_home())


def _discover_mcp_tools(logger: logging.Logger) -> None:
    """Register configured MCP tools before serving ACP sessions."""
    try:
        from tools.mcp_tool import discover_mcp_tools
        discover_mcp_tools()
    except Exception:
        logger.debug("MCP tool discovery failed at ACP relay startup", exc_info=True)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="hermes-acp-relay",
        description="WebSocket server exposing Hermes ACP for network access.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Interface to bind (default: 127.0.0.1). Set to a private-network "
        "IP that your reverse proxy can reach. Do not bind to a public "
        "interface — this endpoint is unauthenticated; auth lives at the proxy.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port to bind (default: 8765).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    parser.add_argument(
        "--exec-routes",
        help="TOML file defining command-backed /acp/<name> routes. Defaults "
        "to HERMES_ACP_RELAY_EXEC_ROUTES when set.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    exec_route_path = args.exec_routes or os.environ.get("HERMES_ACP_RELAY_EXEC_ROUTES")
    exec_routes = {}
    if exec_route_path:
        from .exec_routes import ExecRouteConfigError, load_exec_routes

        try:
            exec_routes = load_exec_routes(Path(exec_route_path))
        except ExecRouteConfigError as e:
            logger.error("Invalid exec routes config: %s", e)
            return 1

    logger.info("Bootstrapping Hermes environment")
    try:
        _load_hermes_env()
    except Exception:
        logger.exception("Failed to bootstrap Hermes environment — is hermes-agent installed?")
        return 1

    # Install the get_hermes_home shim BEFORE importing server.create_app —
    # which transitively imports upstream Hermes modules that do
    # `from hermes_constants import get_hermes_home`. The shim must be in
    # place by then so those imports bind to the patched function and pick
    # up per-WS profile overrides.
    from .server import install_profile_override_shim
    install_profile_override_shim()
    _discover_mcp_tools(logger)

    from .server import create_app

    app = create_app(exec_routes=exec_routes) if exec_routes else create_app()
    if exec_routes:
        logger.info(
            "hermes-acp-relay listening on http://%s:%d "
            "(WebSocket at /acp, sub-profile at /acp/<suffix>, "
            "exec routes: %s)",
            args.host,
            args.port,
            ", ".join(sorted(exec_routes)),
        )
    else:
        logger.info(
            "hermes-acp-relay listening on http://%s:%d "
            "(WebSocket at /acp, sub-profile at /acp/<suffix>)",
            args.host, args.port,
        )
    # aiohttp.web.run_app handles SIGINT/SIGTERM cleanly.
    web.run_app(app, host=args.host, port=args.port, print=None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
