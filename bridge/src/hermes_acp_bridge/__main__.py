"""Entry point for the hermes-acp-bridge CLI."""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from .client import run
from .config import ConfigError, load_config


def _setup_logging(level: str) -> None:
    # stdout belongs to ACP JSON-RPC — all logs go to stderr.
    logging.basicConfig(
        stream=sys.stderr,
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="hermes-acp-bridge",
        description="Bridge Obsidian Agent Client's stdio ACP to a relay server.",
    )
    parser.add_argument(
        "-c", "--config",
        help="Path to config TOML (default: $HERMES_BRIDGE_CONFIG or ~/.config/hermes-acp-bridge/config.toml).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    try:
        cfg = load_config(args.config)
    except ConfigError as e:
        logger.error("%s", e)
        return 1

    try:
        return asyncio.run(run(cfg))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
