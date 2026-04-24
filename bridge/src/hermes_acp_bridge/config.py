"""Config loader for hermes-acp-bridge.

Reads a TOML file with the relay URL and HTTP Basic auth credentials.
Kept outside the Obsidian vault so it doesn't accidentally sync.
"""
from __future__ import annotations

import logging
import os
import stat
import tomllib
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse


DEFAULT_CONFIG_PATH = Path.home() / ".config" / "hermes-acp-bridge" / "config.toml"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BridgeConfig:
    url: str
    username: str | None
    password: str | None


class ConfigError(Exception):
    pass


def _warn_if_world_or_group_readable(path: Path) -> None:
    try:
        mode = path.stat().st_mode
    except OSError:
        return
    if mode & (stat.S_IRGRP | stat.S_IROTH | stat.S_IWGRP | stat.S_IWOTH):
        logger.warning(
            "Config file %s has permissions %o — credentials may be readable by other "
            "local users. Run `chmod 600 %s` to secure it.",
            path, mode & 0o777, path,
        )


def _resolve_path(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser()
    env = os.environ.get("HERMES_BRIDGE_CONFIG")
    if env:
        return Path(env).expanduser()
    return DEFAULT_CONFIG_PATH


def load_config(explicit_path: str | None = None) -> BridgeConfig:
    path = _resolve_path(explicit_path)
    if not path.is_file():
        raise ConfigError(
            f"Config file not found at {path}. Create it with at least a 'url' "
            f"key, plus 'username' and 'password' if the relay is behind an "
            f"auth-enforcing proxy. See README for an example."
        )

    try:
        with path.open("rb") as f:
            data = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"Failed to parse {path}: {e}") from e
    except OSError as e:
        raise ConfigError(f"Failed to read {path}: {e}") from e

    _warn_if_world_or_group_readable(path)

    url = data.get("url")
    if not isinstance(url, str) or not url:
        raise ConfigError(f"{path}: 'url' is required and must be a non-empty string.")

    parsed = urlparse(url)
    if parsed.username or parsed.password:
        raise ConfigError(
            f"{path}: 'url' must not contain userinfo (user:pass@host); "
            f"use the 'username' and 'password' keys instead."
        )

    username = data.get("username")
    password = data.get("password")
    if (username is None) != (password is None):
        raise ConfigError(
            f"{path}: 'username' and 'password' must both be set, or both omitted."
        )
    if username is not None and password is not None:
        if not isinstance(username, str) or not isinstance(password, str):
            raise ConfigError(
                f"{path}: 'username' and 'password' must both be strings when set."
            )
        if username == "" or password == "":
            raise ConfigError(
                f"{path}: 'username' and 'password' must be non-empty strings when set."
            )
        if parsed.scheme not in ("wss", "https"):
            raise ConfigError(
                f"{path}: credentials require a wss:// or https:// URL (got scheme "
                f"{parsed.scheme!r}); refusing to send Basic auth in cleartext. "
                f"aiohttp's ws_connect upgrades https:// to a TLS-protected "
                f"WebSocket, so either scheme is fine."
            )

    return BridgeConfig(url=url, username=username, password=password)
