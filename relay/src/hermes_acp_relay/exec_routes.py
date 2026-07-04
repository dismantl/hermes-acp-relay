"""Configuration model and loader for command-backed ACP routes."""
from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ExecRouteConfigError(ValueError):
    """Raised when the exec-routes TOML file is missing or invalid."""


@dataclass(frozen=True)
class ExecRoute:
    name: str
    command: tuple[str, ...]
    env: dict[str, str]
    cwd: str | None


# Keep in sync with server._PROFILE_SUFFIX_PATTERN. This module deliberately
# avoids importing server.py so server -> exec_proc -> exec_routes stays acyclic.
_ROUTE_NAME_PATTERN = r"[a-z][a-z0-9-]*"
_ROUTE_KEYS = frozenset({"command", "env", "cwd"})


def load_exec_routes(path: Path) -> dict[str, ExecRoute]:
    """Load command-backed ACP routes from TOML."""
    try:
        with path.open("rb") as f:
            data = tomllib.load(f)
    except FileNotFoundError as e:
        raise ExecRouteConfigError(f"Exec routes config not found: {path}") from e
    except tomllib.TOMLDecodeError as e:
        raise ExecRouteConfigError(f"Failed to parse exec routes config {path}: {e}") from e
    except OSError as e:
        raise ExecRouteConfigError(f"Failed to read exec routes config {path}: {e}") from e

    routes = data.get("routes", {})
    if not isinstance(routes, dict):
        raise ExecRouteConfigError("'routes' must be a table")

    return {
        name: _parse_route(name, table)
        for name, table in routes.items()
    }


def _parse_route(name: str, table: Any) -> ExecRoute:
    if not re.fullmatch(_ROUTE_NAME_PATTERN, name):
        raise ExecRouteConfigError(
            f"Invalid exec route name {name!r}; must match {_ROUTE_NAME_PATTERN!r}"
        )
    if not isinstance(table, dict):
        raise ExecRouteConfigError(f"Exec route {name!r} must be a table")

    unknown_keys = set(table) - _ROUTE_KEYS
    if unknown_keys:
        keys = ", ".join(sorted(unknown_keys))
        raise ExecRouteConfigError(f"Exec route {name!r} has unknown key(s): {keys}")

    command = table.get("command")
    if (
        not isinstance(command, list)
        or not command
        or not all(isinstance(part, str) and part for part in command)
    ):
        raise ExecRouteConfigError(
            f"Exec route {name!r} command must be a non-empty list of strings"
        )

    env = table.get("env", {})
    if not isinstance(env, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in env.items()
    ):
        raise ExecRouteConfigError(
            f"Exec route {name!r} env must be a table of string values"
        )

    cwd = table.get("cwd")
    if cwd is not None and not isinstance(cwd, str):
        raise ExecRouteConfigError(f"Exec route {name!r} cwd must be a string")

    return ExecRoute(
        name=name,
        command=tuple(command),
        env=dict(env),
        cwd=cwd,
    )
