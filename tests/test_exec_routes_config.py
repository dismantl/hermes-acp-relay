from __future__ import annotations

from pathlib import Path

import pytest

from hermes_acp_relay.exec_routes import (
    ExecRoute,
    ExecRouteConfigError,
    load_exec_routes,
)


def test_load_exec_routes_parses_valid_route(tmp_path: Path) -> None:
    config_path = tmp_path / "exec-routes.toml"
    config_path.write_text(
        "[routes.my-agent]\n"
        'command = ["python3", "/opt/tools/my_agent.py"]\n'
        'env = { MY_AGENT_CONFIG = "/etc/my-agent.toml" }\n'
        'cwd = "/opt/tools"\n'
    )

    routes = load_exec_routes(config_path)

    assert routes == {
        "my-agent": ExecRoute(
            name="my-agent",
            command=("python3", "/opt/tools/my_agent.py"),
            env={"MY_AGENT_CONFIG": "/etc/my-agent.toml"},
            cwd="/opt/tools",
        )
    }


def test_load_exec_routes_defaults_optional_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "exec-routes.toml"
    config_path.write_text(
        "[routes.worker]\n"
        'command = ["python3", "-m", "worker"]\n'
    )

    route = load_exec_routes(config_path)["worker"]

    assert route.env == {}
    assert route.cwd is None


def test_load_exec_routes_rejects_invalid_route_name(tmp_path: Path) -> None:
    config_path = tmp_path / "exec-routes.toml"
    config_path.write_text(
        "[routes.BadName]\n"
        'command = ["python3", "-m", "worker"]\n'
    )

    with pytest.raises(ExecRouteConfigError, match="route name"):
        load_exec_routes(config_path)


def test_load_exec_routes_rejects_empty_command(tmp_path: Path) -> None:
    config_path = tmp_path / "exec-routes.toml"
    config_path.write_text("[routes.worker]\ncommand = []\n")

    with pytest.raises(ExecRouteConfigError, match="command"):
        load_exec_routes(config_path)


def test_load_exec_routes_rejects_non_list_command(tmp_path: Path) -> None:
    config_path = tmp_path / "exec-routes.toml"
    config_path.write_text(
        "[routes.worker]\n"
        'command = "python3 -m worker"\n'
    )

    with pytest.raises(ExecRouteConfigError, match="command"):
        load_exec_routes(config_path)


def test_load_exec_routes_rejects_unknown_route_keys(tmp_path: Path) -> None:
    config_path = tmp_path / "exec-routes.toml"
    config_path.write_text(
        "[routes.worker]\n"
        'command = ["python3", "-m", "worker"]\n'
        "timeout = 10\n"
    )

    with pytest.raises(ExecRouteConfigError, match="unknown"):
        load_exec_routes(config_path)


def test_load_exec_routes_missing_file_gives_helpful_error(tmp_path: Path) -> None:
    with pytest.raises(ExecRouteConfigError, match="not found"):
        load_exec_routes(tmp_path / "missing.toml")


def test_load_exec_routes_allows_empty_routes_table(tmp_path: Path) -> None:
    config_path = tmp_path / "exec-routes.toml"
    config_path.write_text("[routes]\n")

    assert load_exec_routes(config_path) == {}
