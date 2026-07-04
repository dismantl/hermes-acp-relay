from __future__ import annotations

import asyncio
import json
import logging
import sys
import types
from pathlib import Path

import aiohttp
from aiohttp import WSMsgType, web
import pytest

from hermes_acp_relay.exec_routes import ExecRoute


_FAKE_AGENT = Path(__file__).with_name("fake_stdio_agent.py")


def _fake_route(name: str, mode: str = "echo") -> ExecRoute:
    return ExecRoute(
        name=name,
        command=(sys.executable, str(_FAKE_AGENT), mode),
        env={},
        cwd=None,
    )


@pytest.fixture
def fake_acp_adapter(monkeypatch):
    fake_acp_adapter = types.ModuleType("acp_adapter")
    fake_acp_adapter_server = types.ModuleType("acp_adapter.server")

    class StubAgent:
        pass

    fake_acp_adapter_server.HermesACPAgent = StubAgent
    fake_acp_adapter.server = fake_acp_adapter_server
    monkeypatch.setitem(sys.modules, "acp_adapter", fake_acp_adapter)
    monkeypatch.setitem(sys.modules, "acp_adapter.server", fake_acp_adapter_server)


async def _start(app: web.Application, path: str):
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return runner, f"ws://127.0.0.1:{port}{path}", f"http://127.0.0.1:{port}"


@pytest.mark.asyncio
async def test_exec_route_shadows_profile_suffix_route(fake_acp_adapter) -> None:
    from hermes_acp_relay.server import create_app

    app = create_app(exec_routes={"custom": _fake_route("custom")})
    runner, url, _ = await _start(app, "/acp/custom")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(url) as ws:
                await ws.send_str("hello")
                msg = await asyncio.wait_for(ws.receive(), timeout=5.0)
                assert msg.type == WSMsgType.TEXT, msg
                assert json.loads(msg.data) == {"echo": "hello"}
                await ws.close()
    finally:
        await runner.cleanup()


def test_create_app_without_exec_routes_keeps_existing_acp_routes(
    fake_acp_adapter,
) -> None:
    from hermes_acp_relay.server import create_app

    app = create_app()
    routes = [(r.method, r.resource.canonical) for r in app.router.routes()]

    assert ("GET", "/acp") in routes
    assert any("/acp/{profile_suffix}" == c for _m, c in routes)
    assert not any(c == "/acp/{name}" for _m, c in routes)


@pytest.mark.asyncio
async def test_health_route_unaffected_by_exec_routes(fake_acp_adapter) -> None:
    from hermes_acp_relay.server import create_app

    app = create_app(exec_routes={"worker": _fake_route("worker")})
    runner, _ws_url, http_base = await _start(app, "/acp/worker")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{http_base}/health") as resp:
                assert resp.status == 200
                assert await resp.json() == {"status": "ok"}
    finally:
        await runner.cleanup()


def test_relay_cli_loads_exec_routes_from_flag(monkeypatch, tmp_path: Path) -> None:
    from hermes_acp_relay import __main__ as cli

    config_path = tmp_path / "exec-routes.toml"
    config_path.write_text(
        "[routes.worker]\n"
        f'command = ["{sys.executable}", "{_FAKE_AGENT}", "echo"]\n'
    )
    captured: dict[str, object] = {}

    fake_server = types.ModuleType("hermes_acp_relay.server")
    fake_server.install_profile_override_shim = lambda: None
    fake_server.create_app = (
        lambda *, exec_routes=None: captured.setdefault("exec_routes", exec_routes)
        or object()
    )

    monkeypatch.setitem(sys.modules, "hermes_acp_relay.server", fake_server)
    monkeypatch.setattr(cli, "_load_hermes_env", lambda: None)
    monkeypatch.setattr(cli, "_discover_mcp_tools", lambda logger: None)
    monkeypatch.setattr(cli.web, "run_app", lambda app, **kwargs: None)

    assert cli.main(["--host", "127.0.0.1", "--port", "0", "--exec-routes", str(config_path)]) == 0
    exec_routes = captured["exec_routes"]
    assert isinstance(exec_routes, dict)
    assert exec_routes["worker"].command == (sys.executable, str(_FAKE_AGENT), "echo")


def test_relay_cli_loads_exec_routes_from_env(monkeypatch, tmp_path: Path) -> None:
    from hermes_acp_relay import __main__ as cli

    config_path = tmp_path / "exec-routes.toml"
    config_path.write_text(
        "[routes.worker]\n"
        f'command = ["{sys.executable}", "{_FAKE_AGENT}", "echo"]\n'
    )
    captured: dict[str, object] = {}

    fake_server = types.ModuleType("hermes_acp_relay.server")
    fake_server.install_profile_override_shim = lambda: None
    fake_server.create_app = (
        lambda *, exec_routes=None: captured.setdefault("exec_routes", exec_routes)
        or object()
    )

    monkeypatch.setenv("HERMES_ACP_RELAY_EXEC_ROUTES", str(config_path))
    monkeypatch.setitem(sys.modules, "hermes_acp_relay.server", fake_server)
    monkeypatch.setattr(cli, "_load_hermes_env", lambda: None)
    monkeypatch.setattr(cli, "_discover_mcp_tools", lambda logger: None)
    monkeypatch.setattr(cli.web, "run_app", lambda app, **kwargs: None)

    assert cli.main(["--host", "127.0.0.1", "--port", "0"]) == 0
    exec_routes = captured["exec_routes"]
    assert isinstance(exec_routes, dict)
    assert sorted(exec_routes) == ["worker"]


def test_relay_cli_invalid_exec_routes_config_fails_fast(
    monkeypatch,
    tmp_path: Path,
    caplog,
) -> None:
    from hermes_acp_relay import __main__ as cli

    config_path = tmp_path / "exec-routes.toml"
    config_path.write_text(
        "[routes.BadName]\n"
        'command = ["python3", "-m", "worker"]\n'
    )
    run_app_called = False

    fake_server = types.ModuleType("hermes_acp_relay.server")
    fake_server.install_profile_override_shim = lambda: None
    fake_server.create_app = lambda *, exec_routes=None: object()

    def fake_run_app(app, **kwargs):
        nonlocal run_app_called
        run_app_called = True

    monkeypatch.setitem(sys.modules, "hermes_acp_relay.server", fake_server)
    monkeypatch.setattr(cli, "_load_hermes_env", lambda: None)
    monkeypatch.setattr(cli, "_discover_mcp_tools", lambda logger: None)
    monkeypatch.setattr(cli.web, "run_app", fake_run_app)

    with caplog.at_level(logging.ERROR, logger="hermes_acp_relay.__main__"):
        assert cli.main(["--host", "127.0.0.1", "--port", "0", "--exec-routes", str(config_path)]) == 1

    assert not run_app_called
    assert any("Invalid exec routes config" in record.getMessage() for record in caplog.records)
