"""Sanity-check: both packages are importable after install."""

import sys
import types


def test_relay_imports():
    import hermes_acp_relay  # noqa: F401
    from hermes_acp_relay import ws_streams  # noqa: F401
    from hermes_acp_relay import server  # noqa: F401


def test_bridge_imports():
    import hermes_acp_bridge  # noqa: F401
    from hermes_acp_bridge import client  # noqa: F401
    from hermes_acp_bridge import config  # noqa: F401


def test_bridge_cli_help(capsys):
    from hermes_acp_bridge.__main__ import _parse_args

    ns = _parse_args([])
    assert ns.log_level == "INFO"


def test_relay_cli_help():
    from hermes_acp_relay.__main__ import _parse_args

    ns = _parse_args([])
    assert ns.host == "127.0.0.1"
    assert ns.port == 8765


def test_relay_cli_discovers_mcp_tools_before_serving(monkeypatch):
    from hermes_acp_relay import __main__ as cli

    events: list[str] = []

    fake_server = types.ModuleType("hermes_acp_relay.server")
    fake_server.install_profile_override_shim = lambda: events.append("shim")
    fake_server.create_app = lambda: events.append("create_app") or object()

    fake_tools = types.ModuleType("tools")
    fake_tools.__path__ = []
    fake_mcp_tool = types.ModuleType("tools.mcp_tool")
    fake_mcp_tool.discover_mcp_tools = lambda: events.append("discover")

    monkeypatch.setitem(sys.modules, "hermes_acp_relay.server", fake_server)
    monkeypatch.setitem(sys.modules, "tools", fake_tools)
    monkeypatch.setitem(sys.modules, "tools.mcp_tool", fake_mcp_tool)
    monkeypatch.setattr(cli, "_load_hermes_env", lambda: events.append("env"))
    monkeypatch.setattr(cli.web, "run_app", lambda app, **kwargs: events.append("run_app"))

    assert cli.main(["--host", "127.0.0.1", "--port", "0"]) == 0
    assert events == ["env", "shim", "discover", "create_app", "run_app"]
