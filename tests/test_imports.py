"""Sanity-check: both packages are importable after install."""


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
