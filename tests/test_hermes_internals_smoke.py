"""Real-integration smoke test for the hermes-agent internals the relay touches.

Unlike ``test_relay_server.py`` — which STUBS ``acp_adapter.server.HermesACPAgent``
and monkeypatches ``hermes_constants.get_hermes_home`` so it can run without the
upstream package — this module imports the REAL hermes-agent and exercises the
exact touchpoints a version bump could break:

1. ``hermes_constants.get_hermes_home`` exists and the profile-override shim
   (``server.install_profile_override_shim``) can wrap it, with the per-WS
   ContextVar override routing through it.
2. ``acp_adapter.server.HermesACPAgent`` imports and constructs.
3. ``server._build_serialized_agent_class()`` produces a real ``HermesACPAgent``
   subclass whose ``prompt()`` serialization wrapper is installed, and it
   instantiates.
4. The upstream ``prompt()`` signature still exposes ``session_id``, which the
   relay's ``SerializedHermesACPAgent.prompt`` override reads (by keyword or as
   the second positional arg).

These tests REQUIRE hermes-agent to be installed; CI installs it via
``uv sync --all-packages --locked``. There is deliberately no ``importorskip`` —
if the pinned hermes-agent is missing or an assertion here fails, the deployed
relay would break on that version. This file is therefore the compatibility gate
for bumping the ``hermes-agent`` pin in the root ``pyproject.toml``
``[tool.uv.sources]``: a pin-bump PR must not merge while any assertion here is
red.

Imports of the upstream package are kept inside the test bodies so a missing
install surfaces as a targeted failure on these tests rather than a
collection-time error for the whole file.
"""
from __future__ import annotations

import inspect

import pytest


@pytest.fixture
def fresh_shim_state(monkeypatch):
    """Start with a clean shim global and restore the real get_hermes_home.

    ``install_profile_override_shim()`` captures ``_ORIGINAL_GET_HERMES_HOME``
    once and re-binds the module attribute ``hermes_constants.get_hermes_home``.
    Reset the former so the shim installs fresh, and snapshot the latter through
    monkeypatch so the shim's global mutation is reverted after the test instead
    of leaking into other tests.
    """
    import hermes_acp_relay.server as srv
    import hermes_constants

    monkeypatch.setattr(srv, "_ORIGINAL_GET_HERMES_HOME", None)
    monkeypatch.setattr(
        hermes_constants, "get_hermes_home", hermes_constants.get_hermes_home
    )
    yield


def test_real_get_hermes_home_is_shimmable(fresh_shim_state, tmp_path, monkeypatch):
    """The real hermes_constants.get_hermes_home can be wrapped by the shim, and
    the per-WS ContextVar override routes through the wrapped function."""
    import hermes_constants

    from hermes_acp_relay.server import (
        _HERMES_HOME_OVERRIDE,
        install_profile_override_shim,
    )

    # The shim reads and re-binds this attribute; it must exist and be callable.
    assert callable(hermes_constants.get_hermes_home)

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    unpatched = hermes_constants.get_hermes_home
    install_profile_override_shim()
    assert hermes_constants.get_hermes_home is not unpatched, (
        "install_profile_override_shim() did not wrap get_hermes_home"
    )

    # No override set: falls through to the real default (resolved via HERMES_HOME).
    assert hermes_constants.get_hermes_home() == tmp_path

    # Override set: the shimmed function returns it for the lifetime of the set.
    override = tmp_path / "sub"
    token = _HERMES_HOME_OVERRIDE.set(override)
    try:
        assert hermes_constants.get_hermes_home() == override
    finally:
        _HERMES_HOME_OVERRIDE.reset(token)

    # After reset, falls back to the real default again.
    assert hermes_constants.get_hermes_home() == tmp_path


def test_real_hermes_acp_agent_imports_and_constructs(tmp_path, monkeypatch):
    """The upstream HermesACPAgent imports and constructs with no arguments."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from acp_adapter.server import HermesACPAgent

    agent = HermesACPAgent()
    assert isinstance(agent, HermesACPAgent)


def test_serialized_agent_wraps_real_hermes_acp_agent(tmp_path, monkeypatch):
    """_build_serialized_agent_class() subclasses the real HermesACPAgent,
    installs the prompt serialization wrapper, and instantiates."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from acp_adapter.server import HermesACPAgent

    from hermes_acp_relay.server import _build_serialized_agent_class

    cls = _build_serialized_agent_class()
    assert issubclass(cls, HermesACPAgent)

    # The prompt wrapper must override upstream and stay a coroutine function.
    assert cls.prompt is not HermesACPAgent.prompt, "prompt wrapper not installed"
    assert inspect.iscoroutinefunction(cls.prompt)

    inst = cls()
    assert isinstance(inst, HermesACPAgent)


def test_upstream_prompt_signature_still_exposes_session_id():
    """SerializedHermesACPAgent.prompt reads session_id from kwargs or args[1];
    guard the upstream contract that extraction relies on."""
    from acp_adapter.server import HermesACPAgent

    params = inspect.signature(HermesACPAgent.prompt).parameters
    assert "session_id" in params, (
        f"upstream HermesACPAgent.prompt no longer exposes session_id: {list(params)}"
    )
