from __future__ import annotations

import asyncio
import logging
import sys
import types

import aiohttp
from aiohttp import web
import pytest


async def _start(app):
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return runner, f"ws://127.0.0.1:{port}/acp"


@pytest.mark.asyncio
async def test_handler_unwinds_when_run_agent_ignores_stream_close(monkeypatch):
    """H1 regression: if run_agent never reacts to its streams closing, the
    pump-vs-runner race in _handle_acp still forces the handler to finish."""
    fake_acp = types.ModuleType("acp")

    async def hanging_run_agent(agent, *, input_stream, output_stream, **kwargs):
        await asyncio.Event().wait()  # would block forever without cancellation

    fake_acp.run_agent = hanging_run_agent
    monkeypatch.setitem(sys.modules, "acp", fake_acp)

    from hermes_acp_relay.server import _handle_acp

    class StubAgent:
        pass

    app = web.Application()
    app["agent_cls"] = StubAgent

    completed = asyncio.Event()

    async def tracked_handler(request):
        try:
            return await _handle_acp(request)
        finally:
            completed.set()

    app.router.add_get("/acp", tracked_handler)

    runner, url = await _start(app)
    try:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(url) as ws:
                await ws.close()
        await asyncio.wait_for(completed.wait(), timeout=5.0)
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_prompt_lock_released_promptly_when_ws_disconnects_mid_prompt(
    monkeypatch,
):
    """Regression for acab-ansible#567 _prompt_lock cascade.

    If a runner_task is holding _prompt_lock when the client WS disconnects
    (e.g., on a slow Honcho dialectic call), the handler's cancellation path
    must release the lock promptly so the next connection's prompt() can
    proceed. Previously a 2-second EOF-unwind grace let the lock stay held
    for the full grace duration before cancellation fired; now cancellation
    on pump-completion is immediate.
    """
    import time

    from hermes_acp_relay.server import _prompt_lock

    # Sanity: the lock must start free. If it doesn't, a previous test
    # leaked state — fail loudly here rather than producing misleading
    # results downstream.
    assert not _prompt_lock.locked(), (
        "_prompt_lock leaked from a previous test"
    )

    fake_acp = types.ModuleType("acp")

    async def lock_holder_run_agent(agent, *, input_stream, output_stream, **kwargs):
        # Simulate SerializedHermesACPAgent.prompt() holding _prompt_lock
        # across a slow upstream call (e.g. a Honcho dialectic that hangs).
        async with _prompt_lock:
            await asyncio.Event().wait()  # blocks forever without cancellation

    fake_acp.run_agent = lock_holder_run_agent
    monkeypatch.setitem(sys.modules, "acp", fake_acp)

    from hermes_acp_relay.server import _handle_acp

    class StubAgent:
        pass

    app = web.Application()
    app["agent_cls"] = StubAgent

    completed = asyncio.Event()

    async def tracked_handler(request):
        try:
            return await _handle_acp(request)
        finally:
            completed.set()

    app.router.add_get("/acp", tracked_handler)

    runner, url = await _start(app)
    try:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(url) as ws:
                # Give the runner a moment to acquire _prompt_lock.
                await asyncio.sleep(0.1)
                assert _prompt_lock.locked(), (
                    "runner_task should have acquired _prompt_lock by now"
                )
                # Disconnect mid-prompt — pumps will exit first, triggering
                # the cancellation path under test.
                t_disconnect = time.monotonic()
                await ws.close()
            await asyncio.wait_for(completed.wait(), timeout=5.0)
            elapsed = time.monotonic() - t_disconnect

        # The lock must be released so the next connection can proceed.
        assert not _prompt_lock.locked(), (
            "_prompt_lock should be released after WS disconnect; "
            "regression for acab-ansible#567 cascade"
        )
        # The handler should have unwound quickly. With the previous
        # 2-second EOF-unwind grace, this typically took ~2s. With immediate
        # cancellation, well under 1s. 1.5s threshold leaves CI flake margin
        # while still catching a regression that re-introduces the grace.
        assert elapsed < 1.5, (
            f"handler took {elapsed:.2f}s to unwind after WS disconnect; "
            f"expected < 1.5s. Suggests the EOF-unwind grace was "
            f"re-introduced or the cancellation path is not propagating "
            f"into the prompt() coroutine promptly."
        )
    finally:
        # Defensive cleanup: release the lock if the test somehow left it
        # held, so subsequent tests in the module don't deadlock on it.
        if _prompt_lock.locked():
            _prompt_lock.release()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_prompt_lock_acquisition_times_out_when_previous_holder_wedged(
    monkeypatch, caplog
):
    """Belt-and-suspenders for the acab-ansible#567 cascade.

    If _prompt_lock is held by a wedged earlier prompt — i.e., the primary
    recovery path in _handle_acp (PR #3) failed to release the lock — a
    queued caller must time out after _PROMPT_LOCK_TIMEOUT_S rather than
    waiting forever. The caller receives a refusal PromptResponse so its
    client can decide how to recover.
    """
    import logging
    import time

    import hermes_acp_relay.server as srv

    # Reduce the timeout so the test completes quickly. Production default
    # is 60s; 0.3s is enough to exercise the path.
    monkeypatch.setattr(srv, "_PROMPT_LOCK_TIMEOUT_S", 0.3)

    assert not srv._prompt_lock.locked(), (
        "_prompt_lock leaked from a previous test"
    )

    cls = srv._build_serialized_agent_class()
    agent = cls()

    # Simulate a wedged earlier prompt by holding the lock and never
    # releasing within the test's bound.
    await srv._prompt_lock.acquire()
    try:
        with caplog.at_level(logging.ERROR, logger="hermes_acp_relay.server"):
            t_start = time.monotonic()
            result = await agent.prompt([], "test-session-abc123")
            elapsed = time.monotonic() - t_start

        # Must have timed out at roughly the configured bound (~0.3s with
        # margin for asyncio scheduling).
        assert 0.2 < elapsed < 0.8, (
            f"expected ~0.3s acquisition timeout, got {elapsed:.2f}s"
        )

        # Must return a refusal PromptResponse, not raise or hang.
        assert hasattr(result, "stop_reason"), (
            f"expected PromptResponse, got {type(result).__name__}: {result!r}"
        )
        assert result.stop_reason == "refusal", (
            f"expected stop_reason='refusal', got {result.stop_reason!r}"
        )

        # Must log an error naming the session id so an operator can
        # correlate the timeout to a specific stuck session.
        timeout_logs = [
            r for r in caplog.records
            if "prompt_lock acquisition timed out" in r.message
        ]
        assert timeout_logs, "expected a timeout error log"
        assert "test-session-abc123" in timeout_logs[0].message, (
            f"expected session id in log; got: {timeout_logs[0].message!r}"
        )
    finally:
        # Defensive cleanup: release the lock if still held.
        if srv._prompt_lock.locked():
            srv._prompt_lock.release()


@pytest.mark.asyncio
async def test_handler_logs_and_returns_when_run_agent_raises_mid_execution(
    monkeypatch, caplog
):
    """If run_agent raises during its main execution (before pumps complete),
    the handler must log the crash via runner_task.exception() and return
    cleanly — not let the exception escape to aiohttp.

    Note: this used to test the run-agent-raises-during-EOF-cleanup path,
    where the handler granted a 2s grace before cancelling. That grace is
    gone (see acab-ansible#567 lock cascade fix), so cleanup-raises are no
    longer possible — runner_task is cancelled before reaching its own
    cleanup. This test now covers the more general 'run_agent raises during
    execution' contract, which is still meaningful.
    """
    fake_acp = types.ModuleType("acp")

    async def crashing_run_agent(agent, *, input_stream, output_stream, **kwargs):
        # Raise on the coroutine's first scheduled iteration. runner_task
        # transitions to done before the pumps notice the impending WS
        # close, so FIRST_COMPLETED returns with runner_task in `done` and
        # the logging branch fires.
        raise RuntimeError("mid-execution crash")

    fake_acp.run_agent = crashing_run_agent
    monkeypatch.setitem(sys.modules, "acp", fake_acp)

    from hermes_acp_relay.server import _handle_acp

    class StubAgent:
        pass

    app = web.Application()
    app["agent_cls"] = StubAgent

    handler_raised = asyncio.Event()
    completed = asyncio.Event()

    async def tracked_handler(request):
        try:
            return await _handle_acp(request)
        except Exception:
            handler_raised.set()
            raise
        finally:
            completed.set()

    app.router.add_get("/acp", tracked_handler)

    runner, url = await _start(app)
    try:
        with caplog.at_level(logging.ERROR, logger="hermes_acp_relay.server"):
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(url) as ws:
                    await ws.close()
            await asyncio.wait_for(completed.wait(), timeout=5.0)
    finally:
        await runner.cleanup()

    assert not handler_raised.is_set(), \
        "handler must not let run_agent's exception escape to aiohttp"
    crash_logs = [
        r for r in caplog.records
        if r.name == "hermes_acp_relay.server" and "ACP handler crashed" in r.message
    ]
    assert crash_logs, "expected an 'ACP handler crashed' error log"
    assert crash_logs[0].exc_info and crash_logs[0].exc_info[0] is RuntimeError


# ---------------------------------------------------------------------------
# Per-WS profile override (ContextVar shim + /acp/<suffix> route)
#
# Background: the relay serves multiple ACP consumers from one process. The
# Honcho policy layer redesign requires each consumer's WS connection to
# load a different Hermes sub-profile (e.g., voice sessions use a different
# honcho.json than the default text-channel profile). Mechanism: a ContextVar
# carries the per-WS profile-home override; a shim wraps
# hermes_constants.get_hermes_home() to consult it; the /acp/<suffix> route
# sets the ContextVar based on URL path before instantiating the agent.
# ---------------------------------------------------------------------------


@pytest.fixture
def fresh_profile_shim_state(monkeypatch):
    """Reset the relay's module-level shim state so each test starts clean.

    Without this fixture, `install_profile_override_shim()` is idempotent
    only by virtue of an early-return guard; subsequent tests would inherit
    whatever `_ORIGINAL_GET_HERMES_HOME` was set to in the previous test.
    """
    import hermes_acp_relay.server as srv
    monkeypatch.setattr(srv, "_ORIGINAL_GET_HERMES_HOME", None)
    yield


def test_install_profile_override_shim_is_idempotent(fresh_profile_shim_state):
    """Calling install_profile_override_shim() twice does not re-wrap the
    function; the captured _ORIGINAL stays as the truly-original."""
    import hermes_constants

    from hermes_acp_relay.server import install_profile_override_shim

    truly_original = hermes_constants.get_hermes_home

    install_profile_override_shim()
    first_patched = hermes_constants.get_hermes_home
    assert first_patched is not truly_original, (
        "first install should have patched the function"
    )

    install_profile_override_shim()
    second_patched = hermes_constants.get_hermes_home
    assert second_patched is first_patched, (
        "second install should be a no-op; got a re-patched function"
    )


def test_get_hermes_home_returns_override_when_contextvar_set(
    fresh_profile_shim_state, tmp_path, monkeypatch,
):
    """When _HERMES_HOME_OVERRIDE is set, the shimmed get_hermes_home()
    returns the override path instead of the underlying default."""
    import hermes_constants

    from hermes_acp_relay.server import _HERMES_HOME_OVERRIDE, install_profile_override_shim

    default_home = tmp_path / "default-home"
    default_home.mkdir()
    override_home = tmp_path / "override-home"
    override_home.mkdir()

    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: default_home)
    install_profile_override_shim()

    # No override set yet — falls through to original.
    assert hermes_constants.get_hermes_home() == default_home

    token = _HERMES_HOME_OVERRIDE.set(override_home)
    try:
        assert hermes_constants.get_hermes_home() == override_home
    finally:
        _HERMES_HOME_OVERRIDE.reset(token)

    # After reset, falls back to default again.
    assert hermes_constants.get_hermes_home() == default_home


@pytest.mark.asyncio
async def test_profile_override_isolates_between_concurrent_tasks(
    fresh_profile_shim_state, tmp_path, monkeypatch,
):
    """Two concurrent asyncio Tasks setting different overrides see each
    its own value — ContextVar semantics guarantee no cross-task leak.
    This is the property that makes the shim safe for concurrent WS
    connections in one relay process.
    """
    import hermes_constants

    from hermes_acp_relay.server import _HERMES_HOME_OVERRIDE, install_profile_override_shim

    default_home = tmp_path / "default-home"
    default_home.mkdir()
    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: default_home)
    install_profile_override_shim()

    task_a_home = tmp_path / "task-a"
    task_a_home.mkdir()
    task_b_home = tmp_path / "task-b"
    task_b_home.mkdir()

    observations_a: list[Path] = []
    observations_b: list[Path] = []

    async def task_a():
        _HERMES_HOME_OVERRIDE.set(task_a_home)
        # Yield to allow task_b to set its own override.
        await asyncio.sleep(0)
        observations_a.append(hermes_constants.get_hermes_home())
        await asyncio.sleep(0)
        observations_a.append(hermes_constants.get_hermes_home())

    async def task_b():
        _HERMES_HOME_OVERRIDE.set(task_b_home)
        await asyncio.sleep(0)
        observations_b.append(hermes_constants.get_hermes_home())
        await asyncio.sleep(0)
        observations_b.append(hermes_constants.get_hermes_home())

    await asyncio.gather(task_a(), task_b())

    assert all(p == task_a_home for p in observations_a), (
        f"task_a saw cross-task leak: {observations_a}"
    )
    assert all(p == task_b_home for p in observations_b), (
        f"task_b saw cross-task leak: {observations_b}"
    )


@pytest.mark.asyncio
async def test_acp_subprofile_route_resolves_path_and_sets_contextvar(
    fresh_profile_shim_state, tmp_path, monkeypatch,
):
    """GET /acp/<suffix> resolves to ${HERMES_HOME}/profiles/<suffix>/ and
    sets the ContextVar before delegating to _handle_acp. We observe the
    override by stubbing out _handle_acp to read get_hermes_home() and
    record what it sees.
    """
    import hermes_constants

    import hermes_acp_relay.server as srv

    base = tmp_path / "hermes-home"
    base.mkdir()
    (base / "profiles").mkdir()
    voice_profile = base / "profiles" / "voice"
    voice_profile.mkdir()

    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: base)
    srv.install_profile_override_shim()

    # Replace _handle_acp with a stub that just records what get_hermes_home
    # returns inside the handler call. We can't drive a real ACP session
    # without standing up the full agent fixtures; this is enough to verify
    # the ContextVar is set BEFORE the delegation.
    observed: dict[str, Path] = {}

    async def fake_handle_acp(request):
        observed["home"] = hermes_constants.get_hermes_home()
        # Don't actually do WS work; return a stub response.
        return web.Response(text="ok")

    monkeypatch.setattr(srv, "_handle_acp", fake_handle_acp)

    app = web.Application()
    app.router.add_get(
        "/acp/{profile_suffix:" + srv._PROFILE_SUFFIX_PATTERN + "}",
        srv._handle_acp_with_profile_override,
    )

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"http://127.0.0.1:{port}/acp/voice") as resp:
                assert resp.status == 200, await resp.text()
    finally:
        await runner.cleanup()

    assert observed.get("home") == voice_profile, (
        f"expected override to {voice_profile}, got {observed.get('home')!r}"
    )

    # After the request returns, the ContextVar should be reset to default.
    assert hermes_constants.get_hermes_home() == base


@pytest.mark.asyncio
async def test_acp_subprofile_route_returns_404_for_missing_profile(
    fresh_profile_shim_state, tmp_path, monkeypatch,
):
    """If the suffix maps to a non-existent directory, the route returns
    404 and never instantiates an agent or runs ACP."""
    import hermes_constants

    import hermes_acp_relay.server as srv

    base = tmp_path / "hermes-home"
    base.mkdir()
    (base / "profiles").mkdir()
    # NOTE: no "voice" subdir created.

    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: base)
    srv.install_profile_override_shim()

    handler_called = False

    async def fake_handle_acp(request):
        nonlocal handler_called
        handler_called = True
        return web.Response(text="should not reach here")

    monkeypatch.setattr(srv, "_handle_acp", fake_handle_acp)

    app = web.Application()
    app.router.add_get(
        "/acp/{profile_suffix:" + srv._PROFILE_SUFFIX_PATTERN + "}",
        srv._handle_acp_with_profile_override,
    )
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"http://127.0.0.1:{port}/acp/voice") as resp:
                assert resp.status == 404, await resp.text()
    finally:
        await runner.cleanup()

    assert not handler_called, "_handle_acp should not have been invoked for 404"


@pytest.mark.asyncio
async def test_acp_subprofile_route_rejects_symlink_path_traversal(
    fresh_profile_shim_state, tmp_path, monkeypatch,
):
    """A profile suffix whose resolved directory is a symlink escaping
    profiles_root must be rejected with 404. The URL regex already
    restricts the suffix shape; this catches the symlink-escape case
    that the regex alone can't.
    """
    import hermes_constants

    import hermes_acp_relay.server as srv

    base = tmp_path / "hermes-home"
    base.mkdir()
    (base / "profiles").mkdir()
    # Create an "escape" directory OUTSIDE base, then symlink it into
    # profiles/ so the suffix `escape` would resolve outside profiles_root.
    outside = tmp_path / "outside"
    outside.mkdir()
    (base / "profiles" / "escape").symlink_to(outside)

    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: base)
    srv.install_profile_override_shim()

    handler_called = False

    async def fake_handle_acp(request):
        nonlocal handler_called
        handler_called = True
        return web.Response(text="should not reach here")

    monkeypatch.setattr(srv, "_handle_acp", fake_handle_acp)

    app = web.Application()
    app.router.add_get(
        "/acp/{profile_suffix:" + srv._PROFILE_SUFFIX_PATTERN + "}",
        srv._handle_acp_with_profile_override,
    )
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"http://127.0.0.1:{port}/acp/escape") as resp:
                assert resp.status == 404, await resp.text()
    finally:
        await runner.cleanup()

    assert not handler_called, (
        "_handle_acp should not have been invoked when the suffix "
        "resolves outside profiles_root"
    )


def test_default_acp_route_unchanged_when_no_subprofile_used(fresh_profile_shim_state):
    """create_app() still registers /acp as the original handler. Sub-profile
    routing is purely additive — existing Obsidian Agent Client and
    hermes-acp-bridge clients keep working without changes.
    """
    # Stub upstream so create_app() can run without a real Hermes install.
    import sys
    import types

    fake_acp_adapter = types.ModuleType("acp_adapter")
    fake_acp_adapter_server = types.ModuleType("acp_adapter.server")

    class _StubAgent:
        pass

    fake_acp_adapter_server.HermesACPAgent = _StubAgent
    fake_acp_adapter.server = fake_acp_adapter_server
    sys.modules["acp_adapter"] = fake_acp_adapter
    sys.modules["acp_adapter.server"] = fake_acp_adapter_server

    try:
        from hermes_acp_relay.server import create_app

        app = create_app()
        routes = [(r.method, r.resource.canonical) for r in app.router.routes()]
        assert ("GET", "/acp") in routes, (
            f"default /acp route missing: {routes}"
        )
        # Sub-profile route present too (dynamic resource — canonical includes
        # the placeholder name).
        assert any("/acp/{profile_suffix}" == c for _m, c in routes), (
            f"/acp/{{profile_suffix}} route missing: {routes}"
        )
    finally:
        sys.modules.pop("acp_adapter", None)
        sys.modules.pop("acp_adapter.server", None)
