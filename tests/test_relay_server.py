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
async def test_handler_logs_and_returns_when_run_agent_raises_during_cleanup(
    monkeypatch, caplog
):
    """If a pump finishes first and run_agent then raises during EOF/cleanup,
    the handler must log the crash via runner_task.exception() and return
    cleanly — not let the exception escape to aiohttp."""
    fake_acp = types.ModuleType("acp")

    async def crashing_run_agent(agent, *, input_stream, output_stream, **kwargs):
        # Block until the server EOFs our output stream (pump-driven shutdown),
        # then raise — mimicking run_agent crashing during its own cleanup.
        await output_stream.read()
        raise RuntimeError("cleanup crash")

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
# Streaming patch tests: SerializedHermesACPAgent must inject stream_callback
# into the underlying agent.run_conversation() so token deltas are delivered
# as agent_message_chunk updates during the turn, not as one tail-end chunk
# after run_conversation returns. (See: V2 voice agent Hermes integration plan.)
# ---------------------------------------------------------------------------


class _FakeAgent:
    """Stand-in for AIAgent — captures kwargs to run_conversation and fires deltas."""

    def __init__(self, deltas=("hello ", "world"), final_response="hello world"):
        self._deltas = list(deltas)
        self._final_response = final_response
        # Upstream's HermesACPAgent.prompt() sets these before invoking run_conversation.
        self.message_callback = None
        self.thinking_callback = None
        self.step_callback = None
        self.tool_progress_callback = None
        self.captured_kwargs = None

    def run_conversation(self, *args, **kwargs):
        self.captured_kwargs = kwargs
        cb = kwargs.get("stream_callback")
        if cb is not None:
            for delta in self._deltas:
                cb(delta)
        return {"final_response": self._final_response, "messages": []}


class _FakeConn:
    """Stand-in for acp.AgentSideConnection — records all session_update calls."""

    def __init__(self):
        self.updates = []  # list of (session_id, update_kind, payload)

    async def session_update(self, session_id, update):
        kind = getattr(update, "session_update", None) or update.__class__.__name__
        text = None
        content = getattr(update, "content", None)
        if content is not None:
            text = getattr(content, "text", None)
        self.updates.append((session_id, kind, text if text is not None else update))


class _FakeSessionState:
    def __init__(self, agent):
        self.agent = agent


class _FakeSessionManager:
    def __init__(self, state):
        self._state = state

    def get_session(self, session_id):
        return self._state


def _install_fake_hermes_parent(monkeypatch, fake_agent, fake_conn):
    """Install a minimal stand-in for acp_adapter.server.HermesACPAgent.

    The stand-in faithfully reproduces the bits of upstream's prompt() flow
    we depend on: it (a) looks up the agent from session_manager, (b) sets
    message_callback before running, (c) runs run_conversation in an executor
    thread, and (d) emits a tail-end update_agent_message_text chunk with
    final_response if non-empty. That's all our patch interacts with.
    """
    import sys
    import types

    parent_holder = {}

    class FakeHermesACPAgent:
        def __init__(self):
            self.session_manager = _FakeSessionManager(_FakeSessionState(fake_agent))
            self._conn = fake_conn

        async def prompt(self, prompt_blocks, session_id, **kwargs):
            state = self.session_manager.get_session(session_id)
            if state is None:
                return "refusal"
            agent = state.agent
            loop = asyncio.get_running_loop()

            # Mimic upstream: wire message_callback to push agent_message_chunk
            # updates over the conn. This is the same shape make_message_cb
            # returns in acp_adapter/events.py.
            def message_cb(text):
                if not text:
                    return
                update = types.SimpleNamespace(
                    session_update="agent_message_chunk",
                    content=types.SimpleNamespace(text=text),
                )
                fut = asyncio.run_coroutine_threadsafe(
                    fake_conn.session_update(session_id, update), loop
                )
                fut.result(timeout=5)

            agent.message_callback = message_cb

            result = await loop.run_in_executor(
                None,
                lambda: agent.run_conversation(
                    user_message="hi",
                    conversation_history=[],
                    task_id=session_id,
                ),
            )

            # Mimic upstream tail-end emit. This is the line our patch must
            # neutralize when streaming actually delivered the response.
            final = result.get("final_response", "") if isinstance(result, dict) else ""
            if final:
                update = types.SimpleNamespace(
                    session_update="agent_message_chunk",
                    content=types.SimpleNamespace(text=final),
                )
                await fake_conn.session_update(session_id, update)
            return "prompt-response"

    parent_holder["cls"] = FakeHermesACPAgent

    # Install a fake acp_adapter.server module that exports FakeHermesACPAgent.
    fake_acp_adapter = types.ModuleType("acp_adapter")
    fake_acp_adapter_server = types.ModuleType("acp_adapter.server")
    fake_acp_adapter_server.HermesACPAgent = FakeHermesACPAgent
    fake_acp_adapter.server = fake_acp_adapter_server
    monkeypatch.setitem(sys.modules, "acp_adapter", fake_acp_adapter)
    monkeypatch.setitem(sys.modules, "acp_adapter.server", fake_acp_adapter_server)

    return parent_holder["cls"]


@pytest.mark.asyncio
async def test_streaming_patch_injects_stream_callback(monkeypatch):
    """Patch must pass stream_callback through to run_conversation so token
    deltas reach the client mid-turn instead of arriving as one chunk after
    the turn ends."""
    fake_agent = _FakeAgent(deltas=("Let me ", "check ", "your calendar."))
    fake_conn = _FakeConn()
    _install_fake_hermes_parent(monkeypatch, fake_agent, fake_conn)

    from hermes_acp_relay.server import _build_serialized_agent_class

    agent_cls = _build_serialized_agent_class()
    serialized = agent_cls()
    await serialized.prompt([], "session-1")

    # stream_callback must have been wired in.
    assert fake_agent.captured_kwargs is not None, "run_conversation was never called"
    assert fake_agent.captured_kwargs.get("stream_callback") is not None, \
        "stream_callback was not injected into run_conversation"


@pytest.mark.asyncio
async def test_streaming_patch_delivers_deltas_as_chunks(monkeypatch):
    """Each stream delta must produce a distinct agent_message_chunk update
    on the conn — that's what gives LiveKit TTS audio to speak before the
    full response is synthesized."""
    deltas = ("Let me ", "check ", "your calendar.")
    fake_agent = _FakeAgent(deltas=deltas, final_response="Let me check your calendar.")
    fake_conn = _FakeConn()
    _install_fake_hermes_parent(monkeypatch, fake_agent, fake_conn)

    from hermes_acp_relay.server import _build_serialized_agent_class

    agent_cls = _build_serialized_agent_class()
    serialized = agent_cls()
    await serialized.prompt([], "session-1")

    chunk_texts = [
        text for (_sid, kind, text) in fake_conn.updates
        if kind == "agent_message_chunk"
    ]
    # Each delta must have arrived as its own chunk, in order.
    assert chunk_texts[: len(deltas)] == list(deltas), \
        f"expected per-delta chunks; got {chunk_texts!r}"


@pytest.mark.asyncio
async def test_streaming_patch_suppresses_duplicate_tail_emit(monkeypatch):
    """When streaming delivered the response, the upstream's tail-end
    final_response emit at acp_adapter/server.py:739-741 must be suppressed
    — otherwise the client sees the full response twice."""
    deltas = ("hello ", "world")
    final = "hello world"
    fake_agent = _FakeAgent(deltas=deltas, final_response=final)
    fake_conn = _FakeConn()
    _install_fake_hermes_parent(monkeypatch, fake_agent, fake_conn)

    from hermes_acp_relay.server import _build_serialized_agent_class

    agent_cls = _build_serialized_agent_class()
    serialized = agent_cls()
    await serialized.prompt([], "session-1")

    chunk_texts = [
        text for (_sid, kind, text) in fake_conn.updates
        if kind == "agent_message_chunk"
    ]
    # We expect exactly the two streamed deltas — NOT a trailing duplicate
    # equal to the full final_response.
    assert chunk_texts == list(deltas), \
        f"tail-end final_response chunk was not suppressed: {chunk_texts!r}"


@pytest.mark.asyncio
async def test_streaming_patch_preserves_tail_emit_when_nothing_streamed(monkeypatch):
    """If streaming delivered zero bytes (e.g. a refusal path that returns
    final_response without firing the stream callback), the upstream's
    tail-end emit must NOT be suppressed — otherwise the user gets nothing."""
    fake_agent = _FakeAgent(deltas=(), final_response="I cannot help with that.")
    fake_conn = _FakeConn()
    _install_fake_hermes_parent(monkeypatch, fake_agent, fake_conn)

    from hermes_acp_relay.server import _build_serialized_agent_class

    agent_cls = _build_serialized_agent_class()
    serialized = agent_cls()
    await serialized.prompt([], "session-1")

    chunk_texts = [
        text for (_sid, kind, text) in fake_conn.updates
        if kind == "agent_message_chunk"
    ]
    assert chunk_texts == ["I cannot help with that."], \
        f"tail-end emit was wrongly suppressed when nothing streamed: {chunk_texts!r}"


@pytest.mark.asyncio
async def test_streaming_patch_preserves_final_response_in_run_conversation_result(monkeypatch):
    """The surgical patch suppresses the tail-end emit at the transport layer,
    NOT by clearing result["final_response"]. That preserves auto-titling
    (acp_adapter/server.py:726-738), which reads final_response from the
    result dict between run_conversation returning and the tail emit."""
    fake_agent = _FakeAgent(deltas=("foo ", "bar"), final_response="foo bar")
    fake_conn = _FakeConn()
    _install_fake_hermes_parent(monkeypatch, fake_agent, fake_conn)

    # Capture the result dict observed by the run_conversation caller. We
    # spy by wrapping the agent's method after the wrapper has been
    # installed — done from inside the fake parent's prompt via a hook.
    observed_results = []
    real_run = fake_agent.run_conversation

    def spy_run(*a, **kw):
        result = real_run(*a, **kw)
        observed_results.append(dict(result) if isinstance(result, dict) else result)
        return result

    fake_agent.run_conversation = spy_run

    from hermes_acp_relay.server import _build_serialized_agent_class

    agent_cls = _build_serialized_agent_class()
    serialized = agent_cls()
    await serialized.prompt([], "session-1")

    assert observed_results, "spy never observed a run_conversation result"
    # final_response must still be present so upstream's auto-titler sees it.
    assert observed_results[-1].get("final_response") == "foo bar", \
        f"streaming patch corrupted final_response: {observed_results[-1]!r}"


@pytest.mark.asyncio
async def test_streaming_patch_restores_run_conversation_after_prompt(monkeypatch):
    """The wrapper must leave the agent's run_conversation unshadowed after
    the prompt returns, so subsequent turns don't pile patches and the
    original class-level method is restored."""
    fake_agent = _FakeAgent(deltas=("a",), final_response="a")
    fake_conn = _FakeConn()
    _install_fake_hermes_parent(monkeypatch, fake_agent, fake_conn)

    from hermes_acp_relay.server import _build_serialized_agent_class

    agent_cls = _build_serialized_agent_class()
    serialized = agent_cls()

    # _FakeAgent has run_conversation defined on the class, so no instance
    # attr exists at first.
    assert "run_conversation" not in fake_agent.__dict__

    await serialized.prompt([], "session-1")

    # After the turn, the patch must be torn down.
    assert "run_conversation" not in fake_agent.__dict__, \
        "wrapper left an instance-attr patch on agent.run_conversation"


@pytest.mark.asyncio
async def test_streaming_patch_passes_through_caller_provided_stream_callback(monkeypatch):
    """If the parent's prompt() ever starts passing stream_callback into
    run_conversation itself (e.g. a future upstream that wires its own
    streaming), our wrapper must respect that: not clobber it and not
    suppress final_response, since the caller owns delivery semantics."""
    import sys
    import types

    captured_by_caller = []

    def caller_cb(text):
        captured_by_caller.append(text)

    fake_agent = _FakeAgent(deltas=("x", "y"), final_response="xy")
    fake_conn = _FakeConn()

    # Install a parent that, unlike the default fake parent, DOES pass
    # stream_callback to run_conversation. This is what an upstream version
    # that owns its own streaming would do.
    class FakeHermesACPAgent:
        def __init__(self):
            self.session_manager = _FakeSessionManager(_FakeSessionState(fake_agent))
            self._conn = fake_conn

        async def prompt(self, prompt_blocks, session_id, **kwargs):
            state = self.session_manager.get_session(session_id)
            agent = state.agent
            # No message_callback wiring — the parent is handling streaming
            # itself via stream_callback.
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                lambda: agent.run_conversation(
                    user_message="hi",
                    task_id=session_id,
                    stream_callback=caller_cb,
                ),
            )
            # Caller-driven tail emit (still fires — this test asserts the
            # wrapper does NOT suppress it when caller provided the cb).
            final = result.get("final_response", "") if isinstance(result, dict) else ""
            if final:
                update = types.SimpleNamespace(
                    session_update="agent_message_chunk",
                    content=types.SimpleNamespace(text=final),
                )
                await fake_conn.session_update(session_id, update)
            return "prompt-response"

    fake_acp_adapter = types.ModuleType("acp_adapter")
    fake_acp_adapter_server = types.ModuleType("acp_adapter.server")
    fake_acp_adapter_server.HermesACPAgent = FakeHermesACPAgent
    fake_acp_adapter.server = fake_acp_adapter_server
    monkeypatch.setitem(sys.modules, "acp_adapter", fake_acp_adapter)
    monkeypatch.setitem(sys.modules, "acp_adapter.server", fake_acp_adapter_server)

    from hermes_acp_relay.server import _build_serialized_agent_class

    agent_cls = _build_serialized_agent_class()
    serialized = agent_cls()
    await serialized.prompt([], "session-1")

    # The caller's stream_callback received the deltas — wrapper didn't clobber.
    assert captured_by_caller == ["x", "y"], \
        f"caller-supplied stream_callback was clobbered: {captured_by_caller!r}"
    # The tail emit fired (wrapper didn't suppress because caller owns delivery).
    chunk_texts = [
        text for (_sid, kind, text) in fake_conn.updates
        if kind == "agent_message_chunk"
    ]
    assert chunk_texts == ["xy"], \
        f"tail emit was wrongly suppressed for caller-owned stream: {chunk_texts!r}"
