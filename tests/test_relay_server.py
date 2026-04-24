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
