from __future__ import annotations

import asyncio
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
