from __future__ import annotations

import asyncio
import sys

from aiohttp import WSCloseCode, web
import pytest

from hermes_acp_bridge.client import _stdin_to_ws, run
from hermes_acp_bridge.config import BridgeConfig


async def _start_test_server(handler):
    app = web.Application()
    app.router.add_get("/acp", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return runner, f"ws://127.0.0.1:{port}/acp"


@pytest.mark.asyncio
async def test_run_returns_error_on_abnormal_websocket_close(monkeypatch):
    async def handler(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.close(code=WSCloseCode.POLICY_VIOLATION, message=b"blocked")
        return ws

    async def fake_stdin_to_ws(ws):
        while not ws.closed:
            await asyncio.sleep(0.01)

    monkeypatch.setattr("hermes_acp_bridge.client._stdin_to_ws", fake_stdin_to_ws)

    runner, url = await _start_test_server(handler)
    try:
        rc = await run(BridgeConfig(url=url, username=None, password=None))
    finally:
        await runner.cleanup()

    assert rc == 2


@pytest.mark.asyncio
async def test_run_returns_success_on_clean_websocket_close(monkeypatch):
    async def handler(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.close(code=WSCloseCode.OK, message=b"done")
        return ws

    async def fake_stdin_to_ws(ws):
        while not ws.closed:
            await asyncio.sleep(0.01)

    monkeypatch.setattr("hermes_acp_bridge.client._stdin_to_ws", fake_stdin_to_ws)

    runner, url = await _start_test_server(handler)
    try:
        rc = await run(BridgeConfig(url=url, username=None, password=None))
    finally:
        await runner.cleanup()

    assert rc == 0


@pytest.mark.asyncio
async def test_stdin_to_ws_closes_read_pipe_transport_on_cancellation(monkeypatch):
    closed = False

    class FakeTransport:
        def close(self):
            nonlocal closed
            closed = True

    class FakeLoop:
        async def connect_read_pipe(self, protocol_factory, pipe):
            return FakeTransport(), protocol_factory()

    class FakeWS:
        closed = False

        async def send_str(self, text):
            return None

    async def blocking_readline(self):
        await asyncio.sleep(3600)

    monkeypatch.setattr(asyncio, "get_running_loop", lambda: FakeLoop())
    monkeypatch.setattr(asyncio.StreamReader, "readline", blocking_readline)

    task = asyncio.create_task(_stdin_to_ws(FakeWS()))
    await asyncio.sleep(0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert closed is True
