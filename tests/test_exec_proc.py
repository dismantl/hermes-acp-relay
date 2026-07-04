from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
from pathlib import Path

import aiohttp
from aiohttp import WSMsgType, web
import pytest

from hermes_acp_relay.exec_proc import run_exec_session
from hermes_acp_relay.exec_routes import ExecRoute


_FAKE_AGENT = Path(__file__).with_name("fake_stdio_agent.py")


async def _start_exec_app(route: ExecRoute):
    app = web.Application()
    completed = asyncio.Event()

    async def handler(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        try:
            await run_exec_session(ws, route)
        finally:
            completed.set()
        return ws

    app.router.add_get("/acp/worker", handler)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return runner, f"ws://127.0.0.1:{port}/acp/worker", completed


def _fake_route(name: str, mode: str, *, env: dict[str, str] | None = None) -> ExecRoute:
    return ExecRoute(
        name=name,
        command=(sys.executable, str(_FAKE_AGENT), mode),
        env=env or {},
        cwd=None,
    )


async def _receive_json(ws: aiohttp.ClientWebSocketResponse) -> dict[str, str]:
    msg = await asyncio.wait_for(ws.receive(), timeout=5.0)
    assert msg.type == WSMsgType.TEXT, msg
    return json.loads(msg.data)


@pytest.mark.asyncio
async def test_run_exec_session_round_trips_frames_in_order() -> None:
    runner, url, completed = await _start_exec_app(_fake_route("worker", "echo"))
    try:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(url) as ws:
                await ws.send_str("first")
                await ws.send_str("second\n")

                assert await _receive_json(ws) == {"echo": "first"}
                assert await _receive_json(ws) == {"echo": "second"}

                await ws.close()
        await asyncio.wait_for(completed.wait(), timeout=5.0)
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_run_exec_session_merges_route_env_into_child_env() -> None:
    runner, url, completed = await _start_exec_app(
        _fake_route("env-route", "env", env={"PROBE_VAR": "from-route"})
    )
    try:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(url) as ws:
                assert await _receive_json(ws) == {"env": "from-route"}
                await ws.close()
        await asyncio.wait_for(completed.wait(), timeout=5.0)
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_run_exec_session_allows_large_stdout_lines() -> None:
    runner, url, completed = await _start_exec_app(_fake_route("large-route", "large"))
    try:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(url) as ws:
                await ws.send_str("large")
                reply = await _receive_json(ws)
                assert reply == {"echo": "x" * 70_000}
                await ws.close()
        await asyncio.wait_for(completed.wait(), timeout=5.0)
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_run_exec_session_closes_ws_when_child_exits() -> None:
    runner, url, completed = await _start_exec_app(_fake_route("exit-route", "exit-now"))
    try:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(url) as ws:
                msg = await asyncio.wait_for(ws.receive(), timeout=5.0)
                assert msg.type in {WSMsgType.CLOSE, WSMsgType.CLOSED}, msg
        await asyncio.wait_for(completed.wait(), timeout=5.0)
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_run_exec_session_terminates_child_when_ws_closes(tmp_path: Path) -> None:
    pid_file = tmp_path / "child.pid"
    runner, url, completed = await _start_exec_app(
        _fake_route("hang-route", "hang", env={"PROBE_PID_FILE": str(pid_file)})
    )
    try:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(url) as ws:
                for _ in range(50):
                    if pid_file.exists():
                        break
                    await asyncio.sleep(0.1)
                assert pid_file.exists(), "fake child did not write its pid"
                pid = int(pid_file.read_text())
                await ws.close()

        await asyncio.wait_for(completed.wait(), timeout=5.0)

        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    finally:
        await runner.cleanup()
        if pid_file.exists():
            try:
                os.kill(int(pid_file.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.asyncio
async def test_run_exec_session_closes_ws_and_logs_when_spawn_fails(caplog) -> None:
    route = ExecRoute(
        name="broken-route",
        command=("/definitely/not/a/real/command",),
        env={},
        cwd=None,
    )
    runner, url, completed = await _start_exec_app(route)
    try:
        with caplog.at_level(logging.ERROR, logger="hermes_acp_relay.exec_proc"):
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(url) as ws:
                    msg = await asyncio.wait_for(ws.receive(), timeout=5.0)
                    assert msg.type in {WSMsgType.CLOSE, WSMsgType.CLOSED}, msg
            await asyncio.wait_for(completed.wait(), timeout=5.0)
    finally:
        await runner.cleanup()

    assert any(
        record.name == "hermes_acp_relay.exec_proc"
        and "broken-route" in record.getMessage()
        and "failed to spawn" in record.getMessage()
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_run_exec_session_logs_child_stderr_with_route_prefix(caplog) -> None:
    runner, url, completed = await _start_exec_app(_fake_route("stderr-route", "stderr"))
    try:
        with caplog.at_level(logging.INFO, logger="hermes_acp_relay.exec_proc"):
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(url) as ws:
                    await ws.send_str("ping")
                    assert await _receive_json(ws) == {"echo": "ping"}
                    await ws.close()
            await asyncio.wait_for(completed.wait(), timeout=5.0)
    finally:
        await runner.cleanup()

    assert any(
        record.name == "hermes_acp_relay.exec_proc"
        and "[stderr-route] stderr probe" in record.getMessage()
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_run_exec_session_keeps_running_when_child_closes_stderr() -> None:
    runner, url, completed = await _start_exec_app(
        _fake_route("close-stderr-route", "close-stderr")
    )
    try:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(url) as ws:
                await asyncio.sleep(0.2)
                await ws.send_str("ping")
                assert await _receive_json(ws) == {"echo": "ping"}
                await ws.close()
        await asyncio.wait_for(completed.wait(), timeout=5.0)
    finally:
        await runner.cleanup()
