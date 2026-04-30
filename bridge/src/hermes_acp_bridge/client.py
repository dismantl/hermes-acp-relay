"""stdio <-> WebSocket pump.

OAC spawns this as its "agent command"; stdin/stdout carry ACP JSON-RPC
to/from OAC, and we proxy each line to/from a WebSocket text frame against
the relay server.
"""
from __future__ import annotations

import asyncio
import logging
import sys

import aiohttp

from .config import BridgeConfig

logger = logging.getLogger(__name__)

_MAX_MSG_SIZE = 50 * 1024 * 1024  # mirror the relay's limit
_CLEAN_CLOSE_CODES = {aiohttp.WSCloseCode.OK, aiohttp.WSCloseCode.GOING_AWAY}


async def run(cfg: BridgeConfig) -> int:
    auth = (
        aiohttp.BasicAuth(cfg.username, cfg.password)
        if cfg.username is not None and cfg.password is not None
        else None
    )

    timeout = aiohttp.ClientTimeout(total=None, sock_connect=15, sock_read=None)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            ws = await session.ws_connect(
                cfg.url,
                auth=auth,
                max_msg_size=_MAX_MSG_SIZE,
                heartbeat=30.0,
                autoclose=True,
                autoping=True,
            )
        except aiohttp.WSServerHandshakeError as e:
            logger.error("Handshake failed: HTTP %s — %s", e.status, e.message)
            return 2
        except aiohttp.ClientError as e:
            logger.error("Connection failed: %s", e)
            return 2

        exit_code = 0
        async with ws:
            logger.info("connected to %s", cfg.url)
            stdin_task = asyncio.create_task(_stdin_to_ws(ws), name="bridge-stdin")
            stdout_task = asyncio.create_task(_ws_to_stdout(ws), name="bridge-stdout")

            done, pending = await asyncio.wait(
                [stdin_task, stdout_task],
                return_when=asyncio.FIRST_COMPLETED,
            )

            stdin_closed_first = stdin_task in done and stdout_task in pending

            if stdin_closed_first:
                # stdin EOF first. Close the WS gracefully; any in-flight
                # server responses get delivered before the close ack, so
                # we let the stdout pump drain naturally.
                if not ws.closed:
                    await ws.close()
                try:
                    await asyncio.wait_for(stdout_task, timeout=5.0)
                except asyncio.TimeoutError:
                    stdout_task.cancel()
                    await asyncio.gather(stdout_task, return_exceptions=True)
            else:
                # WS closed (stdout pump finished) first. Cancel stdin.
                stdin_task.cancel()
                await asyncio.gather(stdin_task, return_exceptions=True)

            pump_exc: BaseException | None = None
            for task, label in ((stdin_task, "stdin"), (stdout_task, "stdout")):
                if task.done() and not task.cancelled():
                    exc = task.exception()
                    if exc and not isinstance(exc, (asyncio.CancelledError, ConnectionResetError)):
                        logger.error("bridge %s pump failed: %r", label, exc)
                        pump_exc = pump_exc or exc

            if not ws.closed:
                await ws.close()
            if pump_exc is not None:
                exit_code = 2
            else:
                exit_code = _bridge_exit_code(ws, url=cfg.url, initiated_locally=stdin_closed_first)
    logger.info("disconnected")
    return exit_code


async def _stdin_to_ws(ws: aiohttp.ClientWebSocketResponse) -> None:
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader(limit=_MAX_MSG_SIZE)
    protocol = asyncio.StreamReaderProtocol(reader)
    transport, _ = await loop.connect_read_pipe(lambda: protocol, sys.stdin)

    try:
        while True:
            line = await reader.readline()
            if not line:
                break
            text = line.decode("utf-8", errors="replace").rstrip("\n")
            if not text:
                continue
            if ws.closed:
                break
            await ws.send_str(text)
    finally:
        transport.close()


async def _ws_to_stdout(ws: aiohttp.ClientWebSocketResponse) -> None:
    async for msg in ws:
        if msg.type == aiohttp.WSMsgType.TEXT:
            sys.stdout.write(msg.data)
            if not msg.data.endswith("\n"):
                sys.stdout.write("\n")
            sys.stdout.flush()
        elif msg.type == aiohttp.WSMsgType.BINARY:
            data = msg.data.decode("utf-8", errors="replace")
            sys.stdout.write(data)
            if not data.endswith("\n"):
                sys.stdout.write("\n")
            sys.stdout.flush()
        elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSING, aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
            return


def _bridge_exit_code(
    ws: aiohttp.ClientWebSocketResponse,
    *,
    url: str,
    initiated_locally: bool,
) -> int:
    exc = ws.exception()
    if exc is not None:
        logger.error("WebSocket to %s closed with exception: %r", url, exc)
        return 2

    # Locally-initiated close may finish before the peer's close frame arrives,
    # leaving close_code unset — that's still a clean exit.
    if initiated_locally and ws.close_code is None:
        return 0

    if ws.close_code in _CLEAN_CLOSE_CODES:
        return 0

    logger.error("WebSocket to %s closed unexpectedly with code %s", url, ws.close_code)
    return 2
