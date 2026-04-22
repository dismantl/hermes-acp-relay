"""stdio <-> WebSocket pump.

OAC spawns this as its "agent command"; stdin/stdout carry ACP JSON-RPC
to/from OAC, and we proxy each line to/from a WebSocket text frame against
the remote relay.
"""
from __future__ import annotations

import asyncio
import logging
import sys

import aiohttp

from .config import BridgeConfig

logger = logging.getLogger(__name__)

_MAX_MSG_SIZE = 50 * 1024 * 1024  # mirror the relay's limit


async def run(cfg: BridgeConfig) -> int:
    auth = (
        aiohttp.BasicAuth(cfg.username, cfg.password)
        if cfg.username and cfg.password
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

        async with ws:
            logger.info("connected to %s", cfg.url)
            stdin_task = asyncio.create_task(_stdin_to_ws(ws), name="bridge-stdin")
            stdout_task = asyncio.create_task(_ws_to_stdout(ws), name="bridge-stdout")

            done, pending = await asyncio.wait(
                [stdin_task, stdout_task],
                return_when=asyncio.FIRST_COMPLETED,
            )

            if stdin_task in done and stdout_task in pending:
                # stdin EOF first. Close the WS gracefully; any in-flight
                # server responses get delivered before the close ack, so
                # we let the stdout pump drain naturally.
                if not ws.closed:
                    await ws.close()
                try:
                    await asyncio.wait_for(stdout_task, timeout=5.0)
                except asyncio.TimeoutError:
                    stdout_task.cancel()
            else:
                # WS closed (stdout pump finished) first. Cancel stdin.
                stdin_task.cancel()

            for task in (stdin_task, stdout_task):
                if task.done() and not task.cancelled():
                    exc = task.exception()
                    if exc and not isinstance(exc, (asyncio.CancelledError, ConnectionResetError)):
                        logger.error("bridge task failed: %r", exc)

            if not ws.closed:
                await ws.close()
    logger.info("disconnected")
    return 0


async def _stdin_to_ws(ws: aiohttp.ClientWebSocketResponse) -> None:
    """Read newline-delimited JSON-RPC from stdin, send each line as a WS text frame."""
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader(limit=_MAX_MSG_SIZE)
    protocol = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: protocol, sys.stdin)

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


async def _ws_to_stdout(ws: aiohttp.ClientWebSocketResponse) -> None:
    """Receive WS text frames, write each as a line to stdout."""
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
