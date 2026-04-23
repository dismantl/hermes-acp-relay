"""Adapter between aiohttp WebSocket and asyncio StreamReader/StreamWriter.

The ACP SDK's AgentSideConnection enforces isinstance checks against
asyncio.StreamReader / asyncio.StreamWriter (acp/agent/connection.py:68),
so we can't hand it a lookalike shim. Instead we create a socket.socketpair():
one end becomes the real asyncio streams we give to ACP; the other end is
proxied byte-for-byte to the WebSocket by two background tasks.

Framing: ACP speaks line-delimited JSON-RPC. One JSON-RPC message ↔ one WS
text frame. Incoming frames get a trailing newline if the peer forgot it;
outgoing lines are stripped of their trailing newline before being sent.
"""
from __future__ import annotations

import asyncio
import logging
import socket
from dataclasses import dataclass

from aiohttp import WSMsgType, web

logger = logging.getLogger(__name__)

_STREAM_LIMIT = 50 * 1024 * 1024  # matches acp.core.DEFAULT_STDIO_BUFFER_LIMIT_BYTES


@dataclass
class AdapterHandles:
    reader: asyncio.StreamReader        # pass as run_agent's output_stream (client -> agent)
    writer: asyncio.StreamWriter        # pass as run_agent's input_stream  (agent -> client)
    bridge_writer: asyncio.StreamWriter # the pump-side socketpair half; caller must close it on teardown to release the FD
    ws_to_acp_task: asyncio.Task        # pumps WS frames into ACP's reader
    acp_to_ws_task: asyncio.Task        # pumps ACP's writes out as WS frames


async def ws_to_asyncio_streams(ws: web.WebSocketResponse) -> AdapterHandles:
    acp_sock, bridge_sock = socket.socketpair()
    acp_sock.setblocking(False)
    bridge_sock.setblocking(False)

    acp_reader, acp_writer = await asyncio.open_connection(sock=acp_sock, limit=_STREAM_LIMIT)
    bridge_reader, bridge_writer = await asyncio.open_connection(sock=bridge_sock, limit=_STREAM_LIMIT)

    async def _pump_ws_to_acp() -> None:
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    data = msg.data.encode("utf-8")
                elif msg.type == WSMsgType.BINARY:
                    data = bytes(msg.data)
                elif msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSING, WSMsgType.CLOSED, WSMsgType.ERROR):
                    break
                else:
                    continue
                if not data.endswith(b"\n"):
                    data += b"\n"
                bridge_writer.write(data)
                await bridge_writer.drain()
        except Exception:
            logger.exception("ws->acp pump failed")
        finally:
            # Signal EOF on ACP's reader.
            try:
                if bridge_writer.can_write_eof():
                    bridge_writer.write_eof()
            except Exception:
                pass

    async def _pump_acp_to_ws() -> None:
        try:
            while True:
                line = await bridge_reader.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").rstrip("\n")
                if not text:
                    continue
                if ws.closed:
                    break
                await ws.send_str(text)
        except Exception:
            logger.exception("acp->ws pump failed")

    ws_to_acp_task = asyncio.create_task(_pump_ws_to_acp(), name="ws->acp")
    acp_to_ws_task = asyncio.create_task(_pump_acp_to_ws(), name="acp->ws")

    return AdapterHandles(
        reader=acp_reader,
        writer=acp_writer,
        bridge_writer=bridge_writer,
        ws_to_acp_task=ws_to_acp_task,
        acp_to_ws_task=acp_to_ws_task,
    )
