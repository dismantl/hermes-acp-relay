"""aiohttp WebSocket server exposing HermesACPAgent.

One SerializedHermesACPAgent instance per WS connection — matches the
per-process semantics of `hermes acp` today, without the process spawn.
"""
from __future__ import annotations

import asyncio
import logging

from aiohttp import web

from .ws_streams import ws_to_asyncio_streams

logger = logging.getLogger(__name__)

_MAX_MSG_SIZE = 50 * 1024 * 1024  # matches acp SDK's stdio buffer default

# Process-wide lock serializing HermesACPAgent.prompt() calls across all
# connections. See the implementation plan's "Known limitations #1": upstream's
# tools/terminal_tool._approval_callback is a module-global that prompt() swaps
# around run_in_executor. Two concurrent prompts in one process would race on
# that global and could misroute approval dialogs between editors. Non-prompt
# methods (initialize, new_session, cancel, …) still run concurrently.
_prompt_lock = asyncio.Lock()


def _build_serialized_agent_class():
    """Import HermesACPAgent lazily — must happen after Hermes env bootstrap."""
    from acp_adapter.server import HermesACPAgent

    class SerializedHermesACPAgent(HermesACPAgent):
        async def prompt(self, *args, **kwargs):
            async with _prompt_lock:
                return await super().prompt(*args, **kwargs)

    return SerializedHermesACPAgent


async def _handle_acp(request: web.Request) -> web.WebSocketResponse:
    import acp  # lazy, post-bootstrap

    agent_cls = request.app["agent_cls"]

    ws = web.WebSocketResponse(max_msg_size=_MAX_MSG_SIZE, heartbeat=30.0)
    await ws.prepare(request)
    peer = request.remote
    logger.info("client connected: %s", peer)

    handles = await ws_to_asyncio_streams(ws)
    agent = agent_cls()
    try:
        await acp.run_agent(
            agent,
            input_stream=handles.writer,
            output_stream=handles.reader,
            use_unstable_protocol=True,
        )
    except (asyncio.CancelledError, ConnectionResetError):
        pass
    except Exception:
        logger.exception("ACP handler crashed for %s", peer)
    finally:
        # Close the ACP writer so the acp->ws pump sees EOF and exits cleanly.
        try:
            handles.writer.close()
            await handles.writer.wait_closed()
        except Exception:
            pass
        # Give pump tasks a moment to drain before we tear down the WS.
        try:
            await asyncio.wait_for(
                asyncio.gather(
                    handles.ws_to_acp_task,
                    handles.acp_to_ws_task,
                    return_exceptions=True,
                ),
                timeout=2.0,
            )
        except asyncio.TimeoutError:
            for t in (handles.ws_to_acp_task, handles.acp_to_ws_task):
                if not t.done():
                    t.cancel()
        if not ws.closed:
            await ws.close()
        logger.info("client disconnected: %s", peer)

    return ws


async def _handle_health(request: web.Request) -> web.Response:
    return web.json_response({"status": "ok"})


def create_app() -> web.Application:
    """Construct the aiohttp application. Caller must have bootstrapped Hermes first."""
    app = web.Application()
    app["agent_cls"] = _build_serialized_agent_class()
    app.router.add_get("/health", _handle_health)
    app.router.add_get("/acp", _handle_acp)
    return app
