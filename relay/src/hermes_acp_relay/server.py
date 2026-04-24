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
# connections. Hermes's prompt() path mutates a process-global approval-callback
# hook around run_in_executor; concurrent prompts in one process race on that
# global and can misroute approval dialogs between editors. Non-prompt methods
# (initialize, new_session, cancel, …) still run concurrently.
#
# Liveness tradeoff: the lock is held for the full duration of one prompt. There
# is no numeric timeout because legitimate agent work (tool loops, long model
# reasoning) can exceed any sane bound. In single-user mode the recovery path
# is: close the client; the WS drops, _handle_acp's finally cancels the lock-
# holding task, lock releases.
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
    runner_task = asyncio.create_task(
        acp.run_agent(
            agent,
            input_stream=handles.writer,
            output_stream=handles.reader,
            use_unstable_protocol=True,
        ),
        name="acp-runner",
    )
    try:
        # If either pump exits (cleanly or by raising) before run_agent does,
        # run_agent would otherwise block forever waiting on dead streams.
        done, _ = await asyncio.wait(
            {runner_task, handles.ws_to_acp_task, handles.acp_to_ws_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if runner_task not in done:
            try:
                handles.writer.close()  # EOF run_agent's input so it unwinds
            except Exception:
                pass
            try:
                await asyncio.wait_for(runner_task, timeout=2.0)
            except asyncio.TimeoutError:
                runner_task.cancel()
                await asyncio.gather(runner_task, return_exceptions=True)
            except Exception:
                # run_agent raised during EOF/cleanup; the exception is captured
                # on runner_task and logged by the branch below. Don't let it
                # escape here or aiohttp will mark the handler as failed.
                pass
        if runner_task.done() and not runner_task.cancelled():
            exc = runner_task.exception()
            if isinstance(exc, ConnectionResetError):
                logger.info("connection reset by %s", peer)
            elif exc is not None and not isinstance(exc, asyncio.CancelledError):
                logger.error("ACP handler crashed for %s", peer, exc_info=exc)
    except asyncio.CancelledError:
        runner_task.cancel()
        await asyncio.gather(runner_task, return_exceptions=True)
        raise
    finally:
        # Close the ACP writer so the acp->ws pump sees EOF and exits cleanly.
        try:
            handles.writer.close()
            await handles.writer.wait_closed()
        except Exception as e:
            logger.warning("error closing ACP writer for %s: %r", peer, e)
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
        # Release the bridge side of the socketpair; without this the FD lingers until GC.
        try:
            handles.bridge_writer.close()
            await handles.bridge_writer.wait_closed()
        except Exception as e:
            logger.warning("error closing bridge writer for %s: %r", peer, e)
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
