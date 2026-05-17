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
                patch_state = self._install_stream_patch(args, kwargs)
                try:
                    return await super().prompt(*args, **kwargs)
                finally:
                    if patch_state is not None:
                        self._uninstall_stream_patch(*patch_state)

        # --- streaming patch ---------------------------------------------------
        #
        # Upstream's prompt() runs the agent non-streaming and emits ONE
        # agent_message_chunk holding the full final_response after the turn
        # ends. For voice (TTS), that means audio can't start until the LLM
        # finishes — blowing the 2-4s latency budget. AIAgent.run_conversation
        # actually accepts a stream_callback that fires per token (run_agent.py
        # ~10141), but upstream's acp_adapter.server.prompt() never passes one.
        #
        # We close the gap by, for the duration of one prompt:
        #   1. monkey-patch agent.run_conversation to inject a stream_callback
        #      that forwards deltas to the message_callback upstream's prompt()
        #      already wired at acp_adapter/server.py:637 (which marshals from
        #      the executor thread to the loop via run_coroutine_threadsafe).
        #   2. monkey-patch self._conn.session_update so the tail-end emit at
        #      acp_adapter/server.py:739-741 is dropped — exactly one
        #      agent_message_chunk for our session_id, fired after the
        #      run_conversation that produced any stream output. Other update
        #      types (tool progress, thinking, history replay) pass through,
        #      and auto-titling (lines 726-738) still sees final_response.
        def _install_stream_patch(self, args, kwargs):
            session_id = kwargs.get("session_id")
            if session_id is None and len(args) >= 2:
                session_id = args[1]
            if session_id is None or self._conn is None:
                return None

            try:
                state = self.session_manager.get_session(session_id)
            except Exception:
                logger.debug("stream patch: session lookup failed", exc_info=True)
                return None
            if state is None:
                return None

            agent = getattr(state, "agent", None)
            if agent is None or not hasattr(agent, "run_conversation"):
                return None

            conn = self._conn
            had_run_attr = "run_conversation" in agent.__dict__
            original_run = agent.run_conversation
            had_send_attr = "session_update" in conn.__dict__
            original_send = conn.session_update

            # Per-prompt state shared between the run wrapper (executor thread)
            # and the session_update wrapper (event loop). "armed" stays False
            # while streaming chunks flow so message_cb's emissions pass through;
            # it flips True only AFTER run_conversation returns, so the very
            # next agent_message_chunk for our session_id — the tail-end emit
            # at acp_adapter/server.py:739-741 — is dropped exactly once.
            arm = {"armed": False, "consumed": False, "caller_owned_stream": False}

            def patched_run(*ra, **rkw):
                caller_provided = rkw.get("stream_callback") is not None
                if caller_provided:
                    # Caller drove its own streaming; don't interfere with
                    # delivery and don't suppress the tail emit.
                    arm["caller_owned_stream"] = True
                    return original_run(*ra, **rkw)

                message_cb = getattr(agent, "message_callback", None)
                if message_cb is None:
                    return original_run(*ra, **rkw)

                streamed = [False]

                def wrapped_cb(text, _cb=message_cb, _flag=streamed):
                    if isinstance(text, str) and text:
                        _flag[0] = True
                        logger.debug("stream delta: %d chars", len(text))
                    try:
                        _cb(text)
                    except Exception:
                        logger.debug("stream callback raised", exc_info=True)

                rkw["stream_callback"] = wrapped_cb
                result = original_run(*ra, **rkw)
                if streamed[0]:
                    arm["armed"] = True
                return result

            async def patched_session_update(s_id, update, _orig=original_send,
                                              _arm=arm, _our_sid=session_id):
                if (
                    s_id == _our_sid
                    and _arm["armed"]
                    and not _arm["consumed"]
                    and not _arm["caller_owned_stream"]
                    and getattr(update, "session_update", None) == "agent_message_chunk"
                ):
                    _arm["consumed"] = True
                    return
                return await _orig(s_id, update)

            agent.run_conversation = patched_run
            conn.session_update = patched_session_update
            return (agent, original_run, had_run_attr,
                    conn, original_send, had_send_attr)

        @staticmethod
        def _uninstall_stream_patch(agent, original_run, had_run_attr,
                                    conn, original_send, had_send_attr):
            if had_run_attr:
                agent.run_conversation = original_run
            else:
                try:
                    del agent.run_conversation
                except AttributeError:
                    pass
            if had_send_attr:
                conn.session_update = original_send
            else:
                try:
                    del conn.session_update
                except AttributeError:
                    pass

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
