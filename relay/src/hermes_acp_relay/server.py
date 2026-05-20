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
                logger.warning(
                    "stream patch: agent missing run_conversation on session %s; "
                    "falling back to non-streaming",
                    session_id,
                )
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
                    logger.warning(
                        "stream patch: agent.message_callback unset on session %s; "
                        "falling back to non-streaming",
                        session_id,
                    )
                    return original_run(*ra, **rkw)

                streamed = [False]

                def wrapped_cb(text, _cb=message_cb, _flag=streamed):
                    if isinstance(text, str) and text:
                        _flag[0] = True
                        logger.debug("stream delta: %d chars", len(text))
                    # _cb here is make_message_cb's _message, which calls
                    # events._send_update — that already swallows its own
                    # exceptions, so we don't wrap.
                    _cb(text)

                rkw["stream_callback"] = wrapped_cb
                result = original_run(*ra, **rkw)
                if streamed[0]:
                    arm["armed"] = True
                return result

            # Match upstream's session_update signature exactly. Two call
            # shapes exist in acp_adapter/server.py: positional (:605, :741)
            # and keyword (:445 _replay_session_history, :780-786
            # _send_available_commands_update). Both can land here while the
            # patch is installed because non-prompt ACP methods run
            # concurrently with prompts (only prompts hold _prompt_lock).
            #
            # Drop strategy is by ordinal, not by content: the first
            # agent_message_chunk for our session_id seen AFTER the executor
            # returns is upstream's tail emit at acp_adapter/server.py:739-741.
            # The race window for misfiring is narrow — only a same-session
            # session/load replay (_replay_session_history at
            # acp_adapter/server.py:422-449) that fires between
            # arm["armed"] = True and the tail emit could trip us. Content
            # matching would be more robust but fragile to upstream
            # post-processing (think-block stripping, scrubber tail flush,
            # etc.).
            async def patched_session_update(
                session_id,
                update,
                _orig=original_send,
                _arm=arm,
                _our_sid=session_id,
            ):
                if (
                    session_id == _our_sid
                    and _arm["armed"]
                    and not _arm["consumed"]
                    and not _arm["caller_owned_stream"]
                    and getattr(update, "session_update", None) == "agent_message_chunk"
                ):
                    _arm["consumed"] = True
                    return
                return await _orig(session_id, update)

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
            # WS pumps exited first — client disconnected (or the heartbeat
            # timed out). Cancel the in-flight runner IMMEDIATELY so any held
            # `_prompt_lock` is released without delay; the previous 2-second
            # EOF-unwind grace let slow Honcho calls keep the lock held past
            # this WS's lifetime, queueing subsequent voice/text connections
            # on the next prompt() attempt. The client is already gone — there
            # is no partial response we need to flush — so cancelling fast is
            # strictly safe. See: dismantl/acab-ansible#567 for the originally
            # observed failure mode (relay wedged across sessions until manual
            # container restart).
            runner_task.cancel()
            try:
                await runner_task
            except asyncio.CancelledError:
                # Propagated CancelledError from runner_task.cancel() above —
                # consume; we initiated this and the WS handler should still
                # run its finally cleanup. If _handle_acp itself is being
                # externally cancelled (aiohttp shutdown), the cancellation
                # re-fires at the next await in the finally block; the outer
                # `except asyncio.CancelledError` handler catches it there.
                pass
            except Exception:
                # run_agent's own error during cleanup. Exceptions are still
                # surfaced via the branch below using runner_task.exception().
                # Narrower than `BaseException` so KeyboardInterrupt and
                # SystemExit propagate as a process exit signal rather than
                # being silently swallowed mid-cleanup.
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
