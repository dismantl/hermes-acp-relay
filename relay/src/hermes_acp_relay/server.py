"""aiohttp WebSocket server exposing HermesACPAgent.

One SerializedHermesACPAgent instance per WS connection — matches the
per-process semantics of `hermes acp` today, without the process spawn.
"""
from __future__ import annotations

import asyncio
import contextvars
import logging
from pathlib import Path
from typing import Callable, Optional

from aiohttp import web

from .ws_streams import ws_to_asyncio_streams

logger = logging.getLogger(__name__)

_MAX_MSG_SIZE = 50 * 1024 * 1024  # matches acp SDK's stdio buffer default

# ---------------------------------------------------------------------------
# Per-WS profile override
#
# The relay serves multiple ACP consumers (Obsidian Agent Client, hermes-
# acp-bridge, the V2 voice agent) from the same process. Each consumer may
# want different Hermes-side behavior — most notably, the voice agent wants
# a different Honcho memory policy than the text channels (see acab-ansible
# voice plan + Honcho policy layer redesign).
#
# Approach: a ContextVar holds an optional profile-home override per
# asyncio.Task; the /acp/<profile-suffix> URL route sets it before the
# agent is instantiated; a shim wrapped around hermes_constants.get_hermes_home
# returns the override (when set) or the default. Hermes runtime resolves
# get_hermes_home() per-call in the profile-state code paths the redesign
# cares about (memory manager, honcho.json reader, skill discovery, config,
# plugins, AIAgent), so the override flows through.
#
# Caveat: ContextVars are NOT automatically propagated to executor threads
# (loop.run_in_executor). If Hermes' upstream prompt() reads get_hermes_home()
# inside an executor, that read sees the default profile, not the override.
# The Phase 0 audit (see acab-ansible voice plan) found no executor-thread
# reads in the ACP path that would matter for the Honcho-policy use case,
# but this is the known limitation. Audit follow-up if a profile-specific
# leak is ever observed.
_HERMES_HOME_OVERRIDE: contextvars.ContextVar[Optional[Path]] = contextvars.ContextVar(
    "hermes_home_override", default=None
)

# Captured at install_profile_override_shim() time so the route handler can
# compute sub-profile paths from the default base without recursing through
# the patched function.
_ORIGINAL_GET_HERMES_HOME: Optional[Callable[[], Path]] = None

# Suffix shape allowed by /acp/<suffix>: lowercase letter start, then
# lowercase letters / digits / hyphens. Excludes anything that could be
# interpreted as a path component (`.`, `/`, `..`). Combined with the
# resolve() + relative_to() check in the handler, this prevents
# URL-encoded traversal and symlink escapes.
_PROFILE_SUFFIX_PATTERN = r"[a-z][a-z0-9-]*"


def install_profile_override_shim() -> None:
    """Patch hermes_constants.get_hermes_home to consult _HERMES_HOME_OVERRIDE.

    Must be called AFTER the Hermes environment is bootstrapped (env loaded,
    HERMES_HOME resolved) and BEFORE any module that does
    `from hermes_constants import get_hermes_home` is imported. Calling this
    in __main__.main(), immediately after _load_hermes_env() and before
    `from .server import create_app`, satisfies both constraints — at that
    point hermes_constants has been imported (by _load_hermes_env) but no
    downstream hermes module has rebound the name.

    Idempotent: calling twice is a no-op.
    """
    import hermes_constants  # type: ignore[import-not-found]

    global _ORIGINAL_GET_HERMES_HOME
    if _ORIGINAL_GET_HERMES_HOME is not None:
        # Already installed.
        return
    _ORIGINAL_GET_HERMES_HOME = hermes_constants.get_hermes_home

    def get_hermes_home_with_override() -> Path:
        override = _HERMES_HOME_OVERRIDE.get()
        if override is not None:
            return override
        return _ORIGINAL_GET_HERMES_HOME()  # type: ignore[misc]

    hermes_constants.get_hermes_home = get_hermes_home_with_override

# Process-wide lock serializing HermesACPAgent.prompt() calls across all
# connections. Hermes's prompt() path mutates a process-global approval-callback
# hook around run_in_executor; concurrent prompts in one process race on that
# global and can misroute approval dialogs between editors. Non-prompt methods
# (initialize, new_session, cancel, …) still run concurrently.
#
# Liveness tradeoff: the lock is held for the full duration of one prompt.
# Two safety nets keep a wedged holder from blocking subsequent callers
# indefinitely:
#
#   1. _handle_acp cancels its runner_task immediately when WS pumps exit
#      (i.e., the client disconnected). The cancellation propagates through
#      `async with _prompt_lock:` to the lock's __aexit__, releasing it.
#      This is the primary recovery path.
#
#   2. Lock acquisition itself is bounded by _PROMPT_LOCK_TIMEOUT_S. If a
#      previous prompt is truly stuck and (1) failed to release the lock,
#      a queued caller times out after the bound and returns a refusal
#      PromptResponse rather than waiting forever. Belt-and-suspenders for
#      the (1) path; should not fire in normal operation.
#
# The timeout is a wall-clock upper bound on QUEUE WAIT, not on a single
# prompt's total duration. A prompt that legitimately takes 90s to complete
# is fine — only the NEXT caller's wait for the lock is bounded. Tune via
# _PROMPT_LOCK_TIMEOUT_S if voice or text channels surface cases where the
# legitimate queue wait exceeds the default.
_prompt_lock = asyncio.Lock()
_PROMPT_LOCK_TIMEOUT_S = 60.0


def _build_serialized_agent_class():
    """Import HermesACPAgent lazily — must happen after Hermes env bootstrap."""
    from acp_adapter.server import HermesACPAgent

    class SerializedHermesACPAgent(HermesACPAgent):
        async def prompt(self, *args, **kwargs):
            # Acquire the process-wide prompt lock with a wall-clock bound.
            # Under normal operation _prompt_lock is uncontested or briefly
            # held; the timeout only fires if a previous prompt is wedged
            # (e.g., a Honcho dialectic call hung and the cancellation path
            # in _handle_acp failed to release the lock). Returns a refusal
            # PromptResponse rather than queueing the caller forever.
            session_id = kwargs.get("session_id")
            if session_id is None and len(args) >= 2:
                session_id = args[1]
            session_id = session_id or "<unknown>"
            try:
                await asyncio.wait_for(
                    _prompt_lock.acquire(),
                    timeout=_PROMPT_LOCK_TIMEOUT_S,
                )
            except asyncio.TimeoutError:
                logger.error(
                    "prompt_lock acquisition timed out after %.1fs on session "
                    "%s; an earlier prompt is wedged. Returning refusal so the "
                    "client can recover. This is the belt-and-suspenders safety "
                    "net for the issue 567 cascade — its firing indicates the "
                    "primary recovery path in _handle_acp did not release the "
                    "lock as expected.",
                    _PROMPT_LOCK_TIMEOUT_S,
                    session_id,
                )
                # Lazy import — only reached on the rare timeout path; acp is
                # imported lazily elsewhere in this module per post-bootstrap
                # convention.
                from acp.schema import PromptResponse
                return PromptResponse(stop_reason="refusal")
            try:
                patch_state = self._install_stream_patch(args, kwargs)
                try:
                    return await super().prompt(*args, **kwargs)
                finally:
                    if patch_state is not None:
                        self._uninstall_stream_patch(*patch_state)
            finally:
                _prompt_lock.release()

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


async def _handle_acp_with_profile_override(request: web.Request) -> web.WebSocketResponse:
    """WS handler for /acp/<profile-suffix> — loads a Hermes sub-profile.

    The sub-profile must exist at ${HERMES_HOME}/profiles/<suffix>/ (a
    `gateway: false` profile dir as provisioned by acab-ansible's hermes
    role). Returns 404 if the directory is missing. Sets the per-WS
    _HERMES_HOME_OVERRIDE ContextVar for the lifetime of this WS connection
    and delegates to _handle_acp; the shim around hermes_constants.get_hermes_home
    routes profile-state reads to the override path.
    """
    suffix = request.match_info["profile_suffix"]

    if _ORIGINAL_GET_HERMES_HOME is None:
        # install_profile_override_shim() should have been called by main()
        # before create_app() ran. Defensive log + 500 if we somehow ended
        # up here without it.
        logger.error(
            "profile override requested but shim is not installed; "
            "call install_profile_override_shim() before create_app()"
        )
        raise web.HTTPInternalServerError(reason="profile shim not installed")

    base = Path(_ORIGINAL_GET_HERMES_HOME())
    profiles_root = (base / "profiles").resolve()
    candidate = (profiles_root / suffix).resolve()

    # Defense in depth: the candidate's resolved path must live under
    # profiles_root. The URL regex already restricts the suffix shape;
    # this catches symlink-based escape attempts where the regex-valid
    # directory name points outside the tree.
    try:
        candidate.relative_to(profiles_root)
    except ValueError:
        logger.warning(
            "rejecting profile suffix that resolves outside profiles_root: "
            "%s (resolved to %s, profiles_root=%s)",
            suffix, candidate, profiles_root,
        )
        raise web.HTTPNotFound(reason=f"unknown profile: {suffix}")

    if not candidate.is_dir():
        raise web.HTTPNotFound(reason=f"unknown profile: {suffix}")

    token = _HERMES_HOME_OVERRIDE.set(candidate)
    try:
        logger.info(
            "loading sub-profile %s (home=%s) for client %s",
            suffix, candidate, request.remote,
        )
        return await _handle_acp(request)
    finally:
        _HERMES_HOME_OVERRIDE.reset(token)


async def _handle_health(request: web.Request) -> web.Response:
    return web.json_response({"status": "ok"})


def create_app() -> web.Application:
    """Construct the aiohttp application. Caller must have bootstrapped Hermes
    and called install_profile_override_shim() first.
    """
    app = web.Application()
    app["agent_cls"] = _build_serialized_agent_class()
    app.router.add_get("/health", _handle_health)
    app.router.add_get("/acp", _handle_acp)
    # Per-WS profile-switching route. URL suffix maps to a sub-profile
    # directory under ${HERMES_HOME}/profiles/. See the _HERMES_HOME_OVERRIDE
    # docstring at module top for the design.
    app.router.add_get(
        "/acp/{profile_suffix:" + _PROFILE_SUFFIX_PATTERN + "}",
        _handle_acp_with_profile_override,
    )
    return app
