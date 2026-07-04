"""Per-connection subprocess sessions for command-backed ACP routes."""
from __future__ import annotations

import asyncio
import logging
import os

from aiohttp import WSMsgType, web

from .exec_routes import ExecRoute


logger = logging.getLogger(__name__)

_TERMINATE_GRACE_S = 5.0


async def run_exec_session(ws: web.WebSocketResponse, route: ExecRoute) -> None:
    """Spawn route.command and pipe WebSocket frames to the child stdio."""
    try:
        child = await asyncio.create_subprocess_exec(
            *route.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, **route.env},
            cwd=route.cwd,
        )
    except OSError:
        logger.error("exec route %s failed to spawn", route.name, exc_info=True)
        if not ws.closed:
            await ws.close()
        return

    assert child.stdin is not None
    assert child.stdout is not None
    assert child.stderr is not None

    async def pump_ws_to_child() -> None:
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    data = msg.data.encode("utf-8")
                elif msg.type == WSMsgType.BINARY:
                    data = bytes(msg.data)
                elif msg.type in (
                    WSMsgType.CLOSE,
                    WSMsgType.CLOSING,
                    WSMsgType.CLOSED,
                    WSMsgType.ERROR,
                ):
                    break
                else:
                    continue
                if not data.endswith(b"\n"):
                    data += b"\n"
                child.stdin.write(data)
                await child.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            logger.info("exec route %s stdin closed", route.name)
        except Exception:
            logger.exception("exec route %s ws->stdin pump failed", route.name)
        finally:
            try:
                if child.stdin.can_write_eof():
                    child.stdin.write_eof()
                else:
                    child.stdin.close()
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception:
                logger.exception("exec route %s failed to close stdin", route.name)

    async def pump_child_to_ws() -> None:
        try:
            while True:
                line = await child.stdout.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").rstrip("\n")
                if not text:
                    continue
                if ws.closed:
                    break
                await ws.send_str(text)
        except Exception:
            logger.exception("exec route %s stdout->ws pump failed", route.name)

    async def pump_stderr_to_log() -> None:
        try:
            while True:
                line = await child.stderr.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").rstrip("\n")
                if text:
                    logger.info("[%s] %s", route.name, text)
        except Exception:
            logger.exception("exec route %s stderr pump failed", route.name)

    pump_tasks = {
        asyncio.create_task(pump_ws_to_child(), name=f"exec-{route.name}-ws-to-stdin"),
        asyncio.create_task(pump_child_to_ws(), name=f"exec-{route.name}-stdout-to-ws"),
        asyncio.create_task(pump_stderr_to_log(), name=f"exec-{route.name}-stderr"),
    }
    wait_task = asyncio.create_task(child.wait(), name=f"exec-{route.name}-wait")

    try:
        done, _ = await asyncio.wait(
            pump_tasks | {wait_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if wait_task in done:
            returncode = wait_task.result()
            if returncode:
                logger.warning(
                    "exec route %s child exited with code %s",
                    route.name,
                    returncode,
                )
            else:
                logger.info("exec route %s child exited", route.name)
            if not ws.closed:
                await ws.close()
        else:
            await _terminate_child(child)
    except asyncio.CancelledError:
        await _terminate_child(child)
        raise
    finally:
        for task in pump_tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*pump_tasks, return_exceptions=True)
        if child.returncode is None:
            await _terminate_child(child)
        if not wait_task.done():
            wait_task.cancel()
        await asyncio.gather(wait_task, return_exceptions=True)
        if not ws.closed:
            await ws.close()


async def _terminate_child(child: asyncio.subprocess.Process) -> None:
    if child.returncode is not None:
        return
    try:
        child.terminate()
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(child.wait(), timeout=_TERMINATE_GRACE_S)
    except asyncio.TimeoutError:
        logger.warning("exec route child ignored terminate; killing")
        try:
            child.kill()
        except ProcessLookupError:
            return
        await child.wait()
