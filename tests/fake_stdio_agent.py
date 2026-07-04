"""Fake stdio-ACP child for exec-route tests.

Modes via argv[1]:
  echo      -- for each stdin line, write back {"echo": <line>} as one line
  env       -- print {"env": os.environ.get("PROBE_VAR", "")} once, then echo
  exit-now  -- exit 0 immediately
  stderr    -- write one line to stderr, then echo
  close-stderr -- close stderr, then echo
  large     -- write one stdout line larger than asyncio's default limit
  hang      -- run until terminated, never writing stdout
"""
from __future__ import annotations

import json
import os
import sys
import time


def _echo() -> None:
    for line in sys.stdin:
        print(json.dumps({"echo": line.rstrip("\n")}), flush=True)


def main() -> int:
    mode = sys.argv[1]
    pid_file = os.environ.get("PROBE_PID_FILE")
    if pid_file:
        with open(pid_file, "w", encoding="utf-8") as f:
            f.write(str(os.getpid()))

    if mode == "echo":
        _echo()
    elif mode == "env":
        print(json.dumps({"env": os.environ.get("PROBE_VAR", "")}), flush=True)
        _echo()
    elif mode == "exit-now":
        return 0
    elif mode == "stderr":
        print("stderr probe", file=sys.stderr, flush=True)
        _echo()
    elif mode == "close-stderr":
        sys.stderr.close()
        os.close(2)
        _echo()
    elif mode == "large":
        for line in sys.stdin:
            print(json.dumps({"echo": "x" * 70_000}), flush=True)
    elif mode == "hang":
        while True:
            time.sleep(60)
    else:
        raise SystemExit(f"unknown mode: {mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
