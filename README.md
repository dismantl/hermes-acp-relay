# hermes-acp-relay

WebSocket relay and stdio bridge for exposing `hermes-agent`'s ACP adapter over
a private network, without modifying upstream Hermes.

Two packages in one repo because they run on different machines:

- **`hermes-acp-relay`** (server) — aiohttp WebSocket server wrapping
  `HermesACPAgent`. One instance per WS connection. Binds to a
  non-public interface.
- **`hermes-acp-bridge`** (client) — stdio ↔ WebSocket pump. Obsidian Agent
  Client (OAC) and other stdio-only ACP clients spawn it as their "agent
  command"; it proxies ACP JSON-RPC to the relay server.

Auth and TLS are the reverse proxy's job. This repo assumes a proxy
(Caddy, Traefik, nginx, …) terminates TLS and enforces HTTP Basic auth or
similar on the WS upgrade request. The relay itself has zero auth code.

## Install

Requires Python ≥ 3.11. The workspace fetches `hermes-agent` from its public
GitHub repository.

```sh
# from this repo checkout
uv sync --all-packages
```

This installs both workspace members and `hermes-agent` with the `[acp]` extra.

## Run — server side

```sh
uv run hermes-acp-relay --host 127.0.0.1 --port 8765
```

Binding notes:

- `127.0.0.1` is the safe default. For a reverse-proxy deployment, bind to the
  private-network IP the proxy reaches you on (e.g. `10.0.0.42`), **not** a
  public interface.
- The relay has **no authentication**. Never expose it directly to the public
  internet — the proxy is the only gatekeeper.

### Reverse-proxy config

- **Target:** `http://<relay-host>:8765/acp`
- **Sub-profile target (optional):** `http://<relay-host>:8765/acp/<suffix>` —
  see "Sub-profile routing" below.
- **Auth:** HTTP Basic. Generate a password: `openssl rand -base64 32`.
- **WebSocket:** must be enabled for this resource (standard HTTP/1.1
  `Upgrade: websocket`).
- **Health check:** `GET /health` returns `{"status":"ok"}`.

The same shape works for Caddy, Traefik, nginx; only the config syntax
differs.

### Sub-profile routing

In addition to the default `/acp` endpoint, the relay accepts
`/acp/<profile-suffix>` connections that load a Hermes sub-profile for the
lifetime of that WS connection. Use cases:

- Voice clients connect to `/acp/voice` to get a profile with a
  low-latency Honcho memory policy while default clients keep
  using `/acp` with the default profile.
- Operator-triggered deep-memory sessions connect to `/acp/deep` for
  expensive reflective dialectic.

**Layout convention:** the suffix maps to `${HERMES_HOME}/profiles/<suffix>/`.
Provision sub-profiles as standalone Hermes homes, typically with
`gateway: false` and `home: "${HERMES_HOME}/profiles/<suffix>"`. The relay
returns 404 if the directory is missing.

**Allowed suffixes:** lowercase letter start, then lowercase letters,
digits, hyphens. URL-encoded traversal (`%2e%2e`) and symlinks that escape
`${HERMES_HOME}/profiles/` are rejected at the handler.

**Shared state:** sub-profiles inherit the relay process's loaded `.env`
and `auth.json`, so all profiles share provider API keys and Anthropic
OAuth. Honcho `workspace` / `peerName` are typically the same across
profiles too (set in each sub-profile's `honcho.json`), so memory is
unified across consumers.

## Run — client side

```sh
uv run hermes-acp-bridge
```

Reads `~/.config/hermes-acp-bridge/config.toml` (override with `-c PATH` or
`HERMES_BRIDGE_CONFIG`). Example:

```toml
url = "wss://relay.example.com/acp"
username = "hermes-acp"
password = "<paste the password you set in the proxy>"
```

Make it private: `chmod 600 ~/.config/hermes-acp-bridge/config.toml`.

### Obsidian Agent Client setup

Add a custom agent in OAC:

- **Command:** `hermes-acp-bridge` (or an absolute path to the executable)
- **Args:** (none)

No credentials go into OAC's config; they're all in the bridge's local config
file.

## Architecture

```
[OAC on client]
        | stdio (ACP JSON-RPC, line-delimited)
        v
[hermes-acp-bridge]                 ← this repo, client-side
        | wss:// (HTTP Basic auth, TLS terminated by proxy)
        v
[Reverse proxy]                     ← deployment environment
        | ws:// (plain HTTP on the internal network)
        v
[hermes-acp-relay]                  ← this repo, server-side
        |
        v
[HermesACPAgent]                    ← upstream hermes-agent, unmodified
```

## Known limitations

- **Prompts are serialized across the whole relay process.** Two editors can
  both be connected and interact with non-prompt ACP methods freely, but only
  one prompt runs at a time. This is intentional — it mitigates an upstream
  process-global-race in `tools/terminal_tool._approval_callback` without
  requiring a hermes-agent patch.
- **Not safe for multi-user.** The shared `~/.hermes/state.db` means anyone
  authenticated can load any session by ID. Single-user self-to-self only.
- **Bridge doesn't auto-reconnect.** If the network drops, OAC sees EOF; the
  user clicks Restore in OAC to reopen. Session history is persistent, so the
  conversation resumes in place.
- **Sub-profile override doesn't reach executor threads.** The per-WS
  profile override (`/acp/<suffix>` routing) uses `contextvars.ContextVar`
  to scope the active profile to one asyncio.Task. asyncio child tasks
  inherit the parent's context; `loop.run_in_executor` threads DO NOT.
  Hermes' upstream `prompt()` runs the LLM call in an executor thread; if
  any code in that thread reads `hermes_constants.get_hermes_home()` it
  sees the default profile, not the override. A local audit of upstream
  Hermes found no executor-thread reads in the ACP path that affect
  Honcho-policy behavior, but a future regression could surface this.
  Audit follow-up if profile-specific state leaks between channels.
