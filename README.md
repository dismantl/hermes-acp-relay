# hermes-acp-relay

Path 3 from `Hermes ACP Over Network Analysis` — a WebSocket relay that exposes
[`hermes-agent`](../hermes-agent)'s ACP adapter over a network, without
modifying upstream Hermes.

Two packages in one repo because they run on different machines:

- **`hermes-acp-relay`** (remote, e.g. home-lab box) — aiohttp WebSocket server
  wrapping `HermesACPAgent`. One instance per WS connection. Binds to a
  non-public interface.
- **`hermes-acp-bridge`** (laptop) — stdio ↔ WebSocket pump. Obsidian Agent
  Client (OAC) and other stdio-only ACP clients spawn it as their "agent
  command"; it proxies ACP JSON-RPC to the remote relay.

Auth and TLS are the reverse proxy's job. This repo assumes a proxy
(Pangolin, Caddy, Traefik, …) terminates TLS and enforces HTTP Basic auth or
similar on the WS upgrade request. The relay itself has zero auth code.

## Install

Requires Python ≥ 3.11 and a local checkout of `hermes-agent` at
`../hermes-agent` (relative to this repo).

```sh
# in ~/code/hermes-acp-relay
uv sync
```

This installs both workspace members and the local editable `hermes-agent`
(with the `[acp]` extra).

## Run — remote side (home-lab)

```sh
uv run hermes-acp-relay --host 127.0.0.1 --port 8765
```

Binding notes:

- `127.0.0.1` is the safe default. For a reverse-proxy deployment, bind to the
  private-network IP the proxy reaches you on (e.g. `10.0.0.42`), **not** a
  public interface.
- The relay has **no authentication**. Never expose it directly to the public
  internet — the proxy is the only gatekeeper.

### Reverse-proxy config (Pangolin example)

- **Target:** `http://<relay-host>:8765/acp`
- **Auth:** HTTP Basic. Generate a password: `openssl rand -base64 32`.
- **WebSocket:** must be enabled for this resource (standard HTTP/1.1
  `Upgrade: websocket`).
- **Health check:** `GET /health` returns `{"status":"ok"}`.

The same shape works for Caddy, Traefik, nginx; only the config syntax
differs.

## Run — laptop side

```sh
uv run hermes-acp-bridge
```

Reads `~/.config/hermes-acp-bridge/config.toml` (override with `-c PATH` or
`HERMES_BRIDGE_CONFIG`). Example:

```toml
url = "wss://hermes.mydomain.example/acp"
username = "hermes-acp"
password = "<paste the password you set in the proxy>"
```

Make it private: `chmod 600 ~/.config/hermes-acp-bridge/config.toml`.

### Obsidian Agent Client setup

Add a custom agent in OAC:

- **Command:** `hermes-acp-bridge` (or absolute path, e.g. `/home/you/.local/bin/hermes-acp-bridge`)
- **Args:** (none)

No credentials go into OAC's config; they're all in the bridge's config
file outside Obsidian's vault.

## Architecture

```
[OAC in Obsidian on laptop]
        | stdio (ACP JSON-RPC, line-delimited)
        v
[hermes-acp-bridge]                 ← this repo, laptop-side
        | wss:// (HTTP Basic auth, TLS terminated by proxy)
        v
[Reverse proxy]                     ← user's existing infra
        | ws:// (plain HTTP on the internal network)
        v
[hermes-acp-relay]                  ← this repo, remote-side
        |
        v
[HermesACPAgent]                    ← upstream hermes-agent, unmodified
```

See `../../Personal/1 Projects/Hermes Remote ACP/Path 3 Implementation Plan.md`
for the full design notes (concurrency model, session-resume behaviour,
known limitations).

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
