# total-recall

Persistent cross-session memory for Claude Code. Stores decisions, patterns, and lessons learned in a semantic search index (qmd) so every new session starts with full context of past work.

Works on a single machine with iCloud backup, or across multiple machines via a built-in HTTP server with a web UI for browsing and editing memories.

## Quick Start

### Claude Code Plugin (recommended)

```text
/install radu2lupu/total-recall
```

Then run `/memory-setup` inside any project to configure it.

### npm

```bash
npm install -g total-recall-memory
total-recall install --project my-project
```

### From source

```bash
git clone https://github.com/radu2lupu/total-recall.git
cd total-recall
./scripts/install.sh --project my-project
```

## What It Does

Once installed, memory is fully automatic:

- **Session start**: Claude queries past memories for relevant context before starting work
- **Session end**: Claude writes a summary of what was done, decisions made, and lessons learned
- **Mid-session**: Claude queries memory when encountering problems where prior context could help

No manual commands needed. Three enforcement layers make this reliable:

1. **SessionStart hook** — fires before Claude sees the user's message, forcing a memory query as the first action
2. **CLAUDE.md instructions** — detailed rules for when to query and when to write
3. **Stop hook** — blocks session end until a memory summary is written

## Architecture

```
~/.ai-memory/knowledge/<project>/
  ├── sessions/           # Session summaries (auto-generated)
  │   ├── 2025-06-15-auth-refactor.md
  │   └── imported/       # Ingested from Claude/Codex logs
  │       ├── claude/
  │       └── codex/
  ├── decisions/          # Architectural decisions
  ├── patterns/           # Reusable patterns
  ├── bugs/               # Bug investigations
  └── MEMORY.md           # Project-level notes
```

Each memory file contains structured metadata — date, machine, project, topic — and is indexed by qmd for hybrid BM25 + vector search.

## Multi-Machine Setup

Share memories across machines on the same network using the built-in HTTP server. One command per machine.

### On the server machine

```bash
./scripts/install.sh --project my-project --server
```

This does everything: local memory setup, iCloud backup, session ingestion, hooks, **and** starts an HTTP server on port 7899 with auto-restart on boot. It prints the API key and client connection command.

### On client machines

```bash
./scripts/install.sh --project my-project --client \
  --server-url http://server.tailscale:7899 \
  --api-key tr_sk_...
```

Client installs are lightweight — no local qmd or memory directories needed. Just the CLI, hooks, and instruction injection. All `write` and `query` operations route to the server automatically.

### Web UI

The server includes a web UI at `http://server:7899/` for browsing, viewing, editing, and deleting memories. Memories are grouped by project with metadata pills showing date, machine, and source tool.

### Manual server/client commands

For more control, the individual commands are still available:

```bash
total-recall server init              # Generate config + API key
total-recall server start             # Start server manually
total-recall server stop              # Stop server
total-recall server status            # Check server status
total-recall server add-key           # Generate additional API key
total-recall server install-launchd   # Auto-start on boot (macOS)

total-recall client configure --server-url URL --api-key KEY
total-recall client status            # Check connection
total-recall client enable            # Re-enable remote mode
total-recall client disable           # Switch back to local mode
```

## All Commands

```
# Install (three modes)
total-recall install --project NAME                    Standalone: full local setup
total-recall install --project NAME --server            Server: local + HTTP API + launchd
total-recall install --project NAME --client --server-url URL --api-key KEY   Client: remote only

# Core
total-recall write   --project NAME "<summary>"        Write a memory note
total-recall query   --project NAME "<query>"          Semantic search across memories
total-recall ingest  --project NAME                    Import Claude/Codex session logs
total-recall status  --project NAME                    Show memory stats and config

# iCloud
total-recall icloud-enable   --project NAME            Move memory to iCloud Drive
total-recall icloud-sync     --project NAME            Manual iCloud push/pull
total-recall icloud-status   --project NAME            Check iCloud sync state

# Server management
total-recall server init / start / stop / status / add-key / install-launchd

# Client management
total-recall client configure / status / enable / disable
```

### Plugin Slash Commands

When installed as a Claude Code plugin:

- `/memory-setup` — Interactive setup wizard for the current project
- `/memory-write` — Write a session summary to memory
- `/memory-rebuild` — Rebuild memory from git history and Codex sessions

## Install Options

```bash
# Standalone options
total-recall install --project my-project --no-icloud          # Skip iCloud backup
total-recall install --project my-project --no-launch-agent    # Skip background sync
total-recall install --project my-project --interval-minutes 10 # Sync every 10 min
total-recall install --project my-project --skip-embed         # Skip vector embeddings

# Server options
total-recall install --project my-project --server --port 8080 # Custom port

# Client (no local qmd or bun needed)
total-recall install --project my-project --client \
  --server-url http://server:7899 --api-key tr_sk_...
```

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `TOTAL_RECALL_SHARED_ROOT` | `~/.ai-memory/knowledge` | Base directory for all project memories |
| `TOTAL_RECALL_ICLOUD_ROOT` | `~/Library/Mobile Documents/.../AI-Memory/knowledge` | iCloud sync directory |
| `TOTAL_RECALL_STATE_ROOT` | `~/.ai-memory/state` | Ingestion state tracking |
| `TOTAL_RECALL_SERVER_URL` | — | Override client server URL |
| `TOTAL_RECALL_API_KEY` | — | Override client API key |
| `TOTAL_RECALL_SYNC_INTERVAL_MINUTES` | `15` | Background sync interval |

## Requirements

**Standalone / Server:**
- macOS (iCloud + launchd features are macOS-only; core memory works anywhere)
- [bun](https://bun.sh) (for installing qmd)
- [qmd](https://github.com/tobi/qmd) (installed automatically)
- Python 3.8+ (for server and session ingestion)

**Client only:**
- Python 3.8+ (for CLI)
- No bun or qmd needed — everything routes to the server

## License

MIT
