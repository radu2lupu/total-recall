# total-recall

Persistent cross-session memory for Claude Code. Stores decisions, patterns, and lessons learned in a semantic search index (qmd) so every new session starts with full context of past work.

Works on a single machine with iCloud backup, or across multiple machines via a built-in HTTP server with a web UI for browsing and editing memories.

## Quick Start

### Claude Code Plugin (recommended)

```text
/install radu2lupu/total-recall
```

Then run `/memory-setup` inside any project to configure it.

### Manual Install

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

Share memories across machines on the same network using the built-in HTTP server.

### On the server machine

```bash
total-recall server init          # Generate API key
total-recall server start         # Start on port 7899
total-recall server install-launchd  # Auto-start on boot (macOS)
```

### On client machines

```bash
total-recall client configure \
  --server-url http://server.tailscale:7899 \
  --api-key tr_sk_...

total-recall client status        # Verify connection
```

Once configured, all `write`, `query`, and `ingest` commands route to the server automatically. If the server is unreachable, operations fall back to local qmd.

### Web UI

The server includes a web UI at `http://server:7899/` for browsing, viewing, editing, and deleting memories. Memories are grouped by project with metadata pills showing date, machine, and source tool.

## All Commands

```
total-recall install    --project NAME   Full setup: shared dirs, symlinks, qmd, hooks, ingestion
total-recall write      --project NAME   "<summary>"  Write a memory note
total-recall query      --project NAME   "<query>"    Semantic search across memories
total-recall ingest     --project NAME   Import Claude/Codex session logs
total-recall status     --project NAME   Show memory stats and configuration
total-recall setup      --project NAME   Link dirs + configure qmd (no ingestion)

total-recall icloud-enable   --project NAME   Move memory to iCloud Drive
total-recall icloud-sync     --project NAME   Manual iCloud push/pull
total-recall icloud-status   --project NAME   Check iCloud sync state

total-recall server init              Generate server config + API key
total-recall server start             Start HTTP API server
total-recall server stop              Stop the server
total-recall server status            Check if server is running
total-recall server add-key           Generate additional API key
total-recall server install-launchd   Auto-start on boot (macOS)

total-recall client configure         Connect to a remote server
total-recall client status            Check client connection
total-recall client enable            Re-enable remote mode
total-recall client disable           Switch back to local mode
```

### Plugin Slash Commands

When installed as a Claude Code plugin:

- `/memory-setup` — Interactive setup wizard for the current project
- `/memory-write` — Write a session summary to memory
- `/memory-rebuild` — Rebuild memory from git history and Codex sessions

## Install Options

```bash
total-recall install --project my-project --no-icloud          # Skip iCloud backup
total-recall install --project my-project --no-launch-agent    # Skip background sync
total-recall install --project my-project --interval-minutes 10 # Sync every 10 min
total-recall install --project my-project --skip-embed         # Skip vector embeddings
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

- macOS (iCloud + launchd features are macOS-only; core memory works anywhere)
- [bun](https://bun.sh) (for installing qmd)
- [qmd](https://github.com/tobi/qmd) (installed automatically)
- Python 3.8+ (for server and session ingestion)

## License

MIT
