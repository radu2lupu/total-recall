# total-recall

Persistent cross-session memory for Claude Code and Codex. Stores decisions, patterns, and lessons learned in a semantic search index (qmd) so every new session starts with full context of past work.

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

- **Session start**: Claude/Codex query past memories for relevant context before starting work
- **Retrieval output is compact**: query results are auto-shortlisted by relevance, recency, and actionability to reduce token usage
- **Project retrieval is isolated**: each project uses its own qmd index to reduce cross-project interference
- **Durable memories are promoted**: high-signal writes are auto-promoted to `decisions/`, `bugs/`, or `patterns/` for stronger future recall
- **Session end**: Claude/Codex write a summary of what was done, decisions made, and lessons learned
- **Mid-session**: Claude/Codex query memory when encountering problems where prior context could help

No manual commands needed. Enforcement layers:

1. **SessionStart hook (Claude plugin)** — fires before Claude sees the user's message, forcing a memory query as the first action
2. **Stop hook (Claude plugin)** — blocks session end until a memory summary is written
3. **Instruction blocks (`CLAUDE.md` + `AGENTS.md`)** — explicit rules for when to query and when to write (used by both Claude and Codex)

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

## Verify Codex Integration

Run a deterministic integration check to ensure Codex has:
- Total Recall instructions in `~/.codex/AGENTS.md`
- qmd MCP wiring in `~/.codex/config.toml` (required for local standalone/server mode; optional in client-mode sandbox checks)
- working `total-recall query` command path

```bash
# Isolated sandbox validation (does not use your real ~/.codex)
python3 scripts/evaluate_codex_integration.py --mode sandbox

# Validate your current local Codex setup
python3 scripts/evaluate_codex_integration.py --mode local --run-query-smoke
```

## Benchmarking Memory Effectiveness

Run integration benchmarks from this repo:

```bash
python3 scripts/evaluate_memory_optimizer.py --project total-recall --query "memory optimization token budget retrieval quality"
python3 scripts/evaluate_memory_write.py --max-session-tokens 420
python3 scripts/evaluate_agent_replay.py --token-budget 900
python3 scripts/evaluate_agent_gauntlet.py --mode agent --token-budget 900 --agent-timeout-seconds 420
# Durable multi-scenario gauntlet suite (recommended trust benchmark):
python3 scripts/evaluate_agent_gauntlet_suite.py --mode agent --repeats 2 --order-policy alternate --agent-timeout-seconds 240 --json
# Optional: add raw top-memory excerpt into prompts (default is off for lower prompt overhead)
python3 scripts/evaluate_agent_gauntlet_suite.py --mode agent --repeats 2 --order-policy alternate --include-memory-excerpt --agent-timeout-seconds 240 --json
# Optional: disable focus-alignment gating (inject memory even if file cues do not match)
python3 scripts/evaluate_agent_gauntlet_suite.py --mode agent --repeats 2 --order-policy alternate --no-focus-alignment-gate --agent-timeout-seconds 240 --json
# Fast deterministic sanity check of suite wiring:
python3 scripts/evaluate_agent_gauntlet_suite.py --mode reference --repeats 1 --json
# Backend comparison (Qdrant vs QMD):
# Local Qdrant engine mode (requires qdrant-client in a venv)
.venv-bench/bin/python scripts/evaluate_qdrant_vs_qmd.py --qdrant-local
# Dedicated service mode (Qdrant must run on :6333)
python3 scripts/evaluate_qdrant_vs_qmd.py --qdrant-url http://127.0.0.1:6333
# Dedicated service mode with ephemeral Docker startup:
python3 scripts/evaluate_qdrant_vs_qmd.py --start-qdrant-docker
# Real project corpus hybrid benchmark (dense+sparse+RRF+rerank):
.venv-bench/bin/python scripts/evaluate_qdrant_hybrid_real.py --project total-recall --sample-size 20 --top-k 5
# Include real-memory gauntlet replay (runs codex exec tracks; slower):
TOTAL_RECALL_GAUNTLET_TIMEOUT=220 .venv-bench/bin/python scripts/evaluate_qdrant_hybrid_real.py --project total-recall --sample-size 8 --top-k 5 --run-gauntlet-replay
# Faster iteration mode with bounded qmd timeouts and compact warm memory cues:
TOTAL_RECALL_GAUNTLET_TIMEOUT=220 .venv-bench/bin/python scripts/evaluate_qdrant_hybrid_real.py --project total-recall --sample-size 6 --top-k 5 --qmd-timeout 20 --qmd-search-timeout 8 --gauntlet-memory-k 2 --gauntlet-token-budget 420 --run-gauntlet-replay
# Stability check across multiple gauntlet runs (median/faster-rate in JSON summary):
TOTAL_RECALL_GAUNTLET_TIMEOUT=220 .venv-bench/bin/python scripts/evaluate_qdrant_hybrid_real.py --project total-recall --sample-size 2 --top-k 5 --qmd-timeout 20 --qmd-search-timeout 8 --gauntlet-memory-k 2 --gauntlet-token-budget 420 --gauntlet-repeats 2 --run-gauntlet-replay --json
```

Gauntlet trust methodology:

- Use multiple scenarios (`benchmarks/gauntlet/scenarios.json`), not a single puzzle.
- Repeat runs with alternating execution order to reduce run-order bias.
- Compare `warm` not only against `cold`, but also against `control` (irrelevant-memory injection) to detect placebo effects from extra prompt text.
- Inspect patch quality (`changed_files_count`, `changed_lines`, `focus_only`) so "faster" does not hide sloppy edits.
- Require memory-focus alignment (`warm_memory_aligned_rate`) so only task-matching memory is injected.
- Trust outcomes only when warm advantage is stable across repeats (median delta and CI, plus pass-rate gates).

Research-to-implementation notes:

- [Human memory research notes](/Users/radulupu/Code/total-recall/docs/human-memory-research-notes.md)

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `TOTAL_RECALL_SHARED_ROOT` | `~/.ai-memory/knowledge` | Base directory for all project memories |
| `TOTAL_RECALL_ICLOUD_ROOT` | `~/Library/Mobile Documents/.../AI-Memory/knowledge` | iCloud sync directory |
| `TOTAL_RECALL_STATE_ROOT` | `~/.ai-memory/state` | Ingestion state tracking |
| `TOTAL_RECALL_SERVER_URL` | — | Override client server URL |
| `TOTAL_RECALL_API_KEY` | — | Override client API key |
| `TOTAL_RECALL_SYNC_INTERVAL_MINUTES` | `15` | Background sync interval |
| `TOTAL_RECALL_QUERY_TOKEN_BUDGET` | `900` | Token budget used for query result shortlisting |
| `TOTAL_RECALL_QUERY_RAW` | `0` | Set to `1` to disable shortlist optimization and return raw qmd output |
| `TOTAL_RECALL_PROCEDURAL_MIN_CONFIDENCE` | `0.58` | Minimum confidence to emit procedural recipe mode; lower confidence falls back to shortlist mode |
| `TOTAL_RECALL_QUERY_MIN_RESULTS` | `4` | Minimum initial query hits before expanding retrieval with search/vsearch |
| `TOTAL_RECALL_QMD_INDEX_PREFIX` | `total-recall` | Prefix used for per-project qmd index files (`<prefix>-<project>`) |

## Requirements

**Standalone / Server:**
- macOS (iCloud + launchd features are macOS-only; core memory works anywhere)
- [bun](https://bun.sh) (for installing qmd)
- [qmd](https://github.com/tobi/qmd) (installed automatically)
- Python 3.8+ (for server and session ingestion)

**Client only:**
- Python 3.8+ (for CLI)
- No bun or qmd needed — everything routes to the server

**Optional (Qdrant benchmark tooling):**
- `qdrant-client` in a virtualenv (for `scripts/evaluate_qdrant_vs_qmd.py --qdrant-local`)
- `fastembed` in the same virtualenv (for `scripts/evaluate_qdrant_hybrid_real.py`)

## License

MIT
