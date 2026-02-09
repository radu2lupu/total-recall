# total-recall

Install once, then Claude Code + Codex share one `qmd` memory, auto-import session logs, and keep it synced in the background (including iCloud).

## One Command Install

```bash
./scripts/install.sh --project total-recall
```

What `install` does automatically:
- Creates shared memory at `~/.ai-memory/knowledge/<project>`.
- Links:
  - `~/.claude/knowledge/<project>`
  - `~/.codex/knowledge/<project>`
- Enables `qmd` MCP in both:
  - `~/.claude/settings.json`
  - `~/.codex/config.toml`
- Enables iCloud-backed memory (default) at:
  - `~/Library/Mobile Documents/com~apple~CloudDocs/AI-Memory/knowledge/<project>`
- Imports existing Codex and Claude session history into markdown memory files.
- Re-indexes + embeds into `qmd`.
- Injects auto-memory instructions into:
  - `~/.claude/CLAUDE.md`
  - `~/.codex/AGENTS.md`
- Installs a macOS `launchd` background job that re-runs ingestion periodically.

## Claude Plugin Usage

In Claude Code:

```text
/plugin marketplace add radu2lupu/total-recall
/plugin install total-recall@total-recall
```

The plugin commands are still available:
- `/memory-setup`
- `/memory-write`
- `/memory-rebuild`

## Core Commands

```bash
./scripts/total-recall install --project my-project
./scripts/total-recall ingest --project my-project
./scripts/total-recall query --project my-project "retry strategy for upload queue"
./scripts/total-recall write --project my-project "implemented deduplicated retry backoff"
./scripts/total-recall status --project my-project
./scripts/total-recall icloud-status --project my-project
```

## Install Options

```bash
./scripts/total-recall install --project my-project --interval-minutes 10
./scripts/total-recall install --project my-project --no-icloud
./scripts/total-recall install --project my-project --no-launch-agent
./scripts/total-recall install --project my-project --skip-embed
```

## How It Enforces Consistent Memory Usage

The hardest part of cross-session memory isn't the storage — it's getting Claude to
actually **use** it every time. Total Recall uses three reinforcement layers:

### 1. SessionStart hook (strongest signal)
A `SessionStart` hook fires when every session begins and injects context telling
Claude that its **first action** must be to run `total-recall query`. This fires before
Claude even sees the user's first message, so there's no "eager to jump into the task"
failure mode.

### 2. CLAUDE.md instruction block
The `install` command injects a detailed instruction block into `~/.claude/CLAUDE.md`
explaining exactly **when** to query (session start, before design decisions) and
**when not to** (mid-task with full context, trivial changes). Vague instructions like
"query when helpful" don't work — Claude treats them as optional.

### 3. Stop hook (session end)
A `Stop` hook fires when Claude is about to finish and blocks it until it writes
a memory summary. This ensures learnings are persisted, not lost. The hook checks
`stop_hook_active` to prevent infinite loops — on the second firing it lets Claude end.

### Why this matters
Without these hooks, Claude consistently skips memory operations because:
- The query "feels like a detour" before the real work (but it's actually a shortcut)
- Writing memory at session end feels like cleanup that can be skipped
- Vague instructions ("use memory when helpful") are interpreted as optional

The hooks make it structural — Claude can't skip them even if it wants to.

## Notes

- Ingestion state is stored at `~/.ai-memory/state/<project>.json`.
- Imported logs are written under:
  - `~/.ai-memory/knowledge/<project>/sessions/imported/codex/`
  - `~/.ai-memory/knowledge/<project>/sessions/imported/claude/`
- Existing knowledge dirs are preserved as timestamped backups when replaced by symlinks.
- `qmd embed` failures during cleanup are treated as non-fatal.
- For stronger cloud security, enable Apple Advanced Data Protection.

## License

MIT
