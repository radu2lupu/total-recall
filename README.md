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
