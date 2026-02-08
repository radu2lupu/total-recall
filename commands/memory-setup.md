---
description: "Set up total-recall memory for the current project"
allowed-tools: [Bash, Read, Write, Glob]
---

# Total Recall Setup

Set up cross-session semantic memory for the current project. This runs the full `total-recall install` — shared memory, session ingestion, qmd indexing, and background sync.

## Context

- Current directory: !`pwd`
- Git remote: !`git remote get-url origin 2>/dev/null || echo "no git remote"`
- Detected project name: !`basename "$(git remote get-url origin 2>/dev/null | sed 's/.*\///' | sed 's/\.git$//')" 2>/dev/null || basename "$PWD"`
- Platform: !`uname -s`
- iCloud Drive exists: !`[ -d "$HOME/Library/Mobile Documents/com~apple~CloudDocs" ] && echo "yes" || echo "no"`
- qmd installed: !`command -v qmd >/dev/null 2>&1 && echo "yes" || echo "no"`
- bun installed: !`command -v bun >/dev/null 2>&1 && echo "yes" || echo "no"`
- Plugin root: !`echo "${CLAUDE_PLUGIN_ROOT:-unknown}"`
- User arguments: $ARGUMENTS

## Your Task

### 1. Pre-flight check

If `qmd` is not installed and `bun` is not installed either, stop and tell the user:
```
Install bun first: curl -fsSL https://bun.sh/install | bash
Then re-run /memory-setup
```

If `bun` is available but `qmd` is not, install it:
```bash
bun install -g qmd
```

### 2. Ask the user (interactive)

Ask the user these questions **before** running anything. Present them all at once so the user can answer in one go.

**Project name:** Show the auto-detected name from context above. Ask if it's correct or if they want a different name. If the user provided a name via `$ARGUMENTS`, use that without asking.

**iCloud backup:** Only ask on macOS when iCloud Drive exists. Default yes. Explain: "This syncs your memory to iCloud so it survives machine changes."

**Background sync:** Only ask on macOS. Default yes. Explain: "This installs a background job that imports new Claude/Codex session logs every 15 minutes."

If the platform is not macOS, skip the iCloud and background sync questions — use `--no-icloud --no-launch-agent`.

### 3. Run total-recall install

Build the command from the user's answers:

```bash
<plugin-root>/scripts/total-recall install \
  --project <project-name> \
  [--no-icloud] \
  [--no-launch-agent]
```

Where `<plugin-root>` is the Plugin root from context above. Run it and let the output stream.

### 4. Report results

Tell the user:
- Setup is complete
- How to write memory: `/memory-write` or `total-recall write --project <name> "summary"`
- How to query memory: `total-recall query --project <name> "query"`
- That the Stop hook will remind them to save memory when ending sessions
- That `/memory-rebuild` can backfill memory from git history
