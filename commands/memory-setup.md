---
description: "Set up qmd semantic memory for the current project"
allowed-tools: [Bash, Read, Write, Glob]
---

# Memory Setup

Set up qmd-based semantic memory for the current project. This enables cross-session knowledge persistence via BM25 + vector search.

## Context

- Current directory: !`pwd`
- Git remote: !`git remote get-url origin 2>/dev/null || echo "no git remote"`
- Existing qmd status: !`qmd status 2>/dev/null || echo "qmd not installed or no index"`

## Your Task

Follow these steps in order:

### 1. Check/Install qmd

Check if `qmd` CLI is available. If not, install it:
```bash
bun install -g qmd
```
If `bun` is not available either, inform the user they need to install bun first (`curl -fsSL https://bun.sh/install | bash`).

### 2. Derive Project Name

Determine the project name from:
- Git remote URL (extract repo name, e.g., `ohapotato-ios` from `github.com/.../ohapotato-ios.git`)
- Fall back to current directory basename

### 3. Create Knowledge Directory

Create `~/.claude/knowledge/<project>/` with subdirectories:
```
sessions/     # Dated session summaries
decisions/    # Architectural decisions
patterns/     # Recurring patterns & solutions
bugs/         # Notable bugs & fixes
```

### 4. Create qmd Collection

```bash
qmd collection add ~/.claude/knowledge/<project> --name <project> --mask "**/*.md"
```

If collection already exists, skip this step.

### 5. Add Context

```bash
qmd context add ~/.claude/knowledge/<project> "Session history, architectural decisions, debugging notes, and patterns for the <project> app"
```

### 6. Initial Index + Embed

```bash
qmd update
qmd embed
```

(The `qmd embed` command may crash on cleanup due to a known bun NAPI bug — this is harmless, embeddings still get created.)

### 7. Set Up MEMORY.md

Create or update `~/.claude/projects/<project-path-hash>/memory/MEMORY.md` with instructions for future sessions to query qmd at session start and write knowledge at session end.

The MEMORY.md should include:
- How to query qmd (`qmd search`, `qmd vsearch`, `qmd query`)
- When to write knowledge (after non-trivial sessions)
- Session summary template
- Knowledge directory structure
- Key project reminders (extract from CLAUDE.md if present)
- **Memory citation rule**: When you recall something from qmd that's relevant to the current task, highlight it inline with brain emojis so the user can see the memory in action. Format: `🧠 <recalled insight> 🧠`. For example: "🧠 Last time we hit this bug, the fix was to invalidate the confirmation session before checking staleness 🧠". Only use this for genuinely recalled knowledge, not general reasoning.

### 8. Add qmd MCP Server

Add the qmd MCP server to `~/.claude/settings.json` if not already present:

```json
"mcpServers": {
  "qmd": {
    "command": "qmd",
    "args": ["mcp"]
  }
}
```

**Be idempotent**: If a `qmd` entry already exists in `mcpServers`, skip this step.

### 9. Auto-Approve qmd Bash Commands

Add `"Bash(qmd *)"` to the `allowedTools` array in `~/.claude/settings.json` so qmd commands run without prompting.

If `allowedTools` doesn't exist yet, create it as an array. If it exists but doesn't contain `"Bash(qmd *)"`, append it.

**Be idempotent**: If the entry is already present, skip this step.

### 10. Report Results

Tell the user:
- What was set up
- How many files were indexed
- How to use the memory commands (`/memory-write`, `/memory-rebuild`)
- That future Claude Code sessions will automatically have access to the memory via MEMORY.md
- That the Stop hook is bundled with the plugin and active automatically
