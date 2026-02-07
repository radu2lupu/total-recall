# total-recall

Cross-session semantic memory for [Claude Code](https://claude.com/claude-code). Gives Claude persistent knowledge across conversations using [qmd](https://github.com/tobi/qmd) for BM25 + vector search.

## What It Does

- Stores session summaries, architectural decisions, patterns, and bug notes as markdown files
- Searches them with BM25 full-text and vector similarity via qmd
- Auto-detects project name from git remote
- Reminds you to write a summary when ending a session (Stop hook)
- Provides an MCP server for semantic search within Claude Code

## Installation

In Claude Code:

```
/plugin marketplace add radu2lupu/total-recall
/plugin install total-recall@total-recall
```

Or install prerequisites first via shell:

```bash
bash <(curl -fsSL https://raw.githubusercontent.com/radu2lupu/total-recall/main/scripts/install.sh)
```

### Prerequisites

- [bun](https://bun.sh/) (`curl -fsSL https://bun.sh/install | bash`)
- [jq](https://jqlang.github.io/jq/) (for the Stop hook)

qmd itself is installed automatically by `/memory-setup`.

## Setup

After installing the plugin, open Claude Code in any project and run:

```
/memory-setup
```

This will:
1. Install `qmd` if needed
2. Create `~/.claude/knowledge/<project>/` with subdirectories
3. Initialize the qmd index and embeddings
4. Create a project MEMORY.md for cross-session context
5. Add the qmd MCP server to your settings
6. Auto-approve `qmd` bash commands (so Claude can search memory without prompting)

Run `/memory-setup` once per project.

## Commands

### `/memory-write [description]`

Write a session summary after completing work.

```
/memory-write "fixed auth bug and added rate limiting"
```

### `/memory-rebuild [--since YYYY-MM-DD] [--codex]`

Reconstruct session history from git commits. Useful for backfilling memory on existing projects.

```
/memory-rebuild --since 2025-06-01
/memory-rebuild --codex
```

### `/memory-setup`

Initialize memory for the current project (first-time setup per project).

## Knowledge Directory

```
~/.claude/knowledge/<project>/
├── sessions/     # YYYY-MM-DD-topic.md session summaries
├── decisions/    # Architectural decisions
├── patterns/     # Recurring patterns & solutions
└── bugs/         # Notable bugs & fixes
```

## How It Works

1. **Plugin hooks** — A Stop hook reminds you to run `/memory-write` after non-trivial sessions
2. **Slash commands** — `/memory-write` analyzes git history and conversation context, writes a structured markdown summary
3. **qmd indexing** — Summaries are indexed for BM25 full-text search and embedded for vector similarity
4. **MEMORY.md** — Each project gets a MEMORY.md that's loaded into Claude's system prompt, telling it how to query past knowledge
5. **MCP server** — qmd runs as an MCP server so Claude can search your knowledge base directly

## License

MIT
