---
description: "Rebuild session memory from git history and optionally Codex sessions"
allowed-tools: [Bash, Write, Read, Glob, Grep]
argument-hint: "[--since YYYY-MM-DD] [--codex]"
---

# Rebuild Session Memory

Reconstruct the full session history from git commits and optionally from OpenAI Codex CLI session logs.

## Context

- Project root: !`git -C "$PWD" rev-parse --show-toplevel 2>/dev/null || echo "$PWD"`
- Project name: !`basename "$(git -C "$PWD" remote get-url origin 2>/dev/null | sed 's/.*\///' | sed 's/\.git$//')" 2>/dev/null || basename "$PWD"`
- Knowledge dir: !`PROJECT=$(basename "$(git remote get-url origin 2>/dev/null | sed 's/.*\///' | sed 's/\.git$//')" 2>/dev/null || basename "$PWD"); echo "$HOME/.claude/knowledge/$PROJECT"`
- Git log summary: !`git log --oneline --since="2025-01-01" | wc -l` commits since 2025
- Existing session files: !`PROJECT=$(basename "$(git remote get-url origin 2>/dev/null | sed 's/.*\///' | sed 's/\.git$//')" 2>/dev/null || basename "$PWD"); ls "$HOME/.claude/knowledge/$PROJECT/sessions/" 2>/dev/null | wc -l` files
- Codex sessions exist: !`ls ~/.codex/sessions/ 2>/dev/null && echo "yes" || echo "no"`
- Arguments: $ARGUMENTS

## Your Task

Use the detected project name and knowledge directory from the context above.

### 1. Parse Arguments

- `--since YYYY-MM-DD`: Only rebuild from this date forward (default: all history)
- `--codex`: Also parse Codex CLI `.jsonl` session files from `~/.codex/sessions/`

### 2. Analyze Git History

```bash
git log --format="%H %ai %s" --since="<date>"
```

Group commits into logical sessions by:
- Date proximity (same day = same session usually)
- Topic coherence (related commit messages)
- Author patterns (structured `feat/fix/chore` commits vs informal)

### 3. Parse Codex Sessions (if --codex)

For each `.jsonl` file in `~/.codex/sessions/` (and `~/.codex/archived_sessions/`):
- Extract user messages (role: "user", type: "input_text")
- Skip system instructions and environment context blocks
- Handle both old format (direct content) and new format (event_msg with payload)
- Group by session file (each file = one session)

Use this Python extraction pattern:
```python
import sys, json
for line in sys.stdin:
    try:
        obj = json.loads(line.strip())
        if obj.get('role') == 'user':
            for c in obj.get('content', []):
                text = c.get('text', '')
                if text and not text.startswith('<user_instructions>') and not text.startswith('<environment_context>'):
                    print(text[:500])
    except: pass
```

### 4. Write Session Files

For each identified session, write to `<knowledge-dir>/sessions/YYYY-MM-DD-<topic>.md` using the standard template:

```markdown
# Session: [Brief Title]

**Date:** YYYY-MM-DD
**Topic:** [Category — specific area]
**Tool:** [Claude Code / Codex CLI / Cursor / Manual]

## What Was Done
- [Bullet points]

## Decisions Made
- [Choices and reasoning]

## Lessons Learned
- [Gotchas and insights]
```

Rules:
- Don't overwrite existing session files — skip dates that already have files
- For Codex sessions, note the tool as "OpenAI Codex CLI"
- For Cursor PRs (cursor/ branch prefixes), note as "Cursor"
- Group small related sessions on the same day into one file
- Use concise but informative descriptions

### 5. Re-index and Embed

```bash
qmd update && qmd embed
```

### 6. Report

Tell the user how many session files were created/updated and the total knowledge base size.
