---
description: "Write a session summary to qmd memory"
model: sonnet
allowed-tools: [Bash, Write, Read, Glob, Grep]
---

# Write Session Memory

Write a summary of the current session's work to the qmd knowledge base.

## Context

- Today's date: !`date +%Y-%m-%d`
- Project root: !`git -C "$PWD" rev-parse --show-toplevel 2>/dev/null || echo "$PWD"`
- Project name: !`basename "$(git -C "$PWD" remote get-url origin 2>/dev/null | sed 's/.*\///' | sed 's/\.git$//')" 2>/dev/null || basename "$PWD"`
- Knowledge dir: !`PROJECT=$(basename "$(git remote get-url origin 2>/dev/null | sed 's/.*\///' | sed 's/\.git$//')" 2>/dev/null || basename "$PWD"); echo "$HOME/.claude/knowledge/$PROJECT"`
- Recent commits (last 24h): !`git log --oneline --since="24 hours ago" 2>/dev/null || echo "no recent commits"`
- Uncommitted changes: !`git diff --stat HEAD 2>/dev/null || echo "no changes"`
- Untracked files: !`git ls-files --others --exclude-standard 2>/dev/null | head -10`
- Latest session file: !`PROJECT=$(basename "$(git remote get-url origin 2>/dev/null | sed 's/.*\///' | sed 's/\.git$//')" 2>/dev/null || basename "$PWD"); ls -t "$HOME/.claude/knowledge/$PROJECT/sessions/" 2>/dev/null | head -1`
- User request: $ARGUMENTS

## Your Task

Write a session summary capturing what was done in this session. Use the detected project name and knowledge directory from the context above.

### 1. Determine What Was Done

Analyze the git log, diffs, and any context from the conversation to understand what work was completed. If the user provided a description via `$ARGUMENTS`, use that as the primary source.

### 2. Generate Session File

Write the summary to `<knowledge-dir>/sessions/YYYY-MM-DD-<topic>.md` using this template:

```markdown
# Session: [Brief Title]

**Date:** YYYY-MM-DD
**Machine:** !`scutil --get ComputerName 2>/dev/null || hostname -s`
**Topic:** [Category — specific area]

## What Was Done
- [Bullet points of completed work]

## Decisions Made
- [Any choices made and why]

## Lessons Learned
- [Gotchas, insights, things to remember]
```

Rules:
- Topic slug should be kebab-case, 2-4 words (e.g., `ingredient-display-fixes`)
- If a file for today with the same topic already exists, append a number suffix
- Be concise but capture the important details — especially decisions and lessons
- Include specific file paths, function names, or error messages that would help future sessions

### 3. Index and Embed

Run:
```bash
qmd update && qmd embed
```

The `qmd embed` may crash on cleanup (known bun bug) — this is harmless.

### 4. Confirm

Tell the user what file was written and that it's been indexed.
