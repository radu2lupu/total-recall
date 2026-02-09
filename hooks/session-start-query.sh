#!/bin/bash
# SessionStart hook — tells Claude to query Total Recall before doing anything else.
INPUT=$(cat)

# Get cwd from hook input (more reliable than $PWD in hook context)
CWD=$(echo "$INPUT" | jq -r '.cwd // empty')
CWD="${CWD:-$PWD}"

# Ensure bun-installed globals (like qmd) are on PATH
for p in "$HOME/.bun/bin" "/opt/homebrew/bin" "/usr/local/bin"; do
  [ -d "$p" ] && export PATH="$p:$PATH"
done

# Extract project name from the total-recall instruction block in CLAUDE.md
# This is the most reliable source — it's set during `total-recall install`
PROJECT=""
CLAUDE_MD="$HOME/.claude/CLAUDE.md"
if [ -f "$CLAUDE_MD" ]; then
  PROJECT=$(sed -n 's/.*Total Recall Auto-Memory (\([^)]*\)).*/\1/p' "$CLAUDE_MD" | head -1)
fi

# Fallback: derive from git remote or directory name
if [ -z "$PROJECT" ]; then
  PROJECT=$(basename "$(git -C "$CWD" remote get-url origin 2>/dev/null | sed 's/.*\///' | sed 's/\.git$//')" 2>/dev/null || basename "$CWD")
fi

CLI="${CLAUDE_PLUGIN_ROOT}/scripts/total-recall"
KNOWLEDGE_DIR="$HOME/.ai-memory/knowledge/$PROJECT"

if [ -d "$KNOWLEDGE_DIR" ] && command -v qmd >/dev/null 2>&1; then
  jq -n --arg cli "$CLI" --arg project "$PROJECT" '{
    "hookSpecificOutput": {
      "hookEventName": "SessionStart",
      "additionalContext": ("TOTAL RECALL MEMORY SYSTEM ACTIVE.\nYour FIRST action — before reading files, before planning, before anything — MUST be to query prior session memory:\n`" + $cli + " query --project " + $project + " \"<what the user is asking about>\"`\nThis retrieves context from all previous sessions. Without it, you will repeat past mistakes and miss prior decisions. This is mandatory, not optional.")
    }
  }'
fi

exit 0
