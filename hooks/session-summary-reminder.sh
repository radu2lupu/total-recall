#!/bin/bash
INPUT=$(cat)

# Prevent infinite loop — if we already triggered a stop hook continuation, let it end
if [ "$(echo "$INPUT" | jq -r '.stop_hook_active // false')" = "true" ]; then
  exit 0
fi

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
CLIENT_JSON="$HOME/.ai-memory/client.json"

# Activate if local qmd memory exists OR if a remote server is configured
if { [ -d "$KNOWLEDGE_DIR" ] && command -v qmd >/dev/null 2>&1; } || [ -f "$CLIENT_JSON" ]; then
  jq -n --arg cli "$CLI" --arg project "$PROJECT" '{
    "decision": "block",
    "reason": ("BEFORE STOPPING: You must persist what you learned this session. Run:\n`" + $cli + " write --project " + $project + " \"<concise summary: what changed, why, and what was learned>\"`\nThen you may stop. If this session was truly trivial (only reading files, no changes, no decisions), you may skip this — but explain why.")
  }'
else
  exit 0
fi
