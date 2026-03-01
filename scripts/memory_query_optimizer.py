#!/usr/bin/env python3
"""Compact, high-signal formatting for qmd query output."""

from __future__ import annotations

import argparse
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional


KEYWORDS = {
    "fix",
    "fixed",
    "bug",
    "issue",
    "regression",
    "decision",
    "decided",
    "implemented",
    "implementation",
    "verify",
    "verified",
    "test",
    "optimization",
    "optimize",
    "performance",
}

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "with",
}

PATH_RE = re.compile(r"(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+\.[A-Za-z0-9]+")
CODE_FILE_EXTS = {
    ".py", ".ts", ".tsx", ".js", ".jsx", ".swift", ".kt", ".java", ".go", ".rs", ".rb",
    ".php", ".c", ".cc", ".cpp", ".h", ".hpp", ".m", ".mm", ".sh", ".zsh", ".bash",
    ".json", ".yaml", ".yml", ".toml",
}


@dataclass
class MemoryItem:
    source: str
    title: str
    text: str
    token_set: set[str]
    score: float
    recency: float
    overlap: float
    actionability: float
    durable_boost: float
    scope_penalty: float
    intent_boost: float
    cue_boost: float
    distinctiveness: float
    procedural_fit: float
    utility: float
    action_hint: str
    file_hints: list[str]
    command_hints: list[str]
    fix_hints: list[str]
    cue_problem: str
    cue_action: str
    cue_verify: str


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _extract_date(text: str) -> Optional[datetime]:
    match = re.search(r"(20\d{2}-\d{2}-\d{2})", text)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _recency_score(text: str, now: datetime) -> float:
    dt = _extract_date(text)
    if not dt:
        return 0.35
    days = max(0.0, (now - dt).total_seconds() / 86400.0)
    return math.exp(-days / 60.0)


def _parse_score(block: str) -> float:
    match = re.search(r"^Score:\s*([0-9]+(?:\.[0-9]+)?)%?", block, re.MULTILINE)
    if not match:
        return 0.35
    return max(0.0, min(1.0, float(match.group(1)) / 100.0))


def _parse_title(block: str) -> str:
    match = re.search(r"^Title:\s*(.+)$", block, re.MULTILINE)
    if match:
        return match.group(1).strip()
    first = block.strip().splitlines()[0] if block.strip() else "Untitled memory"
    return first[:200]


def _parse_source(block: str) -> str:
    first = block.strip().splitlines()[0] if block.strip() else ""
    return first.strip()[:240]


def _extract_frontmatter_value(block: str, key: str) -> str:
    match = re.search(rf"(?m)^{re.escape(key)}:\s*(.+)$", block)
    return match.group(1).strip().lower() if match else ""


def _extract_cue_value(block: str, cue: str) -> str:
    fm_val = _extract_frontmatter_value(block, cue).strip()
    if fm_val:
        return fm_val

    cue_map = {
        "cue_problem": "problem cue:",
        "cue_action": "action cue:",
        "cue_verify": "verify cue:",
    }
    needle = cue_map.get(cue, "")
    if not needle:
        return ""
    for line in block.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith(f"- {needle}"):
            return stripped.split(":", 1)[1].strip().lower()
    return ""


def _actionability_score(text: str) -> float:
    tokens = _tokenize(text)
    hits = sum(1 for keyword in KEYWORDS if keyword in tokens)
    return min(1.0, hits / 5.0)


def _query_focus_tokens(query: str) -> set[str]:
    return {tok for tok in _tokenize(query) if tok not in STOPWORDS and len(tok) > 2}


def _durable_boost(source: str, text: str) -> float:
    src = source.lower()
    boost = 0.0
    if "/decisions/" in src:
        boost += 0.18
    elif "/patterns/" in src:
        boost += 0.14
    elif "/bugs/" in src:
        boost += 0.1

    lower = text.lower()
    if "durable_signal: high" in lower:
        boost += 0.08
    elif "durable_signal: medium" in lower:
        boost += 0.04
    return boost


def _repo_hints_from_text(text: str) -> set[str]:
    hints: set[str] = set()
    for repo in re.findall(r"/Code/([A-Za-z0-9._-]+)", text):
        hints.add(repo.lower())
    for token in re.findall(r"\b([a-z0-9]+(?:-[a-z0-9]+)+)\b", text.lower()):
        if len(token) > 5:
            hints.add(token)
    return hints


def _scope_penalty(query: str, text: str) -> float:
    query_hints = _repo_hints_from_text(query)
    if not query_hints:
        return 0.0
    text_hints = _repo_hints_from_text(text)
    if not text_hints:
        return 0.0
    if query_hints & text_hints:
        return 0.0
    return -0.12


def _infer_query_intent(query: str) -> str:
    q = query.lower()
    if any(tok in q for tok in ("bug", "fix", "crash", "regression", "error")):
        return "bug-fix"
    if any(tok in q for tok in ("decide", "decision", "tradeoff", "architecture")):
        return "decision"
    if any(tok in q for tok in ("pattern", "workflow", "reusable", "playbook")):
        return "pattern"
    if any(tok in q for tok in ("optimize", "optimization", "performance", "token")):
        return "optimization"
    return "generic"


def _intent_alignment_boost(query: str, block: str) -> float:
    query_intent = _infer_query_intent(query)
    if query_intent == "generic":
        return 0.0
    memory_intent = _extract_frontmatter_value(block, "intent")
    if not memory_intent:
        return 0.0
    if memory_intent == query_intent:
        return 0.08
    return -0.04


def _cue_match_boost(query: str, block: str) -> float:
    q = _tokenize(query)
    if not q:
        return 0.0
    cue_text = " ".join(
        [
            _extract_frontmatter_value(block, "cue_problem"),
            _extract_frontmatter_value(block, "cue_action"),
            _extract_frontmatter_value(block, "cue_verify"),
        ]
    ).strip()
    if not cue_text:
        cue_lines = []
        for line in block.splitlines():
            if line.lower().startswith("- keywords:") or line.lower().startswith("- commands:") or line.lower().startswith("- files:"):
                cue_lines.append(line)
        cue_text = " ".join(cue_lines)
    if not cue_text:
        return 0.0
    overlap = len(_tokenize(cue_text) & q) / max(1, len(q))
    return min(0.08, overlap * 0.12)


def _overlap_score(query: str, text: str) -> float:
    q = _tokenize(query)
    if not q:
        return 0.4
    t = _tokenize(text)
    if not t:
        return 0.0
    shared = len(q & t)
    return shared / len(q)


def _extract_action_hint(block: str) -> str:
    bullets = []
    for line in block.splitlines():
        stripped = line.strip()
        if not stripped.startswith("- "):
            continue
        low = stripped.lower()
        if any(tok in low for tok in ("fixed", "updated", "edited", "changed", "verified", "run ", "pytest", "xcodebuild")):
            bullets.append(stripped[2:])
    if not bullets:
        return ""
    merged = "; ".join(bullets[:2])
    return merged[:220]


def _extract_file_hints(block: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in PATH_RE.findall(block):
        path = raw.strip().strip("`'\"(),.:;")
        if "/" not in path:
            continue
        lower = path.lower()
        ext = "." + lower.rsplit(".", 1)[-1] if "." in lower else ""
        if ext not in CODE_FILE_EXTS:
            continue
        if lower.startswith(("qmd/", "gauntlet/", "sessions/", "bugs/", "patterns/", "decisions/")):
            continue
        if path in seen:
            continue
        seen.add(path)
        out.append(path)
        if len(out) >= 4:
            break
    return out


def _extract_command_hints(block: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in re.findall(r"`([^`]+)`", block):
        cmd = raw.strip()
        lower = cmd.lower()
        if len(cmd) < 5:
            continue
        if not any(tok in lower for tok in ("python", "pytest", "unittest", "xcodebuild", "npm", "bun", "qmd", "swift", "cargo", "go test")):
            continue
        if cmd in seen:
            continue
        seen.add(cmd)
        out.append(cmd)
        if len(out) >= 4:
            break
    return out


def _extract_fix_hints(block: str) -> list[str]:
    out: list[str] = []
    for line in block.splitlines():
        s = line.strip()
        if not s:
            continue
        low = s.lower()
        if not s.startswith("- "):
            continue
        if low.startswith(("- problem cue:", "- action cue:", "- verify cue:", "- files:")):
            continue
        if any(tok in low for tok in ("fixed", "changed", "replace", "replaced", "use ", "using ", "set ", "clamp", "sort", "expire", "boundary", "tie-break", "over-alloc", "round", "threshold", "coupon", "return", "available_at")):
            snippets = [frag.strip() for frag in re.findall(r"`([^`]+)`", s)]
            code_snippets = [
                frag for frag in snippets
                if any(ch in frag for ch in ("=", "(", ")", "<", ">", "+", "-", "*", "/", "[", "]"))
                and re.search(r"[A-Za-z_]", frag)
            ]
            if code_snippets:
                for frag in code_snippets[:2]:
                    delta = f"Set `{frag}`"
                    if delta not in out:
                        out.append(delta)
            else:
                out.append(s[2:].strip())
        if len(out) >= 4:
            break
    return out


def _compress_fix_hint(text: str) -> str:
    s = text.strip().rstrip(".")
    if s.startswith("Set `") and s.endswith("`"):
        return s
    s = re.sub(r"\s+so\s+.+$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+to\s+prevent.+$", "", s, flags=re.IGNORECASE)
    return s[:120].strip()


def _distinctiveness_score(query_tokens: set[str], block_tokens: set[str], doc_freq: dict[str, int], total_docs: int) -> float:
    overlap = query_tokens & block_tokens
    if not overlap or total_docs <= 0:
        return 0.0
    weights = [math.log((1.0 + total_docs) / (1.0 + doc_freq.get(tok, 0))) + 1.0 for tok in overlap]
    max_weight = math.log(1.0 + total_docs / 1.0) + 1.0
    if max_weight <= 0:
        return 0.0
    return max(0.0, min(1.0, (sum(weights) / len(weights)) / max_weight))


def _procedural_fit_score(block: str, fix_hints: list[str], command_hints: list[str]) -> float:
    score = 0.0
    if _extract_cue_value(block, "cue_action"):
        score += 0.35
    if _extract_cue_value(block, "cue_verify"):
        score += 0.25
    if fix_hints:
        score += 0.25
    if command_hints:
        score += 0.15
    return min(1.0, score)


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _parse_items(raw: str, query: str) -> list[MemoryItem]:
    entry_pattern = re.compile(r"(?ms)^(tr://.+?)(?=^tr://|\Z)")
    blocks = [match.group(1).strip() for match in entry_pattern.finditer(raw)]
    if not blocks:
        blocks = [block.strip() for block in re.split(r"\n{2,}", raw) if block.strip()]

    now = datetime.now(timezone.utc)
    query_tokens = _query_focus_tokens(query)
    items: list[MemoryItem] = []
    seen_sources: set[str] = set()
    interim_rows = []
    for block in blocks:
        source = _parse_source(block)
        if source in seen_sources:
            continue
        seen_sources.add(source)
        title = _parse_title(block)
        score = _parse_score(block)
        recency = _recency_score(block, now)
        overlap = _overlap_score(query, block)
        actionability = _actionability_score(block)
        durable_boost = _durable_boost(source, block)
        scope_penalty = _scope_penalty(query, block)
        intent_boost = _intent_alignment_boost(query, block)
        cue_boost = _cue_match_boost(query, block)
        token_set = _tokenize(title + " " + block)
        fix_hints = _extract_fix_hints(block)
        command_hints = _extract_command_hints(block)
        interim_rows.append(
            {
                "source": source,
                "title": title,
                "block": block,
                "score": score,
                "recency": recency,
                "overlap": overlap,
                "actionability": actionability,
                "durable_boost": durable_boost,
                "scope_penalty": scope_penalty,
                "intent_boost": intent_boost,
                "cue_boost": cue_boost,
                "token_set": token_set,
                "action_hint": _extract_action_hint(block),
                "file_hints": _extract_file_hints(block),
                "command_hints": command_hints,
                "fix_hints": fix_hints,
                "cue_problem": _extract_cue_value(block, "cue_problem"),
                "cue_action": _extract_cue_value(block, "cue_action"),
                "cue_verify": _extract_cue_value(block, "cue_verify"),
                "procedural_fit": _procedural_fit_score(block, fix_hints, command_hints),
            }
        )

    if not interim_rows:
        return []

    doc_freq = {
        tok: sum(1 for row in interim_rows if tok in row["token_set"])
        for tok in query_tokens
    }
    total_docs = len(interim_rows)

    for row in interim_rows:
        distinctiveness = _distinctiveness_score(query_tokens, row["token_set"], doc_freq, total_docs)
        utility = (
            (0.22 * row["score"])
            + (0.34 * row["overlap"])
            + (0.16 * row["recency"])
            + (0.1 * row["actionability"])
            + (0.09 * distinctiveness)
            + (0.07 * row["procedural_fit"])
            + row["durable_boost"]
            + row["scope_penalty"]
            + row["intent_boost"]
            + row["cue_boost"]
        )
        items.append(
            MemoryItem(
                source=row["source"],
                title=row["title"],
                text=row["block"],
                token_set=row["token_set"],
                score=row["score"],
                recency=row["recency"],
                overlap=row["overlap"],
                actionability=row["actionability"],
                durable_boost=row["durable_boost"],
                scope_penalty=row["scope_penalty"],
                intent_boost=row["intent_boost"],
                cue_boost=row["cue_boost"],
                distinctiveness=distinctiveness,
                procedural_fit=row["procedural_fit"],
                utility=utility,
                action_hint=row["action_hint"],
                file_hints=row["file_hints"],
                command_hints=row["command_hints"],
                fix_hints=row["fix_hints"],
                cue_problem=row["cue_problem"],
                cue_action=row["cue_action"],
                cue_verify=row["cue_verify"],
            )
        )
    return items


def _mmr_rank(items: list[MemoryItem], diversity_lambda: float = 0.82) -> list[MemoryItem]:
    if not items:
        return []
    remaining = list(items)
    ranked: list[MemoryItem] = []
    while remaining:
        if not ranked:
            best = max(remaining, key=lambda item: item.utility)
            ranked.append(best)
            remaining.remove(best)
            continue
        best_item = None
        best_value = -1e9
        for candidate in remaining:
            max_sim = max(_jaccard(candidate.token_set, prev.token_set) for prev in ranked)
            value = (diversity_lambda * candidate.utility) - ((1.0 - diversity_lambda) * max_sim)
            if value > best_value:
                best_value = value
                best_item = candidate
        ranked.append(best_item)
        remaining.remove(best_item)
    return ranked


def _estimate_tokens(text: str) -> int:
    words = len(re.findall(r"\S+", text))
    return max(1, int(words * 1.3))


def _query_mode(query: str) -> str:
    mode = (os.getenv("TOTAL_RECALL_QUERY_MODE", "auto") or "auto").strip().lower()
    if mode in {"shortlist", "procedural"}:
        return mode
    intent = _infer_query_intent(query)
    if intent in {"bug-fix", "optimization"}:
        return "procedural"
    return "shortlist"


def _procedural_confidence(ranked: list[MemoryItem]) -> float:
    if not ranked:
        return 0.0
    top = ranked[0]
    second = ranked[1] if len(ranked) > 1 else None
    margin = (top.utility - second.utility) if second is not None else top.utility
    margin_score = max(0.0, min(1.0, (margin + 0.02) / 0.2))
    return (
        (0.32 * top.overlap)
        + (0.22 * top.distinctiveness)
        + (0.22 * top.procedural_fit)
        + (0.16 * margin_score)
        + (0.08 * top.actionability)
    )


def _procedural_recipe(ranked: list[MemoryItem], query: str, token_budget: int) -> str:
    picked = [item for item in ranked if (item.overlap >= 0.2 or item.distinctiveness >= 0.35)][:3]
    if not picked:
        picked = ranked[:2]
    if not picked:
        return ""

    files: list[str] = []
    commands: list[str] = []
    fixes: list[str] = []
    sources: list[str] = []
    problem_cue = ""
    action_cue = ""
    verify_cue = ""

    for item in picked:
        if item.source not in sources:
            sources.append(item.source)
        if not problem_cue and item.cue_problem:
            problem_cue = item.cue_problem
        if not action_cue and item.cue_action:
            action_cue = item.cue_action
        if not verify_cue and item.cue_verify:
            verify_cue = item.cue_verify
        for val in item.file_hints:
            if val not in files:
                files.append(val)
        for val in item.command_hints:
            if val not in commands:
                commands.append(val)
        for val in item.fix_hints:
            if val not in fixes:
                fixes.append(val)

    # Fallback to action_hint when fix bullets were not extracted.
    if not fixes:
        for item in picked:
            hint = item.action_hint.strip()
            if hint and hint not in fixes:
                fixes.append(hint)
            if len(fixes) >= 3:
                break

    lines = [f"Memory recipe (budget ≈ {token_budget}):"]
    lines.append(f"Goal: {picked[0].title}")
    if action_cue:
        if problem_cue and verify_cue:
            lines.append(f"If `{problem_cue}` -> `{action_cue}` -> `{verify_cue}`.")
        elif problem_cue:
            lines.append(f"If `{problem_cue}` -> `{action_cue}`.")
        else:
            lines.append(f"Primary action cue: `{action_cue}`.")

    step = 1
    if files:
        lines.append(f"{step}) Edit first:")
        for path in files[:3]:
            lines.append(f"- `{path}`")
        step += 1

    lines.append(f"{step}) Apply code deltas exactly (no comment/doc edits):")
    if fixes:
        compact_fixes: list[str] = []
        for hint in fixes:
            short = _compress_fix_hint(hint)
            if short and short not in compact_fixes:
                compact_fixes.append(short)
            if len(compact_fixes) >= 3:
                break
        for hint in compact_fixes:
            lines.append(f"- {hint}")
    else:
        lines.append("- Reuse the top prior fix with minimal changes and run tests.")
    step += 1

    lines.append(f"{step}) Verify:")
    if commands:
        for cmd in commands[:2]:
            lines.append(f"- `{cmd}`")
    else:
        lines.append('- `python3 -m unittest discover -s tests -p "test_*.py"`')

    if sources:
        lines.append(f"Source: {sources[0]}")

    out = "\n".join(lines)
    est = _estimate_tokens(out)
    if est > token_budget:
        words = re.findall(r"\S+", out)
        soft_cap = max(40, int(token_budget / 1.3))
        out = " ".join(words[:soft_cap]).strip() + " …"
    return out


def optimize_qmd_output(raw: str, query: str, token_budget: int = 900) -> str:
    raw = (raw or "").strip()
    if not raw:
        return "No memory available."
    if os.getenv("TOTAL_RECALL_QUERY_RAW", "") in {"1", "true", "TRUE", "yes"}:
        return raw

    items = _parse_items(raw, query)
    if not items:
        return raw
    ranked = _mmr_rank(items)
    if not ranked:
        return raw

    raw_tokens = _estimate_tokens(raw)
    if _query_mode(query) == "procedural":
        try:
            min_conf = float(os.getenv("TOTAL_RECALL_PROCEDURAL_MIN_CONFIDENCE", "0.58"))
        except ValueError:
            min_conf = 0.58
        proc_conf = _procedural_confidence(ranked)
        recipe = _procedural_recipe(ranked, query, token_budget=token_budget)
        if recipe and (proc_conf >= min_conf) and (_estimate_tokens(recipe) < raw_tokens):
            return recipe

    header = f"Memory shortlist (budget ≈ {token_budget}):"
    out_lines = [header]
    used = _estimate_tokens(header)
    kept = 0
    min_utility = max(0.24, ranked[0].utility * 0.55)
    top_action_hint = ""

    for item in ranked:
        if kept > 0 and item.utility < min_utility:
            continue
        rationale_parts = []
        if item.overlap >= 0.4:
            rationale_parts.append("high query overlap")
        if item.recency >= 0.6:
            rationale_parts.append("recent")
        if item.intent_boost > 0:
            rationale_parts.append("intent aligned")
        if item.cue_boost > 0.02:
            rationale_parts.append("cue match")
        if item.distinctiveness >= 0.4:
            rationale_parts.append("distinctive cues")
        if item.procedural_fit >= 0.55:
            rationale_parts.append("strong action cues")
        if item.actionability >= 0.4:
            rationale_parts.append("implementation details")
        if not rationale_parts:
            rationale_parts.append("strong semantic match")
        rationale = ", ".join(rationale_parts[:2])

        line = (
            f"{kept + 1}. {item.title} "
            f"(utility {item.utility:.2f}, score {item.score:.2f}, durable+ {item.durable_boost:.2f}; {rationale}). "
            f"Source: {item.source}"
        )
        estimated = _estimate_tokens(line)
        if kept > 0 and used + estimated > token_budget:
            break
        out_lines.append(line)
        used += estimated
        kept += 1
        if not top_action_hint and item.action_hint:
            top_action_hint = item.action_hint

    if kept == 0:
        return raw

    if top_action_hint:
        hint_line = f"Top action playbook: {top_action_hint}"
        hint_tokens = _estimate_tokens(hint_line)
        if used + hint_tokens <= token_budget:
            out_lines.append(hint_line)
            used += hint_tokens

    out_lines.append(f"Selected {kept}; estimated tokens ≈ {used}.")
    optimized = "\n".join(out_lines)
    if _estimate_tokens(optimized) >= raw_tokens:
        return raw
    return optimized


def main() -> int:
    parser = argparse.ArgumentParser(description="Optimize qmd query output for token efficiency.")
    parser.add_argument("--query", required=True, help="Original user query text")
    parser.add_argument(
        "--token-budget",
        type=int,
        default=int(os.getenv("TOTAL_RECALL_QUERY_TOKEN_BUDGET", "900")),
    )
    args = parser.parse_args()

    raw = os.sys.stdin.read()
    print(optimize_qmd_output(raw, args.query, token_budget=args.token_budget))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
