#!/usr/bin/env python3
"""Build and persist structured memory notes with retrieval cues."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Optional


STOPWORDS = {
    "about",
    "after",
    "again",
    "along",
    "also",
    "and",
    "because",
    "before",
    "between",
    "could",
    "from",
    "have",
    "into",
    "only",
    "that",
    "this",
    "then",
    "they",
    "what",
    "when",
    "where",
    "with",
    "would",
    "your",
    "were",
    "while",
    "using",
}

INTENT_RULES = [
    ("decision", ("decision", "decided", "choose", "chose", "tradeoff", "architecture")),
    ("bug-fix", ("bug", "fixed", "fix", "regression", "error", "failure", "crash")),
    ("pattern", ("pattern", "playbook", "template", "reusable", "standardize")),
    ("optimization", ("optimize", "optimization", "performance", "token", "memory")),
    ("implementation", ("implement", "added", "built", "created", "wired", "integrated")),
]

OUTCOME_RULES = [
    ("verified", ("verified", "pass", "passed", "success", "resolved", "working")),
    ("investigated", ("investigated", "analyzed", "debugged", "traced")),
]


@dataclass
class CueData:
    topic: str
    intent: str
    outcome: str
    confidence: float
    durable_signal: str
    keywords: list[str]
    files: list[str]
    commands: list[str]
    cue_problem: str
    cue_action: str
    cue_verify: str


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return re.sub(r"-+", "-", slug) or "memory"


def unique_path(directory: Path, stem: str, suffix: str = ".md") -> Path:
    candidate = directory / f"{stem}{suffix}"
    counter = 2
    while candidate.exists():
        candidate = directory / f"{stem}-{counter}{suffix}"
        counter += 1
    return candidate


def classify(summary: str, rules: list[tuple[str, tuple[str, ...]]], fallback: str) -> str:
    lower = summary.lower()
    for label, keys in rules:
        if any(key in lower for key in keys):
            return label
    return fallback


def extract_files(summary: str) -> list[str]:
    pattern = re.compile(r"(?:~?/)?(?:[\w.\-]+/)+[\w.\-]+")
    found = []
    for match in pattern.findall(summary):
        if "/" not in match:
            continue
        clean = match.strip(".,:;()[]{}")
        if clean and clean not in found:
            found.append(clean)
        if len(found) >= 4:
            break
    return found


def extract_commands(summary: str) -> list[str]:
    commands: list[str] = []
    for cmd in re.findall(r"`([^`]+)`", summary):
        clean = cmd.strip()
        if clean and clean not in commands:
            commands.append(clean)
    cli_pattern = re.compile(
        r"\b(?:git|qmd|python3?|bash|zsh|bun|npm|pnpm|yarn|xcodebuild|swift|make|pytest|go|cargo|docker)\b(?: [^.;,\n]+)?"
    )
    for cmd in cli_pattern.findall(summary):
        clean = cmd.strip()
        if clean and clean not in commands:
            commands.append(clean)
        if len(commands) >= 4:
            break
    return commands


def extract_keywords(summary: str, limit: int = 8) -> list[str]:
    words = [w.lower() for w in re.findall(r"[a-zA-Z][a-zA-Z0-9\-]{2,}", summary)]
    kept = [w for w in words if w not in STOPWORDS]
    counts = Counter(kept)
    ordered = sorted(counts.items(), key=lambda item: (-item[1], words.index(item[0])))
    return [word for word, _ in ordered[:limit]]


def estimate_confidence(intent: str, outcome: str, summary: str) -> float:
    conf = 0.55
    if intent in {"decision", "bug-fix", "pattern"}:
        conf += 0.1
    if outcome == "verified":
        conf += 0.2
    if any(t in summary.lower() for t in ("maybe", "unclear", "might", "likely")):
        conf -= 0.15
    return round(max(0.35, min(0.95, conf)), 2)


def estimate_durable_signal(intent: str, outcome: str, files: list[str], commands: list[str], keywords: list[str]) -> str:
    score = 0
    if intent in {"decision", "bug-fix", "pattern"}:
        score += 2
    if outcome == "verified":
        score += 1
    if files:
        score += 1
    if commands:
        score += 1
    if len(keywords) >= 5:
        score += 1
    if score >= 4:
        return "high"
    if score >= 2:
        return "medium"
    return "low"


def build_problem_cue(summary: str) -> str:
    text = summary.strip()
    for splitter in (" by ", " to ", " and "):
        if splitter in text.lower():
            idx = text.lower().index(splitter)
            text = text[:idx]
            break
    return text[:140]


def build_action_cue(summary: str, files: list[str], commands: list[str]) -> str:
    if files:
        return f"Edit {files[0]}"
    lower = summary.lower()
    for verb in ("fix", "fixed", "implement", "implemented", "update", "updated", "refactor", "wire"):
        if verb in lower:
            return f"{verb.capitalize()} core logic from session summary"
    if commands:
        return f"Run {commands[0]}"
    return "Apply prior proven change pattern"


def build_verify_cue(commands: list[str]) -> str:
    for command in commands:
        low = command.lower()
        if any(tok in low for tok in ("pytest", "xcodebuild", "test", "unittest", "swift test")):
            return command
    return commands[0] if commands else "Run targeted regression test for changed area"


def build_cues(summary: str) -> CueData:
    topic = slugify(" ".join(summary.split()[:6]))
    intent = classify(summary, INTENT_RULES, "investigation")
    outcome = classify(summary, OUTCOME_RULES, "noted")
    files = extract_files(summary)
    commands = extract_commands(summary)
    keywords = extract_keywords(summary)
    cue_problem = build_problem_cue(summary)
    cue_action = build_action_cue(summary, files, commands)
    cue_verify = build_verify_cue(commands)
    confidence = estimate_confidence(intent, outcome, summary)
    durable_signal = estimate_durable_signal(intent, outcome, files, commands, keywords)
    return CueData(
        topic=topic,
        intent=intent,
        outcome=outcome,
        confidence=confidence,
        durable_signal=durable_signal,
        keywords=keywords,
        files=files,
        commands=commands,
        cue_problem=cue_problem,
        cue_action=cue_action,
        cue_verify=cue_verify,
    )


def build_tags(cues: CueData) -> list[str]:
    tags = [cues.intent, cues.outcome, cues.durable_signal]
    for keyword in cues.keywords[:5]:
        tags.append(slugify(keyword))
    deduped: list[str] = []
    for tag in tags:
        if tag not in deduped:
            deduped.append(tag)
    return deduped


def should_promote_to_patterns(summary: str, cues: CueData) -> bool:
    lower = summary.lower()
    if cues.intent == "pattern":
        return True
    return any(token in lower for token in ("reusable", "playbook", "workflow", "standardize", "guideline"))


def promotion_directories(summary: str, cues: CueData) -> list[str]:
    dirs: list[str] = []
    if cues.intent == "decision":
        dirs.append("decisions")
    if cues.intent == "bug-fix":
        dirs.append("bugs")
    if should_promote_to_patterns(summary, cues):
        dirs.append("patterns")
    return dirs


def render_session_markdown(summary: str, project: str, machine: str, date_str: str, cues: CueData, session_rel_path: str) -> str:
    tags = ", ".join(build_tags(cues))
    keywords = ", ".join(cues.keywords) if cues.keywords else "(none)"
    files = ", ".join(cues.files) if cues.files else "(none)"
    commands = ", ".join(cues.commands) if cues.commands else "(none)"
    return (
        f"---\n"
        f"memory_kind: session\n"
        f"intent: {cues.intent}\n"
        f"outcome: {cues.outcome}\n"
        f"confidence: {cues.confidence:.2f}\n"
        f"durable_signal: {cues.durable_signal}\n"
        f"cue_problem: {cues.cue_problem}\n"
        f"cue_action: {cues.cue_action}\n"
        f"cue_verify: {cues.cue_verify}\n"
        f"tags: [{tags}]\n"
        f"session_path: {session_rel_path}\n"
        f"---\n\n"
        f"# Session: {summary}\n\n"
        f"**Date:** {date_str}\n"
        f"**Project:** {project}\n"
        f"**Machine:** {machine}\n"
        f"**Topic:** {cues.topic}\n"
        f"**Tool:** Claude Code / Codex\n"
        f"**Intent:** {cues.intent}\n"
        f"**Outcome:** {cues.outcome}\n"
        f"**Confidence:** {cues.confidence:.2f}\n\n"
        f"## What Was Done\n"
        f"- {summary}\n\n"
        f"## Retrieval Cues\n"
        f"- Problem cue: {cues.cue_problem}\n"
        f"- Action cue: {cues.cue_action}\n"
        f"- Verify cue: {cues.cue_verify}\n"
        f"- Keywords: {keywords}\n"
        f"- Files: {files}\n"
        f"- Commands: {commands}\n"
        f"- Durable signal: {cues.durable_signal}\n\n"
        f"## Decisions Made\n"
        f"- (auto-generated note)\n\n"
        f"## Lessons Learned\n"
        f"- Keep memory notes concise and include concrete file paths or commands when relevant.\n"
    )


def render_promoted_markdown(summary: str, project: str, date_str: str, cues: CueData, source_rel_path: str, kind: str) -> str:
    tags = ", ".join(build_tags(cues))
    keywords = ", ".join(cues.keywords) if cues.keywords else "(none)"
    heading = kind[:-1].capitalize() if kind.endswith("s") else kind.capitalize()
    return (
        f"---\n"
        f"memory_kind: {kind[:-1] if kind.endswith('s') else kind}\n"
        f"intent: {cues.intent}\n"
        f"outcome: {cues.outcome}\n"
        f"confidence: {cues.confidence:.2f}\n"
        f"tags: [{tags}]\n"
        f"source_session: {source_rel_path}\n"
        f"---\n\n"
        f"# {heading}: {summary}\n\n"
        f"**Date:** {date_str}\n"
        f"**Project:** {project}\n"
        f"**Source Session:** {source_rel_path}\n"
        f"**Intent:** {cues.intent}\n"
        f"**Outcome:** {cues.outcome}\n"
        f"**Confidence:** {cues.confidence:.2f}\n\n"
        f"## Summary\n"
        f"- {summary}\n\n"
        f"## Retrieval Cues\n"
        f"- Problem cue: {cues.cue_problem}\n"
        f"- Action cue: {cues.cue_action}\n"
        f"- Verify cue: {cues.cue_verify}\n"
        f"- Keywords: {keywords}\n"
    )


def extract_metadata_value(content: str, key: str) -> str:
    match = re.search(rf"^{re.escape(key)}:\s*(.+)$", content, flags=re.MULTILINE)
    return match.group(1).strip() if match else ""


def extract_session_title(content: str) -> str:
    match = re.search(r"^# Session:\s*(.+)$", content, flags=re.MULTILINE)
    return match.group(1).strip() if match else ""


def maybe_write_topic_synthesis(project_dir: Path, project: str, date_str: str, topic: str) -> Optional[str]:
    sessions_dir = project_dir / "sessions"
    topic_files = sorted(sessions_dir.glob(f"*-{topic}*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
    if len(topic_files) < 2:
        return None

    recent = topic_files[:5]
    titles = []
    intents = []
    outcomes = []
    for path in recent:
        content = path.read_text(encoding="utf-8")
        title = extract_session_title(content)
        if title:
            titles.append(title)
        intent = extract_metadata_value(content, "intent")
        outcome = extract_metadata_value(content, "outcome")
        if intent:
            intents.append(intent)
        if outcome:
            outcomes.append(outcome)

    if not titles:
        return None

    intent_counts = Counter(intents)
    outcome_counts = Counter(outcomes)
    dominant_intent = intent_counts.most_common(1)[0][0] if intent_counts else "investigation"
    dominant_outcome = outcome_counts.most_common(1)[0][0] if outcome_counts else "noted"

    pattern_file = project_dir / "patterns" / f"{topic}-synthesized.md"
    source_lines = "\n".join(f"- sessions/{path.name}" for path in recent)
    summary_lines = "\n".join(f"- {title}" for title in titles[:3])
    pattern_text = (
        f"---\n"
        f"memory_kind: pattern\n"
        f"intent: {dominant_intent}\n"
        f"outcome: {dominant_outcome}\n"
        f"confidence: 0.75\n"
        f"tags: [pattern, synthesized, {topic}]\n"
        f"---\n\n"
        f"# Pattern: {topic} synthesized memory\n\n"
        f"**Date:** {date_str}\n"
        f"**Project:** {project}\n"
        f"**Source Count:** {len(recent)}\n\n"
        f"## Consolidated Takeaways\n"
        f"{summary_lines}\n\n"
        f"## Source Sessions\n"
        f"{source_lines}\n"
    )
    pattern_file.write_text(pattern_text, encoding="utf-8")
    return str(pattern_file)


def write_memory_files(project_dir: Path, project: str, summary: str, machine: str, date_str: Optional[str] = None) -> dict[str, Any]:
    date_str = date_str or date.today().isoformat()
    project_dir.mkdir(parents=True, exist_ok=True)
    sessions_dir = project_dir / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "decisions").mkdir(exist_ok=True)
    (project_dir / "patterns").mkdir(exist_ok=True)
    (project_dir / "bugs").mkdir(exist_ok=True)

    cues = build_cues(summary)
    session_path = unique_path(sessions_dir, f"{date_str}-{cues.topic}")
    session_rel = session_path.relative_to(project_dir).as_posix()
    session_md = render_session_markdown(summary, project, machine, date_str, cues, session_rel)
    session_path.write_text(session_md, encoding="utf-8")

    promoted_files: list[str] = []
    for directory in promotion_directories(summary, cues):
        target_dir = project_dir / directory
        target_dir.mkdir(exist_ok=True)
        target_path = unique_path(target_dir, f"{date_str}-{cues.topic}")
        promoted_md = render_promoted_markdown(summary, project, date_str, cues, session_rel, directory)
        target_path.write_text(promoted_md, encoding="utf-8")
        promoted_files.append(str(target_path))

    synthesized = maybe_write_topic_synthesis(project_dir, project, date_str, cues.topic)
    if synthesized:
        promoted_files.append(synthesized)

    return {
        "session_file": str(session_path),
        "promoted_files": promoted_files,
        "topic": cues.topic,
        "intent": cues.intent,
        "outcome": cues.outcome,
        "confidence": cues.confidence,
        "durable_signal": cues.durable_signal,
        "tags": build_tags(cues),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Write structured memory note files.")
    parser.add_argument("write", nargs="?")
    parser.add_argument("--project-dir", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--machine", required=True)
    parser.add_argument("--date")
    args = parser.parse_args()

    result = write_memory_files(
        Path(args.project_dir),
        project=args.project,
        summary=args.summary,
        machine=args.machine,
        date_str=args.date,
    )
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
