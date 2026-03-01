#!/usr/bin/env python3
"""Evaluate memory query optimization quality and token efficiency."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

from memory_query_optimizer import optimize_qmd_output


def estimate_tokens(text: str) -> int:
    words = len(re.findall(r"\S+", text))
    return max(1, int(words * 1.3))


def query_terms(query: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", query.lower()))


def coverage(query: str, text: str) -> float:
    terms = query_terms(query)
    if not terms:
        return 1.0
    tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
    return len(terms & tokens) / len(terms)


def repo_hints(text: str) -> set[str]:
    hints = set(re.findall(r"/Code/([A-Za-z0-9._-]+)", text))
    hints |= set(re.findall(r"\b([a-z0-9]+(?:-[a-z0-9]+)+)\b", text.lower()))
    return {h.lower() for h in hints if len(h) > 5}


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return re.sub(r"-+", "-", slug) or "memory"


def qmd_index_name(project: str) -> str:
    prefix = slugify(os.getenv("TOTAL_RECALL_QMD_INDEX_PREFIX", "total-recall"))
    return f"{prefix}-{slugify(project)}"


def ensure_index_ready(project: str) -> None:
    project_slug = slugify(project)
    shared_root = os.path.realpath(
        os.path.expanduser(os.getenv("TOTAL_RECALL_SHARED_ROOT", "~/.ai-memory/knowledge"))
    )
    project_dir = os.path.join(shared_root, project_slug)
    if not os.path.isdir(project_dir):
        return

    index = qmd_index_name(project_slug)
    cmd_base = ["qmd", "--index", index]

    try:
        ls_result = subprocess.run(
            [*cmd_base, "ls", project_slug],
            capture_output=True, text=True, timeout=20,
        )
        if ls_result.returncode != 0:
            subprocess.run(
                [*cmd_base, "collection", "add", project_dir, "--name", project_slug, "--mask", "**/*.md"],
                capture_output=True, text=True, timeout=20,
            )

        ctx_result = subprocess.run(
            [*cmd_base, "context", "list"],
            capture_output=True, text=True, timeout=20,
        )
        if ctx_result.returncode != 0 or project_dir not in (ctx_result.stdout or ""):
            subprocess.run(
                [*cmd_base, "context", "add", project_dir, f"Shared Claude + Codex memory for project {project_slug}"],
                capture_output=True, text=True, timeout=20,
            )
    except Exception:
        return


def run_qmd(query: str, project: str) -> str:
    project_slug = slugify(project)
    ensure_index_ready(project_slug)
    index = qmd_index_name(project_slug)
    commands = [
        ["qmd", "--index", index, "query", query, "-c", project_slug],
        ["qmd", "--index", index, "search", query, "-c", project_slug],
        ["qmd", "--index", index, "query", query],
    ]
    for cmd in commands:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        except Exception:
            continue
        text = (result.stdout or "").strip()
        if text and (result.returncode == 0 or "qmd://" in text):
            return result.stdout.strip()
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate total-recall memory optimization quality.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--token-budget", type=int, default=900)
    parser.add_argument("--raw-file", help="Optional file containing raw qmd output")
    args = parser.parse_args()

    raw = ""
    if args.raw_file:
        with open(args.raw_file, "r", encoding="utf-8") as f:
            raw = f.read().strip()
    else:
        raw = run_qmd(args.query, args.project)

    if not raw:
        print("FAIL: no raw memory results available for evaluation")
        return 2

    optimized = optimize_qmd_output(raw, args.query, token_budget=args.token_budget)
    raw_tokens = estimate_tokens(raw)
    optimized_tokens = estimate_tokens(optimized)
    compression = 1.0 - (optimized_tokens / max(1, raw_tokens))
    raw_coverage = coverage(args.query, raw)
    optimized_coverage = coverage(args.query, optimized)
    selected = len(re.findall(r"^\d+[.)]\s", optimized, flags=re.MULTILINE))
    procedural_mode = optimized.lstrip().startswith("Memory recipe")
    query_hints = repo_hints(args.query)
    source_lines = re.findall(r"^.*Source:\s+(.+)$", optimized, flags=re.MULTILINE)
    if not source_lines:
        in_sources = False
        for line in optimized.splitlines():
            stripped = line.strip()
            if stripped.lower() == "sources:":
                in_sources = True
                continue
            if in_sources:
                if stripped.startswith("- "):
                    source_lines.append(stripped[2:].strip())
                    continue
                if stripped:
                    in_sources = False
    off_topic = 0
    for source in source_lines:
        source_hints = repo_hints(source)
        if query_hints and source_hints and not (query_hints & source_hints):
            off_topic += 1

    print(f"Raw tokens: {raw_tokens}")
    print(f"Optimized tokens: {optimized_tokens}")
    print(f"Compression: {compression:.1%}")
    print(f"Query coverage (raw): {raw_coverage:.2f}")
    print(f"Query coverage (optimized): {optimized_coverage:.2f}")
    print(f"Selected memories: {selected}")
    print(f"Procedural mode: {procedural_mode}")
    print(f"Off-topic selected: {off_topic}")

    checks = {
        "budget": optimized_tokens <= int(args.token_budget * 1.1),
        "coverage": optimized_coverage >= (raw_coverage - 0.05),
        "selection": (selected >= 1) or procedural_mode,
        "scope": (not query_hints) or (off_topic <= max(1, selected // 2)),
    }

    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        print(f"FAIL: checks failed -> {', '.join(failed)}")
        return 1

    print("PASS: optimization meets token and relevance criteria")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
