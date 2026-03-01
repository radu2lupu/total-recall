#!/usr/bin/env python3
"""Durable gauntlet suite: multi-scenario, repeated cold vs warm runs with aggregate stats."""

from __future__ import annotations

import argparse
import difflib
import json
import random
import re
import shutil
import subprocess
import tempfile
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Dict, List, Optional, Sequence, Tuple

from memory_query_optimizer import optimize_qmd_output


ROOT = Path(__file__).resolve().parent.parent
GAUNTLET_DIR = ROOT / "benchmarks" / "gauntlet"
DEFAULT_SCENARIOS_FILE = GAUNTLET_DIR / "scenarios.json"
DEFAULT_TEST_CMD = ["python3", "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"]


@dataclass
class Scenario:
    id: str
    title: str
    template_dir: Path
    query: str
    focus_files: List[str]
    memory_source: str
    memory_title: str
    memory_bullets: List[str]
    reference_replacements: List[Dict[str, str]]


@dataclass
class TrackResult:
    name: str
    passed: bool
    duration_seconds: float
    return_code: int
    tests_output: str
    agent_stdout: str
    agent_stderr: str
    workspace: Path
    changed_files: List[str]
    changed_files_count: int
    changed_lines: int
    focus_only: bool


@dataclass
class ScenarioRun:
    scenario_id: str
    repeat: int
    order: str
    cold: TrackResult
    control: Optional[TrackResult]
    warm: TrackResult
    shortlist: str
    control_shortlist: str
    warm_memory_aligned: bool
    control_memory_aligned: bool


def run_cmd(cmd: List[str], cwd: Path, timeout: int, input_text: Optional[str] = None) -> Tuple[int, str, str]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            input=input_text,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        return proc.returncode, proc.stdout or "", proc.stderr or ""
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        err = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        err += f"\nTIMEOUT after {timeout}s\n"
        return 124, out, err


def run_tests(workspace: Path, timeout: int) -> Tuple[bool, str]:
    rc, out, err = run_cmd(DEFAULT_TEST_CMD, workspace, timeout=timeout)
    text = (out + "\n" + err).strip()
    return rc == 0, text


def _read_text_best_effort(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return path.read_text(encoding="latin-1")
        except Exception:
            return None
    except Exception:
        return None


def _count_line_delta(left: Optional[str], right: Optional[str]) -> int:
    if left == right:
        return 0
    if left is None or right is None:
        return max(1, len((left or right or "").splitlines()))
    diff = difflib.ndiff(left.splitlines(), right.splitlines())
    return sum(1 for line in diff if line.startswith("- ") or line.startswith("+ "))


def _is_ignored_relpath(rel: str) -> bool:
    if rel.endswith(".pyc") or rel.endswith(".pyo"):
        return True
    if rel.endswith(".DS_Store"):
        return True
    if "__pycache__/" in rel:
        return True
    if rel.startswith(".pytest_cache/"):
        return True
    return False


def workspace_delta(template_dir: Path, workspace: Path) -> Tuple[List[str], int]:
    template_files = {
        str(p.relative_to(template_dir))
        for p in template_dir.rglob("*")
        if p.is_file()
    }
    workspace_files = {
        str(p.relative_to(workspace))
        for p in workspace.rglob("*")
        if p.is_file()
    }
    rel_paths = sorted(rel for rel in (template_files | workspace_files) if not _is_ignored_relpath(rel))
    changed_files: List[str] = []
    changed_lines = 0

    for rel in rel_paths:
        left_path = template_dir / rel
        right_path = workspace / rel
        left = _read_text_best_effort(left_path) if left_path.exists() else None
        right = _read_text_best_effort(right_path) if right_path.exists() else None
        if left != right:
            changed_files.append(rel)
            changed_lines += _count_line_delta(left, right)

    return changed_files, changed_lines


def default_agent_cmd(workspace: Path, timeout: int, prompt: str) -> Tuple[int, str, str]:
    cmd = [
        "codex",
        "exec",
        "--dangerously-bypass-approvals-and-sandbox",
        "--skip-git-repo-check",
        "--cd",
        str(workspace),
        "-",
    ]
    return run_cmd(cmd, workspace, timeout=timeout, input_text=prompt)


def parse_raw_blocks(raw_memory: str) -> Dict[str, str]:
    blocks: Dict[str, str] = {}
    current: List[str] = []
    for line in raw_memory.splitlines():
        if line.startswith("qmd://") and current:
            blocks[current[0].strip()] = "\n".join(current).strip()
            current = []
        if line.strip() or current:
            current.append(line)
    if current:
        blocks[current[0].strip()] = "\n".join(current).strip()
    return blocks


def extract_top_memory_excerpt(shortlist: str, raw_memory: str) -> str:
    source_match = re.search(r"Source:\s*(.+)$", shortlist, flags=re.MULTILINE)
    if not source_match:
        return ""
    source = source_match.group(1).strip()
    block = parse_raw_blocks(raw_memory).get(source, "")
    if not block:
        return ""
    lines = [ln for ln in block.splitlines() if ln.startswith("- ") or ln.startswith("## ") or ln.startswith("Title:")]
    return "\n".join(lines[:8]).strip()


def build_prompt(scenario: Scenario, memory_shortlist: Optional[str], memory_excerpt: Optional[str]) -> str:
    memory_section = ""
    if memory_shortlist:
        memory_section = (
            "\nRetrieved Memory (from previous solved session):\n"
            f"{memory_shortlist}\n"
        )
    memory_details = ""
    if memory_excerpt:
        memory_details = (
            "\nTop Memory Details (open/reuse this first):\n"
            f"{memory_excerpt}\n"
        )

    focus = ", ".join(scenario.focus_files) if scenario.focus_files else "target source files"

    return textwrap.dedent(
        f"""
        You are solving a tricky coding bug in a fresh session.
        Scenario: {scenario.title}
        Goal: make all tests pass with minimal, correct changes.

        Required command:
        `python3 -m unittest discover -s tests -p "test_*.py"`

        Constraints:
        - Work only in this repository.
        - Prefer fixing logic in: {focus}
        - Run tests before finishing.
        - In your final message, include `FINAL_STATUS: PASS` only if tests pass.
        {memory_section}
        {memory_details}
        """
    ).strip()


def build_prompt_with_policy(
    scenario: Scenario,
    memory_shortlist: Optional[str],
    memory_excerpt: Optional[str],
    warm_start_policy: bool,
) -> str:
    base = build_prompt(scenario, memory_shortlist, memory_excerpt)
    if not warm_start_policy:
        return base
    policy = textwrap.dedent(
        """
        Warm-start policy (required):
        - Treat the retrieved memory recipe as the first implementation attempt.
        - Start by editing the target file directly; avoid exploratory scans unless first test run fails.
        - Apply only the listed deltas in the listed file(s) before broader investigation.
        - Do not edit comments/docstrings unless tests still fail after code deltas.
        - Do not inspect unrelated files unless tests still fail after applying memory deltas.
        - Prefer a single verification run after patching; avoid extra intermediate test runs.
        - Final response should be a single line: `FINAL_STATUS: PASS` when green.
        - Run tests immediately after the first patch attempt.
        """
    ).strip()
    return f"{base}\n\n{policy}"


def build_memory_block(scenario: Scenario) -> str:
    bullets = "\n".join(f"- {b}" for b in scenario.memory_bullets)
    primary_file = scenario.focus_files[0] if scenario.focus_files else ""
    file_list = ", ".join(scenario.focus_files) if scenario.focus_files else "(none)"
    cue_action = f"edit {primary_file}" if primary_file else "apply prior fix pattern"
    cue_problem = scenario.query[:140]
    verify_cmd = 'python3 -m unittest discover -s tests -p "test_*.py"'
    return (
        f"{scenario.memory_source}\n"
        "memory_kind: bug\n"
        "intent: bug-fix\n"
        "outcome: verified\n"
        "durable_signal: high\n"
        f"cue_problem: {cue_problem}\n"
        f"cue_action: {cue_action}\n"
        f"cue_verify: {verify_cmd}\n"
        f"Title: {scenario.memory_title}\n"
        "Context: Gauntlet memory\n"
        "Score:  92%\n\n"
        "## What Was Done\n"
        f"{bullets}\n"
        "## Retrieval Cues\n"
        f"- Problem cue: {cue_problem}\n"
        f"- Action cue: {cue_action}\n"
        f"- Verify cue: {verify_cmd}\n"
        f"- Files: {file_list}\n"
    )


def build_distractors(scenarios: Sequence[Scenario], current_id: str, limit: int) -> List[str]:
    out: List[str] = []
    for scenario in scenarios:
        if scenario.id == current_id:
            continue
        other_file = scenario.focus_files[0] if scenario.focus_files else "unrelated/file.py"
        cue_problem = f"{scenario.id} unrelated cleanup and formatting only"
        cue_action = f"edit {other_file}"
        verify_cmd = 'python3 -m unittest discover -s tests -p "test_*.py"'
        out.append(
            f"qmd://gauntlet/sessions/2026-02-15-{scenario.id}-unrelated.md:1 #dist-{scenario.id}\n"
            "memory_kind: session\n"
            "intent: implementation\n"
            "outcome: verified\n"
            "durable_signal: low\n"
            f"cue_problem: {cue_problem}\n"
            f"cue_action: {cue_action}\n"
            f"cue_verify: {verify_cmd}\n"
            f"Title: Session: Investigated {scenario.title} but did UI cleanup only\n"
            "Context: Gauntlet memory\n"
            "Score:  76%\n\n"
            "## What Was Done\n"
            "- Updated comments and formatting in unrelated files.\n"
            "## Retrieval Cues\n"
            f"- Problem cue: {cue_problem}\n"
            f"- Action cue: {cue_action}\n"
            f"- Verify cue: {verify_cmd}\n"
            f"- Files: {other_file}\n"
        )
        if len(out) >= limit:
            break
    return out


def memory_aligns_with_focus(shortlist: str, focus_files: Sequence[str]) -> bool:
    if not shortlist or shortlist.strip().lower().startswith("no memory available"):
        return False
    if not focus_files:
        return True
    lower = shortlist.lower()
    return any(path.lower() in lower for path in focus_files)


def apply_reference_fix(workspace: Path, scenario: Scenario) -> None:
    for rep in scenario.reference_replacements:
        rel = rep.get("file", "")
        old = rep.get("old", "")
        new = rep.get("new", "")
        if not rel:
            raise RuntimeError(f"Scenario {scenario.id}: replacement missing file")
        target = workspace / rel
        if not target.exists():
            raise RuntimeError(f"Scenario {scenario.id}: missing target file {rel}")
        text = target.read_text(encoding="utf-8")
        if old not in text:
            raise RuntimeError(f"Scenario {scenario.id}: replacement old text not found in {rel}")
        text = text.replace(old, new)
        target.write_text(text, encoding="utf-8")


def run_track(name: str, scenario: Scenario, workspace: Path, prompt: str, mode: str, agent_timeout: int, test_timeout: int) -> TrackResult:
    start = time.monotonic()
    if mode == "reference":
        apply_reference_fix(workspace, scenario)
        rc, out, err = 0, "Reference solver applied patch.\nFINAL_STATUS: PASS", ""
    else:
        rc, out, err = default_agent_cmd(workspace, timeout=agent_timeout, prompt=prompt)
    passed, tests_output = run_tests(workspace, timeout=test_timeout)
    changed_files, changed_lines = workspace_delta(scenario.template_dir, workspace)
    focus_set = set(scenario.focus_files)
    focus_only = bool(changed_files) and all(path in focus_set for path in changed_files) if focus_set else True
    duration = time.monotonic() - start
    return TrackResult(
        name=name,
        passed=passed,
        duration_seconds=duration,
        return_code=rc,
        tests_output=tests_output,
        agent_stdout=out,
        agent_stderr=err,
        workspace=workspace,
        changed_files=changed_files,
        changed_files_count=len(changed_files),
        changed_lines=changed_lines,
        focus_only=focus_only,
    )


def load_scenarios(path: Path, include: Sequence[str], exclude: Sequence[str]) -> List[Scenario]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    include_set = set(include)
    exclude_set = set(exclude)

    scenarios: List[Scenario] = []
    for row in raw:
        sid = str(row["id"])
        if include_set and sid not in include_set:
            continue
        if sid in exclude_set:
            continue
        scenarios.append(
            Scenario(
                id=sid,
                title=str(row["title"]),
                template_dir=GAUNTLET_DIR / str(row["template_dir"]),
                query=str(row["query"]),
                focus_files=[str(x) for x in row.get("focus_files", [])],
                memory_source=str(row["memory_source"]),
                memory_title=str(row["memory_title"]),
                memory_bullets=[str(x) for x in row.get("memory_bullets", [])],
                reference_replacements=[dict(x) for x in row.get("reference_replacements", [])],
            )
        )

    missing = [s.id for s in scenarios if not s.template_dir.exists()]
    if missing:
        raise RuntimeError(f"Missing scenario templates: {', '.join(missing)}")
    return scenarios


def run_scenario_repeat(
    scenario: Scenario,
    all_scenarios: Sequence[Scenario],
    scenario_position: int,
    repeat_index: int,
    mode: str,
    order_policy: str,
    token_budget: int,
    agent_timeout: int,
    test_timeout: int,
    max_distractors: int,
    include_control_track: bool,
    include_memory_excerpt: bool,
    require_focus_alignment: bool,
    keep_workspaces: bool,
) -> ScenarioRun:
    root_tmp = Path(tempfile.mkdtemp(prefix=f"tr-gauntlet-{scenario.id}-r{repeat_index + 1}-"))
    cold_ws = root_tmp / "cold"
    control_ws = root_tmp / "control"
    warm_ws = root_tmp / "warm"
    shutil.copytree(scenario.template_dir, cold_ws)
    if include_control_track:
        shutil.copytree(scenario.template_dir, control_ws)
    shutil.copytree(scenario.template_dir, warm_ws)

    distractors = build_distractors(all_scenarios, scenario.id, max_distractors)
    raw_memory = "\n\n".join([build_memory_block(scenario)] + distractors)
    control_memory_raw = "\n\n".join(distractors) if distractors else ""
    shortlist = optimize_qmd_output(raw_memory, scenario.query, token_budget=token_budget)
    control_shortlist = optimize_qmd_output(control_memory_raw, scenario.query, token_budget=token_budget) if control_memory_raw else "No memory available."
    warm_aligned = memory_aligns_with_focus(shortlist, scenario.focus_files)
    control_aligned = memory_aligns_with_focus(control_shortlist, scenario.focus_files)

    use_warm_memory = warm_aligned or (not require_focus_alignment)
    use_control_memory = control_aligned or (not require_focus_alignment)

    shortlist_for_prompt = shortlist if use_warm_memory else "No memory available."
    control_shortlist_for_prompt = control_shortlist if use_control_memory else "No memory available."
    excerpt = (
        extract_top_memory_excerpt(shortlist_for_prompt, raw_memory)
        if (include_memory_excerpt and use_warm_memory)
        else ""
    )
    control_excerpt = (
        extract_top_memory_excerpt(control_shortlist_for_prompt, control_memory_raw)
        if (control_memory_raw and include_memory_excerpt and use_control_memory)
        else ""
    )

    cold_prompt = build_prompt(scenario, memory_shortlist=None, memory_excerpt=None)
    control_prompt = build_prompt_with_policy(
        scenario,
        memory_shortlist=control_shortlist_for_prompt,
        memory_excerpt=control_excerpt,
        warm_start_policy=False,
    )
    warm_prompt = build_prompt_with_policy(
        scenario,
        memory_shortlist=shortlist_for_prompt,
        memory_excerpt=excerpt,
        warm_start_policy=use_warm_memory,
    )

    if include_control_track:
        rotations = ["cold-control-warm", "warm-cold-control", "control-warm-cold"]
    else:
        rotations = ["cold-warm", "warm-cold"]

    if order_policy == "cold-first":
        order = rotations[0]
    elif order_policy == "warm-first":
        order = next((x for x in rotations if x.startswith("warm-")), rotations[0])
    else:
        # Balance run-order effects across scenarios and repeats.
        order = rotations[(repeat_index + scenario_position) % len(rotations)]

    try:
        track_specs: Dict[str, Tuple[Path, str]] = {
            "cold": (cold_ws, cold_prompt),
            "warm": (warm_ws, warm_prompt),
        }
        if include_control_track:
            track_specs["control"] = (control_ws, control_prompt)

        executed: Dict[str, TrackResult] = {}
        for name in order.split("-"):
            ws, prompt = track_specs[name]
            executed[name] = run_track(name, scenario, ws, prompt, mode, agent_timeout, test_timeout)

        cold = executed["cold"]
        warm = executed["warm"]
        control = executed.get("control")

        return ScenarioRun(
            scenario_id=scenario.id,
            repeat=repeat_index + 1,
            order=order,
            cold=cold,
            control=control,
            warm=warm,
            shortlist=shortlist_for_prompt,
            control_shortlist=control_shortlist_for_prompt,
            warm_memory_aligned=warm_aligned,
            control_memory_aligned=control_aligned,
        )
    finally:
        if not keep_workspaces:
            shutil.rmtree(root_tmp, ignore_errors=True)


def bootstrap_median_ci(values: Sequence[float], samples: int, seed: int) -> Optional[Tuple[float, float]]:
    vals = [float(v) for v in values]
    if len(vals) < 2:
        return None
    rng = random.Random(seed)
    meds: List[float] = []
    n = len(vals)
    for _ in range(max(100, samples)):
        sample = [vals[rng.randrange(0, n)] for _ in range(n)]
        meds.append(median(sample))
    meds.sort()
    low_idx = int(0.025 * (len(meds) - 1))
    hi_idx = int(0.975 * (len(meds) - 1))
    return (meds[low_idx], meds[hi_idx])


def summarize_scenario_runs(scenario: Scenario, runs: Sequence[ScenarioRun], bootstrap_samples: int, bootstrap_seed: int) -> Dict[str, object]:
    cold_times = [r.cold.duration_seconds for r in runs]
    warm_times = [r.warm.duration_seconds for r in runs]
    deltas = [c - w for c, w in zip(cold_times, warm_times)]
    both_pass = [r.cold.passed and r.warm.passed for r in runs]

    cold_pass_rate = sum(1 for r in runs if r.cold.passed) / max(1, len(runs))
    warm_pass_rate = sum(1 for r in runs if r.warm.passed) / max(1, len(runs))
    control_runs = [r for r in runs if r.control is not None]
    control_pass_rate = (
        sum(1 for r in control_runs if r.control and r.control.passed) / max(1, len(control_runs))
        if control_runs
        else None
    )
    all_pass_rate = sum(1 for ok in both_pass if ok) / max(1, len(runs))
    warm_faster_rate = sum(1 for r in runs if r.warm.duration_seconds < r.cold.duration_seconds) / max(1, len(runs))
    warm_win_rate = sum(
        1 for r in runs if r.warm.passed and r.cold.passed and (r.warm.duration_seconds < r.cold.duration_seconds)
    ) / max(1, len(runs))
    warm_beats_control_rate = (
        sum(
            1
            for r in control_runs
            if r.control
            and r.warm.passed
            and (
                (not r.control.passed)
                or (r.warm.duration_seconds < r.control.duration_seconds)
            )
        )
        / max(1, len(control_runs))
        if control_runs
        else None
    )
    warm_memory_aligned_rate = sum(1 for r in runs if r.warm_memory_aligned) / max(1, len(runs))
    control_memory_aligned_rate = sum(1 for r in runs if r.control_memory_aligned) / max(1, len(runs))
    cold_focus_only_rate = sum(1 for r in runs if r.cold.focus_only) / max(1, len(runs))
    warm_focus_only_rate = sum(1 for r in runs if r.warm.focus_only) / max(1, len(runs))
    control_focus_only_rate = (
        sum(1 for r in control_runs if r.control and r.control.focus_only) / max(1, len(control_runs))
        if control_runs
        else None
    )
    warm_more_precise_than_control_rate = (
        sum(
            1
            for r in control_runs
            if r.control and (r.warm.changed_lines <= r.control.changed_lines)
        )
        / max(1, len(control_runs))
        if control_runs
        else None
    )

    speedups = [
        (r.cold.duration_seconds / r.warm.duration_seconds) if r.warm.duration_seconds > 0 else 0.0
        for r in runs
    ]

    ci = bootstrap_median_ci(deltas, samples=bootstrap_samples, seed=bootstrap_seed)

    return {
        "scenario": scenario.id,
        "title": scenario.title,
        "runs": len(runs),
        "cold_pass_rate": cold_pass_rate,
        "warm_pass_rate": warm_pass_rate,
        "control_pass_rate": control_pass_rate,
        "all_pass_rate": all_pass_rate,
        "warm_faster_rate": warm_faster_rate,
        "warm_win_rate": warm_win_rate,
        "warm_beats_control_rate": warm_beats_control_rate,
        "warm_memory_aligned_rate": warm_memory_aligned_rate,
        "control_memory_aligned_rate": control_memory_aligned_rate,
        "cold_focus_only_rate": cold_focus_only_rate,
        "warm_focus_only_rate": warm_focus_only_rate,
        "control_focus_only_rate": control_focus_only_rate,
        "warm_more_precise_than_control_rate": warm_more_precise_than_control_rate,
        "cold_median_seconds": round(median(cold_times), 2),
        "warm_median_seconds": round(median(warm_times), 2),
        "median_speedup_cold_over_warm": round(median(speedups), 3),
        "median_delta_seconds": round(median(deltas), 2),
        "median_delta_ci95_seconds": (
            [round(ci[0], 2), round(ci[1], 2)] if ci is not None else None
        ),
        "runs_detail": [
            {
                "repeat": r.repeat,
                "order": r.order,
                "cold": {
                    "passed": r.cold.passed,
                    "seconds": round(r.cold.duration_seconds, 2),
                    "return_code": r.cold.return_code,
                    "workspace": str(r.cold.workspace),
                    "changed_files_count": r.cold.changed_files_count,
                    "changed_lines": r.cold.changed_lines,
                    "focus_only": r.cold.focus_only,
                },
                "warm": {
                    "passed": r.warm.passed,
                    "seconds": round(r.warm.duration_seconds, 2),
                    "return_code": r.warm.return_code,
                    "workspace": str(r.warm.workspace),
                    "changed_files_count": r.warm.changed_files_count,
                    "changed_lines": r.warm.changed_lines,
                    "focus_only": r.warm.focus_only,
                },
                "control": (
                    {
                        "passed": r.control.passed,
                        "seconds": round(r.control.duration_seconds, 2),
                        "return_code": r.control.return_code,
                        "workspace": str(r.control.workspace),
                        "changed_files_count": r.control.changed_files_count,
                        "changed_lines": r.control.changed_lines,
                        "focus_only": r.control.focus_only,
                    }
                    if r.control is not None
                    else None
                ),
                "warm_memory_aligned": r.warm_memory_aligned,
                "control_memory_aligned": r.control_memory_aligned,
            }
            for r in runs
        ],
    }


def summarize_suite(summaries: Sequence[Dict[str, object]]) -> Dict[str, object]:
    if not summaries:
        return {
            "scenarios": 0,
            "macro_cold_pass_rate": 0.0,
            "macro_warm_pass_rate": 0.0,
            "macro_warm_win_rate": 0.0,
        }

    return {
        "scenarios": len(summaries),
        "macro_cold_pass_rate": mean(float(s["cold_pass_rate"]) for s in summaries),
        "macro_warm_pass_rate": mean(float(s["warm_pass_rate"]) for s in summaries),
        "macro_control_pass_rate": (
            mean(float(s["control_pass_rate"]) for s in summaries if s["control_pass_rate"] is not None)
            if any(s["control_pass_rate"] is not None for s in summaries)
            else None
        ),
        "macro_all_pass_rate": mean(float(s["all_pass_rate"]) for s in summaries),
        "macro_warm_faster_rate": mean(float(s["warm_faster_rate"]) for s in summaries),
        "macro_warm_win_rate": mean(float(s["warm_win_rate"]) for s in summaries),
        "macro_warm_beats_control_rate": (
            mean(float(s["warm_beats_control_rate"]) for s in summaries if s["warm_beats_control_rate"] is not None)
            if any(s["warm_beats_control_rate"] is not None for s in summaries)
            else None
        ),
        "macro_warm_memory_aligned_rate": mean(float(s["warm_memory_aligned_rate"]) for s in summaries),
        "macro_control_memory_aligned_rate": mean(float(s["control_memory_aligned_rate"]) for s in summaries),
        "macro_cold_focus_only_rate": mean(float(s["cold_focus_only_rate"]) for s in summaries),
        "macro_warm_focus_only_rate": mean(float(s["warm_focus_only_rate"]) for s in summaries),
        "macro_control_focus_only_rate": (
            mean(float(s["control_focus_only_rate"]) for s in summaries if s["control_focus_only_rate"] is not None)
            if any(s["control_focus_only_rate"] is not None for s in summaries)
            else None
        ),
        "macro_warm_more_precise_than_control_rate": (
            mean(float(s["warm_more_precise_than_control_rate"]) for s in summaries if s["warm_more_precise_than_control_rate"] is not None)
            if any(s["warm_more_precise_than_control_rate"] is not None for s in summaries)
            else None
        ),
    }


def print_report(report: Dict[str, object]) -> int:
    cfg = report["config"]
    mode = str(cfg.get("mode", "agent"))
    print("Gauntlet Suite Benchmark")
    print(f"Mode: {cfg['mode']} | repeats: {cfg['repeats']} | scenarios: {len(cfg['scenario_ids'])}")
    print("")

    for row in report["scenario_summaries"]:
        print(f"Scenario: {row['scenario']} ({row['title']})")
        print(f"  cold_pass_rate={row['cold_pass_rate']:.2f} warm_pass_rate={row['warm_pass_rate']:.2f}")
        if row["control_pass_rate"] is not None:
            print(f"  control_pass_rate={row['control_pass_rate']:.2f}")
        print(f"  warm_faster_rate={row['warm_faster_rate']:.2f} warm_win_rate={row['warm_win_rate']:.2f}")
        if row["warm_beats_control_rate"] is not None:
            print(f"  warm_beats_control_rate={row['warm_beats_control_rate']:.2f}")
        print(f"  warm_memory_aligned_rate={row['warm_memory_aligned_rate']:.2f} control_memory_aligned_rate={row['control_memory_aligned_rate']:.2f}")
        print(f"  cold_focus_only_rate={row['cold_focus_only_rate']:.2f} warm_focus_only_rate={row['warm_focus_only_rate']:.2f}")
        if row["control_focus_only_rate"] is not None:
            print(f"  control_focus_only_rate={row['control_focus_only_rate']:.2f}")
        if row["warm_more_precise_than_control_rate"] is not None:
            print(f"  warm_more_precise_than_control_rate={row['warm_more_precise_than_control_rate']:.2f}")
        print(f"  median_cold={row['cold_median_seconds']}s median_warm={row['warm_median_seconds']}s")
        print(f"  median_delta(cold-warm)={row['median_delta_seconds']}s ci95={row['median_delta_ci95_seconds']}")
        print("")

    s = report["suite_summary"]
    print("Suite Summary")
    print(f"  macro_cold_pass_rate={s['macro_cold_pass_rate']:.2f}")
    print(f"  macro_warm_pass_rate={s['macro_warm_pass_rate']:.2f}")
    if s["macro_control_pass_rate"] is not None:
        print(f"  macro_control_pass_rate={s['macro_control_pass_rate']:.2f}")
    print(f"  macro_all_pass_rate={s['macro_all_pass_rate']:.2f}")
    print(f"  macro_warm_faster_rate={s['macro_warm_faster_rate']:.2f}")
    print(f"  macro_warm_win_rate={s['macro_warm_win_rate']:.2f}")
    if s["macro_warm_beats_control_rate"] is not None:
        print(f"  macro_warm_beats_control_rate={s['macro_warm_beats_control_rate']:.2f}")
    print(f"  macro_warm_memory_aligned_rate={s['macro_warm_memory_aligned_rate']:.2f}")
    print(f"  macro_control_memory_aligned_rate={s['macro_control_memory_aligned_rate']:.2f}")
    print(f"  macro_cold_focus_only_rate={s['macro_cold_focus_only_rate']:.2f}")
    print(f"  macro_warm_focus_only_rate={s['macro_warm_focus_only_rate']:.2f}")
    if s["macro_control_focus_only_rate"] is not None:
        print(f"  macro_control_focus_only_rate={s['macro_control_focus_only_rate']:.2f}")
    if s["macro_warm_more_precise_than_control_rate"] is not None:
        print(f"  macro_warm_more_precise_than_control_rate={s['macro_warm_more_precise_than_control_rate']:.2f}")

    gates = report.get("gates") or {}
    pass_gate = bool(gates.get("pass_gate", False))
    speed_gate = bool(gates.get("speed_gate", False))
    control_gate = bool(gates.get("control_gate", False))
    precision_gate = bool(gates.get("precision_gate", False))
    if pass_gate and speed_gate and control_gate and precision_gate:
        if mode == "reference":
            print("\nRESULT: PASS (reference harness checks passed)")
        else:
            print("\nRESULT: PASS (durable warm-memory advantage under suite criteria)")
        return 0

    if mode == "reference":
        print("\nRESULT: FAIL (reference harness checks failed)")
    else:
        print("\nRESULT: FAIL (suite criteria not yet met)")
    print(f"  Gates -> pass:{pass_gate} speed:{speed_gate} control:{control_gate} precision:{precision_gate}")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Run durable multi-scenario gauntlet suite.")
    parser.add_argument("--scenarios-file", default=str(DEFAULT_SCENARIOS_FILE))
    parser.add_argument("--scenario", action="append", default=[], help="Run only specific scenario id (repeatable)")
    parser.add_argument("--exclude-scenario", action="append", default=[])
    parser.add_argument("--mode", choices=["agent", "reference"], default="agent")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--order-policy", choices=["alternate", "cold-first", "warm-first"], default="alternate")
    parser.add_argument("--token-budget", type=int, default=900)
    parser.add_argument("--agent-timeout-seconds", type=int, default=420)
    parser.add_argument("--test-timeout-seconds", type=int, default=90)
    parser.add_argument("--max-distractors", type=int, default=2)
    parser.add_argument("--no-control-track", action="store_true", help="Disable control run with irrelevant memory.")
    parser.add_argument("--include-memory-excerpt", action="store_true", help="Also include raw top-memory excerpt in prompts.")
    parser.add_argument("--no-focus-alignment-gate", action="store_true", help="Inject memory even if no overlap with scenario focus files.")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--keep-workspaces", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    scenarios = load_scenarios(
        Path(args.scenarios_file),
        include=args.scenario,
        exclude=args.exclude_scenario,
    )
    if not scenarios:
        raise SystemExit("No scenarios selected.")

    all_runs: Dict[str, List[ScenarioRun]] = {s.id: [] for s in scenarios}
    for idx, scenario in enumerate(scenarios):
        for repeat in range(max(1, args.repeats)):
            run = run_scenario_repeat(
                scenario=scenario,
                all_scenarios=scenarios,
                scenario_position=idx,
                repeat_index=repeat,
                mode=args.mode,
                order_policy=args.order_policy,
                token_budget=args.token_budget,
                agent_timeout=args.agent_timeout_seconds,
                test_timeout=args.test_timeout_seconds,
                max_distractors=args.max_distractors,
                include_control_track=(not args.no_control_track),
                include_memory_excerpt=args.include_memory_excerpt,
                require_focus_alignment=(not args.no_focus_alignment_gate),
                keep_workspaces=args.keep_workspaces,
            )
            all_runs[scenario.id].append(run)

    scenario_summaries = [
        summarize_scenario_runs(
            scenario=s,
            runs=all_runs[s.id],
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        )
        for s in scenarios
    ]

    report = {
        "config": {
            "mode": args.mode,
            "repeats": args.repeats,
            "order_policy": args.order_policy,
            "token_budget": args.token_budget,
            "agent_timeout_seconds": args.agent_timeout_seconds,
            "test_timeout_seconds": args.test_timeout_seconds,
            "scenario_ids": [s.id for s in scenarios],
            "control_track": (not args.no_control_track),
            "include_memory_excerpt": args.include_memory_excerpt,
            "focus_alignment_gate": (not args.no_focus_alignment_gate),
        },
        "scenario_summaries": scenario_summaries,
        "suite_summary": summarize_suite(scenario_summaries),
    }
    s = report["suite_summary"]
    control_metric = s.get("macro_warm_beats_control_rate")
    precision_metric = s.get("macro_warm_more_precise_than_control_rate")
    if args.mode == "reference":
        report["gates"] = {
            "pass_gate": s["macro_all_pass_rate"] >= 1.0,
            "speed_gate": True,
            "control_gate": True,
            "precision_gate": True,
        }
    else:
        report["gates"] = {
            "pass_gate": s["macro_all_pass_rate"] >= 0.8,
            "speed_gate": s["macro_warm_win_rate"] >= 0.6,
            "control_gate": (control_metric is None) or (control_metric >= 0.55),
            "precision_gate": (precision_metric is None) or (precision_metric >= 0.55),
        }
    report["passed"] = all(report["gates"].values())

    if args.json:
        print(json.dumps(report, indent=2))
        return 0 if report["passed"] else 1
    return print_report(report)


if __name__ == "__main__":
    raise SystemExit(main())
