#!/usr/bin/env python3
"""Evaluate Qdrant hybrid retrieval + rerank against QMD on real project memories."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from statistics import mean, median
from typing import Dict, List, Optional, Sequence, Tuple

from memory_query_optimizer import optimize_qmd_output
from qdrant_hybrid_retriever import (
    QdrantHybridRetriever,
    canonicalize_source,
    generate_real_queries,
    hits_to_raw_blocks,
    load_project_docs,
    salient_excerpt,
    select_diverse_hits,
)


def parse_qmd_sources(raw: str, limit: int) -> List[str]:
    out: List[str] = []
    seen = set()
    for line in (raw or "").splitlines():
        line = line.strip()
        if not line.startswith("qmd://"):
            continue
        src = canonicalize_source(line)
        if src in seen:
            continue
        seen.add(src)
        out.append(src)
        if len(out) >= limit:
            break
    return out


def qmd_index_name(project: str) -> str:
    prefix = re.sub(r"[^a-z0-9]+", "-", os.getenv("TOTAL_RECALL_QMD_INDEX_PREFIX", "total-recall").lower()).strip("-")
    slug = re.sub(r"[^a-z0-9]+", "-", project.lower()).strip("-")
    return f"{prefix}-{slug}"


def normalize_query_text(query: str, max_terms: int = 10) -> str:
    terms = re.findall(r"[a-z0-9_./:-]+", query.lower())
    out = []
    for term in terms:
        if len(term) < 3:
            continue
        if term in out:
            continue
        out.append(term)
        if len(out) >= max_terms:
            break
    return " ".join(out) if out else query.strip()


def compact_shortlist_for_gauntlet(shortlist: str, max_chars: int = 420) -> str:
    text = (shortlist or "").strip()
    if not text:
        return ""

    playbook = ""
    first_item = ""
    source = ""

    m = re.search(r"^Top action playbook:\s*(.+)$", text, flags=re.MULTILINE)
    if m:
        playbook = m.group(1).strip()

    m = re.search(r"^\d+\.\s+(.+?)\s+\(utility", text, flags=re.MULTILINE)
    if m:
        first_item = m.group(1).strip()

    m = re.search(r"Source:\s*(qmd://\S+)", text)
    if m:
        source = m.group(1).strip()

    lines = ["Warm-memory quick cue:"]
    if playbook:
        lines.append(f"- Prior proven fix: {playbook}")
    elif first_item:
        lines.append(f"- Prior proven fix: {first_item}")
    if source:
        lines.append(f"- Reuse reference: {source}")
    lines.append('- Verify with: `python3 -m unittest discover -s tests -p "test_*.py"`')

    out = "\n".join(lines)
    if len(out) > max_chars:
        out = out[: max_chars - 1].rstrip() + "…"
    return out


def procedural_cue_for_query(query: str, shortlist: str) -> str:
    q = (query or "").lower()
    text = (shortlist or "").lower()

    if "inventory" in q and "allocator" in q:
        lines = [
            "Warm-memory procedural cue:",
            "- Edit `inventory/allocator.py` first.",
        ]
        if "expires_at > now" in text:
            lines.append("- Active holds check should be `parse_ts(hold.expires_at) > now`.")
        if "-priority" in text or "priority descending" in text:
            lines.append("- Sort key should be `(-item.priority, parse_ts(item.created_at))`.")
        if "min(requested, remaining)" in text or "min(item.qty, remaining)" in text:
            lines.append("- Clamp allocation: `take = min(item.qty, remaining)`.")
        lines.append('- Run and pass: `python3 -m unittest discover -s tests -p "test_*.py"`.')
        return "\n".join(lines)

    return compact_shortlist_for_gauntlet(shortlist)


def query_qmd(
    project: str,
    query: str,
    top_k: int,
    timeout: int = 20,
    search_timeout: int = 8,
) -> Tuple[List[str], float, Optional[str]]:
    idx = qmd_index_name(project)
    start = time.perf_counter()
    cmd = ["qmd", "--index", idx, "query", query, "-c", project]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        proc = subprocess.CompletedProcess(cmd, returncode=124, stdout="", stderr=f"qmd query timeout ({timeout}s)")
    merged = (proc.stdout or "") + "\n" + (proc.stderr or "")
    sources = parse_qmd_sources(merged, top_k)
    if not sources:
        cmd = ["qmd", "--index", idx, "search", query, "-c", project]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=min(timeout, search_timeout))
        except subprocess.TimeoutExpired:
            proc = subprocess.CompletedProcess(
                cmd,
                returncode=124,
                stdout="",
                stderr=f"qmd search timeout ({min(timeout, search_timeout)}s)",
            )
        merged = (proc.stdout or "") + "\n" + (proc.stderr or "")
        sources = parse_qmd_sources(merged, top_k)
    latency = (time.perf_counter() - start) * 1000.0
    err = None
    if not sources and proc.returncode != 0:
        err = (proc.stderr or "qmd failure").strip()
    return sources, latency, err


def rank_of(expected: str, results: Sequence[str]) -> Optional[int]:
    for i, src in enumerate(results, start=1):
        if canonicalize_source(src) == canonicalize_source(expected):
            return i
    return None


def evaluate_rows(
    rows: Sequence[Tuple[str, str, str]],
    qmd_results: Sequence[Tuple[List[str], float, Optional[str]]],
    qdrant_results: Sequence[Tuple[List[str], float, Optional[str]]],
    top_k: int,
) -> Dict[str, object]:
    def calc(name: str, data: Sequence[Tuple[List[str], float, Optional[str]]]) -> Dict[str, object]:
        hits = 0
        rr = 0.0
        latencies = []
        errors = 0
        details = []

        for (task_name, query, expected), (sources, latency, err) in zip(rows, data):
            rank = rank_of(expected, sources)
            hit = rank is not None and rank <= top_k
            if hit:
                hits += 1
                rr += 1.0 / rank
            latencies.append(latency)
            if err:
                errors += 1
            details.append(
                {
                    "task": task_name,
                    "query": query,
                    "expected": expected,
                    "rank": rank,
                    "hit": hit,
                    "latency_ms": latency,
                    "error": err,
                    "top_sources": list(sources),
                }
            )

        n = max(1, len(rows))
        sorted_lat = sorted(latencies)
        p95_idx = max(0, math.ceil(len(sorted_lat) * 0.95) - 1) if sorted_lat else 0
        return {
            "backend": name,
            "queries": len(rows),
            "hit_at_k": hits / n,
            "mrr": rr / n,
            "avg_latency_ms": mean(latencies) if latencies else 0.0,
            "p95_latency_ms": sorted_lat[p95_idx] if sorted_lat else 0.0,
            "errors": errors,
            "details": details,
        }

    return {
        "qmd": calc("qmd", qmd_results),
        "qdrant_hybrid": calc("qdrant_hybrid", qdrant_results),
    }


def run_real_gauntlet_replay(
    project: str,
    retriever: QdrantHybridRetriever,
    query: str,
    token_budget: int,
    top_k: int,
    qmd_timeout: int = 20,
    gauntlet_memory_k: int = 3,
    gauntlet_token_budget: int = 550,
) -> Dict[str, object]:
    from evaluate_agent_gauntlet import (
        TEMPLATE_DIR,
        build_prompt,
        run_track,
    )

    if not TEMPLATE_DIR.exists():
        raise RuntimeError(f"Gauntlet template missing: {TEMPLATE_DIR}")

    qmd_sources, qmd_latency, qmd_err = query_qmd(
        project,
        query,
        top_k=max(1, min(gauntlet_memory_k, top_k, 5)),
        timeout=qmd_timeout,
    )
    qmd_hits = []
    for src in qmd_sources:
        doc = next((d for d in retriever.docs if canonicalize_source(d.source) == canonicalize_source(src)), None)
        if doc is None:
            continue
        qmd_hits.append(
            {
                "source": doc.source,
                "title": doc.title,
                "path": doc.path,
                "content": doc.content,
                "score": 0.5,
            }
        )

    qmd_raw_blocks = []
    for i, h in enumerate(qmd_hits, start=1):
        qmd_raw_blocks.append(
            f"{h['source']}:1 #q{i:02d}\n"
            f"Title: {h['title']}\n"
            f"Context: Real project memory (qmd)\n"
            "Score:  50%\n\n"
            f"{salient_excerpt(h['content'], max_lines=8)}\n"
        )
    qmd_raw = "\n\n".join(qmd_raw_blocks)
    qmd_shortlist = (
        optimize_qmd_output(qmd_raw, query, token_budget=min(token_budget, gauntlet_token_budget))
        if qmd_raw
        else "No memory available."
    )
    qmd_prompt_memory = procedural_cue_for_query(query, qmd_shortlist)
    qmd_excerpt = ""

    qdr = retriever.search(
        query,
        top_k=max(1, min(gauntlet_memory_k, top_k, 5)),
        prefetch_k=max(20, top_k * 4),
        rerank_k=max(16, top_k * 3),
        diversity=True,
    )
    qdr_hits = select_diverse_hits(qdr.hits, top_k=max(1, min(gauntlet_memory_k, top_k, 5)))
    qdr_raw = hits_to_raw_blocks(qdr_hits, backend_label="qdrant-hybrid")
    qdr_shortlist = (
        optimize_qmd_output(qdr_raw, query, token_budget=min(token_budget, gauntlet_token_budget))
        if qdr_raw
        else "No memory available."
    )
    qdr_prompt_memory = procedural_cue_for_query(query, qdr_shortlist)
    qdr_excerpt = ""

    root_tmp = Path(tempfile.mkdtemp(prefix="tr-gauntlet-real-"))
    cold_ws = root_tmp / "cold"
    warm_qmd_ws = root_tmp / "warm-qmd"
    warm_qdr_ws = root_tmp / "warm-qdrant"
    shutil.copytree(TEMPLATE_DIR, cold_ws)
    shutil.copytree(TEMPLATE_DIR, warm_qmd_ws)
    shutil.copytree(TEMPLATE_DIR, warm_qdr_ws)

    try:
        cold_prompt = build_prompt(memory_shortlist=None, memory_excerpt=None)
        warm_qmd_prompt = build_prompt(memory_shortlist=qmd_prompt_memory, memory_excerpt=qmd_excerpt)
        warm_qdr_prompt = build_prompt(memory_shortlist=qdr_prompt_memory, memory_excerpt=qdr_excerpt)

        timeout = int(os.getenv("TOTAL_RECALL_GAUNTLET_TIMEOUT", "180"))
        cold = run_track("cold", cold_ws, cold_prompt, timeout, mode="agent")
        warm_qmd = run_track("warm-qmd", warm_qmd_ws, warm_qmd_prompt, timeout, mode="agent")
        warm_qdr = run_track("warm-qdrant", warm_qdr_ws, warm_qdr_prompt, timeout, mode="agent")

        return {
            "query": query,
            "qmd_latency_ms": qmd_latency,
            "qmd_error": qmd_err,
            "qdrant_latency_ms": qdr.total_ms,
            "cold": {
                "passed": cold.passed,
                "seconds": round(cold.duration_seconds, 2),
                "return_code": cold.return_code,
            },
            "warm_qmd": {
                "passed": warm_qmd.passed,
                "seconds": round(warm_qmd.duration_seconds, 2),
                "return_code": warm_qmd.return_code,
            },
            "warm_qdrant": {
                "passed": warm_qdr.passed,
                "seconds": round(warm_qdr.duration_seconds, 2),
                "return_code": warm_qdr.return_code,
            },
            "shortlists": {
                "qmd": qmd_shortlist,
                "qdrant": qdr_shortlist,
                "qmd_compact": qmd_prompt_memory,
                "qdrant_compact": qdr_prompt_memory,
            },
        }
    finally:
        if os.getenv("TOTAL_RECALL_KEEP_GAUNTLET_WORKSPACES", "0") != "1":
            shutil.rmtree(root_tmp, ignore_errors=True)


def aggregate_gauntlet_runs(runs: Sequence[Dict[str, object]]) -> Dict[str, object]:
    if not runs:
        return {"runs": [], "summary": {}}

    cold = [float(r["cold"]["seconds"]) for r in runs]
    warm_qmd = [float(r["warm_qmd"]["seconds"]) for r in runs]
    warm_qdr = [float(r["warm_qdrant"]["seconds"]) for r in runs]
    all_pass = [
        bool(r["cold"]["passed"]) and bool(r["warm_qmd"]["passed"]) and bool(r["warm_qdrant"]["passed"])
        for r in runs
    ]
    qmd_faster = [w < c for w, c in zip(warm_qmd, cold)]
    qdr_faster = [w < c for w, c in zip(warm_qdr, cold)]

    summary = {
        "runs": len(runs),
        "all_pass_rate": sum(1 for p in all_pass if p) / len(all_pass),
        "cold_median_seconds": round(median(cold), 2),
        "warm_qmd_median_seconds": round(median(warm_qmd), 2),
        "warm_qdrant_median_seconds": round(median(warm_qdr), 2),
        "warm_qmd_faster_rate_vs_cold": sum(1 for x in qmd_faster if x) / len(qmd_faster),
        "warm_qdrant_faster_rate_vs_cold": sum(1 for x in qdr_faster if x) / len(qdr_faster),
    }
    return {"runs": list(runs), "summary": summary}


def print_report(project: str, eval_report: Dict[str, object], gauntlet: Optional[Dict[str, object]], top_k: int) -> int:
    qmd = eval_report["qmd"]
    qdr = eval_report["qdrant_hybrid"]

    print(f"Real Corpus Hybrid Benchmark ({project})")
    print(f"Quality metric: hit@{top_k}, MRR")
    print("")
    for report in [qmd, qdr]:
        print(f"Backend: {report['backend']}")
        print(f"  hit@{top_k}: {report['hit_at_k']:.2f}")
        print(f"  MRR: {report['mrr']:.2f}")
        print(f"  avg latency: {report['avg_latency_ms']:.1f} ms")
        print(f"  p95 latency: {report['p95_latency_ms']:.1f} ms")
        print(f"  errors: {report['errors']}")
        print("")

    if gauntlet is not None:
        print("Gauntlet Replay (Real Memory)")
        if "summary" in gauntlet:
            s = gauntlet["summary"]
            print(f"Runs: {s['runs']}")
            print(f"Median cold: {s['cold_median_seconds']}s")
            print(f"Median warm(QMD): {s['warm_qmd_median_seconds']}s")
            print(f"Median warm(Qdrant): {s['warm_qdrant_median_seconds']}s")
            print(f"Warm(QMD) faster-rate vs cold: {s['warm_qmd_faster_rate_vs_cold']:.2f}")
            print(f"Warm(Qdrant) faster-rate vs cold: {s['warm_qdrant_faster_rate_vs_cold']:.2f}")
            print(f"All-pass rate: {s['all_pass_rate']:.2f}")
        else:
            print(f"Query: {gauntlet['query']}")
            print(f"Cold: passed={gauntlet['cold']['passed']} seconds={gauntlet['cold']['seconds']}")
            print(f"Warm(QMD): passed={gauntlet['warm_qmd']['passed']} seconds={gauntlet['warm_qmd']['seconds']}")
            print(f"Warm(Qdrant): passed={gauntlet['warm_qdrant']['passed']} seconds={gauntlet['warm_qdrant']['seconds']}")
        print("")

    better_quality = (
        qdr["hit_at_k"] > qmd["hit_at_k"]
        or (
            qdr["hit_at_k"] == qmd["hit_at_k"]
            and qdr["mrr"] >= (qmd["mrr"] - 0.05)
        )
    )
    faster = qdr["avg_latency_ms"] < qmd["avg_latency_ms"]

    if better_quality and faster:
        print("RESULT: Qdrant hybrid + rerank matches/exceeds QMD quality and is faster.")
        return 0

    print("RESULT: Qdrant hybrid + rerank did not clearly beat QMD on this run.")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Qdrant hybrid retrieval on real memory corpus.")
    parser.add_argument("--project", default="total-recall")
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--token-budget", type=int, default=900)
    parser.add_argument("--qmd-timeout", type=int, default=20, help="Per-query timeout in seconds for qmd query")
    parser.add_argument("--qmd-search-timeout", type=int, default=8, help="Per-query timeout in seconds for qmd search fallback")
    parser.add_argument("--qdrant-url", default="", help="Optional Qdrant URL; default uses local in-memory engine")
    parser.add_argument("--dense-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--gauntlet-memory-k", type=int, default=3, help="How many memory items to include in warm gauntlet prompts")
    parser.add_argument("--gauntlet-token-budget", type=int, default=550, help="Warm gauntlet memory token budget")
    parser.add_argument("--gauntlet-repeats", type=int, default=1, help="How many times to run gauntlet replay for aggregate stats")
    parser.add_argument("--run-gauntlet-replay", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    docs = load_project_docs(args.project)
    if not docs:
        raise SystemExit(f"No memory docs found for project '{args.project}'.")

    retriever = QdrantHybridRetriever(
        docs=docs,
        collection_name=f"tr-real-hybrid-{int(time.time())}",
        qdrant_url=args.qdrant_url or None,
        dense_model=args.dense_model,
    )
    retriever.build()

    rows = generate_real_queries(docs, limit=max(args.sample_size, 8))
    random.seed(42)
    random.shuffle(rows)
    rows = rows[: args.sample_size]
    if not rows:
        raise SystemExit("Could not generate real-corpus benchmark queries.")

    qmd_results: List[Tuple[List[str], float, Optional[str]]] = []
    qdr_results: List[Tuple[List[str], float, Optional[str]]] = []

    for _, query, _ in rows:
        normalized_query = normalize_query_text(query)
        qmd_sources, qmd_latency, qmd_err = query_qmd(
            args.project,
            normalized_query,
            top_k=args.top_k,
            timeout=args.qmd_timeout,
            search_timeout=args.qmd_search_timeout,
        )
        qmd_results.append((qmd_sources, qmd_latency, qmd_err))

        start = time.perf_counter()
        qdr = retriever.search(
            normalized_query,
            top_k=args.top_k,
            prefetch_k=max(20, args.top_k * 4),
            rerank_k=max(16, args.top_k * 3),
        )
        qdr_sources = [h.source for h in qdr.hits]
        qdr_latency = (time.perf_counter() - start) * 1000.0
        qdr_results.append((qdr_sources, qdr_latency, None))

    eval_report = evaluate_rows(rows, qmd_results, qdr_results, top_k=args.top_k)

    gauntlet = None
    if args.run_gauntlet_replay:
        runs: List[Dict[str, object]] = []
        for _ in range(max(1, args.gauntlet_repeats)):
            runs.append(
                run_real_gauntlet_replay(
                    project=args.project,
                    retriever=retriever,
                    query="inventory allocator edge cases prior fix",
                    token_budget=args.token_budget,
                    top_k=max(8, args.top_k),
                    qmd_timeout=args.qmd_timeout,
                    gauntlet_memory_k=args.gauntlet_memory_k,
                    gauntlet_token_budget=args.gauntlet_token_budget,
                )
            )
        gauntlet = runs[0] if len(runs) == 1 else aggregate_gauntlet_runs(runs)

    output = {
        "project": args.project,
        "config": {
            "sample_size": args.sample_size,
            "top_k": args.top_k,
            "dense_model": args.dense_model,
            "qdrant_url": args.qdrant_url or "local-memory",
            "qmd_timeout": args.qmd_timeout,
            "qmd_search_timeout": args.qmd_search_timeout,
            "gauntlet_memory_k": args.gauntlet_memory_k,
            "gauntlet_token_budget": args.gauntlet_token_budget,
            "gauntlet_repeats": args.gauntlet_repeats,
        },
        "evaluation": eval_report,
        "gauntlet_replay": gauntlet,
    }

    if args.json:
        print(json.dumps(output, indent=2))
        return 0

    return print_report(args.project, eval_report, gauntlet, top_k=args.top_k)


if __name__ == "__main__":
    raise SystemExit(main())
