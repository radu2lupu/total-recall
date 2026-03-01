#!/usr/bin/env python3
"""Qdrant-backed memory query helper for Total Recall."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

from qdrant_hybrid_retriever import QdrantHybridRetriever, hits_to_raw_blocks, load_project_docs

try:
    from memory_query_optimizer import optimize_qmd_output as optimize_memory_output
except Exception:  # pragma: no cover - fallback if optimizer missing
    def optimize_memory_output(raw: str, query: str, token_budget: int = 900) -> str:
        return (raw or "").strip()


def slugify(value: str) -> str:
    import re

    slug = re.sub(r"[^a-z0-9]+", "-", (value or "").lower()).strip("-")
    return re.sub(r"-+", "-", slug) or "memory"


def query_project_memories(
    *,
    project: str,
    query: str,
    shared_root: Optional[str] = None,
    token_budget: int = 900,
    top_k: int = 8,
    prefetch_k: int = 40,
    rerank_k: int = 24,
    qdrant_url: Optional[str] = None,
    diversity: bool = True,
) -> tuple[str, dict]:
    shared_root = shared_root or os.path.expanduser(
        os.getenv("TOTAL_RECALL_SHARED_ROOT", "~/.ai-memory/knowledge")
    )
    docs = load_project_docs(project, shared_root=shared_root)
    if not docs:
        return "No memory available.", {"docs": 0, "latency_ms": 0.0}

    collection = f"tr-{slugify(project)}"
    retriever = QdrantHybridRetriever(
        docs=docs,
        collection_name=collection,
        qdrant_url=qdrant_url or None,
        dense_model=os.getenv("TOTAL_RECALL_QDRANT_DENSE_MODEL", "BAAI/bge-small-en-v1.5"),
    )
    retriever.build()
    result = retriever.search(
        query,
        top_k=max(1, top_k),
        prefetch_k=max(prefetch_k, top_k * 2),
        rerank_k=max(rerank_k, top_k * 2),
        diversity=diversity,
    )
    raw_blocks = hits_to_raw_blocks(result.hits, backend_label="qdrant-hybrid")
    if not raw_blocks.strip():
        return "No memory available.", {"docs": len(docs), "latency_ms": result.total_ms}
    optimized = optimize_memory_output(raw_blocks, query, token_budget=token_budget)
    return optimized, {
        "docs": len(docs),
        "hits": len(result.hits),
        "latency_ms": result.total_ms,
        "retrieval_ms": result.retrieval_ms,
        "rerank_ms": result.rerank_ms,
        "backend": "qdrant_hybrid",
        "collection": collection,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Query Total Recall memories via Qdrant hybrid retrieval.")
    parser.add_argument("--project", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--shared-root", default=os.getenv("TOTAL_RECALL_SHARED_ROOT", "~/.ai-memory/knowledge"))
    parser.add_argument("--token-budget", type=int, default=int(os.getenv("TOTAL_RECALL_QUERY_TOKEN_BUDGET", "900")))
    parser.add_argument("--top-k", type=int, default=int(os.getenv("TOTAL_RECALL_QDRANT_TOP_K", "8")))
    parser.add_argument("--prefetch-k", type=int, default=int(os.getenv("TOTAL_RECALL_QDRANT_PREFETCH_K", "40")))
    parser.add_argument("--rerank-k", type=int, default=int(os.getenv("TOTAL_RECALL_QDRANT_RERANK_K", "24")))
    parser.add_argument("--qdrant-url", default=os.getenv("TOTAL_RECALL_QDRANT_URL", ""))
    parser.add_argument("--json-metrics", action="store_true")
    args = parser.parse_args()

    result, metrics = query_project_memories(
        project=args.project,
        query=args.query,
        shared_root=str(Path(args.shared_root).expanduser()),
        token_budget=args.token_budget,
        top_k=args.top_k,
        prefetch_k=args.prefetch_k,
        rerank_k=args.rerank_k,
        qdrant_url=args.qdrant_url or None,
    )
    print(result)
    if args.json_metrics:
        import json

        print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
