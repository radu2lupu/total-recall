#!/usr/bin/env python3
"""Compare Qdrant vs QMD retrieval quality and latency on labeled memory scenarios."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Tuple

from evaluate_agent_replay import scenario_data

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams
except Exception:  # pragma: no cover - optional dependency
    QdrantClient = None
    Distance = None
    PointStruct = None
    VectorParams = None

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_QDRANT_URL = "http://127.0.0.1:6333"


@dataclass
class CorpusDoc:
    source: str
    canonical_source: str
    collection: str
    rel_path: str
    title: str
    body: str


@dataclass
class BackendQueryResult:
    sources: List[str]
    latency_ms: float
    error: Optional[str] = None


def canonicalize_source(source: str) -> str:
    m = re.match(r"^(qmd://[^\s:]+(?:/[^\s:]+)+)", source.strip())
    return m.group(1) if m else source.strip()


def parse_qmd_block(block: str) -> CorpusDoc:
    lines = block.strip().splitlines()
    if not lines:
        raise ValueError("empty qmd block")

    source = lines[0].strip()
    title = ""
    for line in lines[1:]:
        if line.startswith("Title: "):
            title = line[len("Title: ") :].strip()
            break

    parts = block.split("\n\n", 1)
    body = parts[1].strip() if len(parts) > 1 else ""

    m = re.match(r"^qmd://([^/]+)/(.+?):\d+", source)
    if not m:
        raise ValueError(f"invalid source format: {source}")

    collection = m.group(1)
    rel_path = m.group(2)
    return CorpusDoc(
        source=source,
        canonical_source=canonicalize_source(source),
        collection=collection,
        rel_path=rel_path,
        title=title,
        body=body,
    )


def build_corpus() -> Tuple[List[Dict[str, str]], List[CorpusDoc]]:
    scenarios = scenario_data()
    docs_by_source: Dict[str, CorpusDoc] = {}

    for scenario in scenarios:
        for block in [scenario.relevant_memory] + scenario.distractors:
            doc = parse_qmd_block(block)
            docs_by_source.setdefault(doc.source, doc)

    query_rows: List[Dict[str, str]] = []
    for scenario in scenarios:
        query_rows.append(
            {
                "name": scenario.name,
                "query": scenario.query,
                "expected_source": canonicalize_source(scenario.relevant_source),
            }
        )

    return query_rows, list(docs_by_source.values())


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9_./:-]+", text.lower())


class HashingEmbedder:
    def __init__(self, dim: int = 384):
        self.dim = dim
        self.idf: Dict[str, float] = {}

    def fit(self, texts: List[str]) -> None:
        doc_freq: Dict[str, int] = {}
        n_docs = max(1, len(texts))
        for text in texts:
            seen = set(tokenize(text))
            for token in seen:
                doc_freq[token] = doc_freq.get(token, 0) + 1

        self.idf = {
            token: 1.0 + math.log((1.0 + n_docs) / (1.0 + df))
            for token, df in doc_freq.items()
        }

    def encode(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        tokens = tokenize(text)
        if not tokens:
            return vec

        tf: Dict[str, int] = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1

        for token, count in tf.items():
            digest = hashlib.md5(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if (digest[4] & 1) == 0 else -1.0
            weight = (1.0 + math.log(count)) * self.idf.get(token, 1.0)
            vec[idx] += sign * weight

        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec


def http_request(url: str, method: str = "GET", payload: Optional[dict] = None, timeout: float = 20.0) -> dict:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url=url, data=data, method=method, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        if not body:
            return {}
        return json.loads(body)


def qdrant_ready(base_url: str) -> bool:
    try:
        req = urllib.request.Request(url=base_url.rstrip("/") + "/readyz", method="GET")
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            return 200 <= getattr(resp, "status", 200) < 300
    except Exception:
        return False


def start_qdrant_docker(container_name: str, image: str = "qdrant/qdrant:v1.13.6") -> None:
    subprocess.run(["docker", "rm", "-f", container_name], capture_output=True, text=True)
    run = subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--rm",
            "--name",
            container_name,
            "-p",
            "6333:6333",
            image,
        ],
        capture_output=True,
        text=True,
    )
    if run.returncode != 0:
        raise RuntimeError(f"failed to start qdrant docker: {run.stderr.strip()}")


def stop_qdrant_docker(container_name: str) -> None:
    subprocess.run(["docker", "rm", "-f", container_name], capture_output=True, text=True)


def ensure_qmd_corpus(index_name: str, collection: str, docs: List[CorpusDoc]) -> Path:
    tmp_dir = Path(tempfile.mkdtemp(prefix="tr-qmd-corpus-"))

    for doc in docs:
        rel = doc.rel_path
        if rel.startswith(collection + "/"):
            rel = rel[len(collection) + 1 :]
        file_path = tmp_dir / rel
        file_path.parent.mkdir(parents=True, exist_ok=True)
        content = f"# {doc.title}\n\n{doc.body}\n"
        file_path.write_text(content, encoding="utf-8")

    def run_qmd(*args: str, timeout: int = 120, required: bool = True) -> None:
        cmd = ["qmd", "--index", index_name, *args]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0:
            if required:
                raise RuntimeError(f"qmd command failed: {' '.join(cmd)}\n{proc.stderr.strip()}")
            print(
                f"Warning: qmd command failed but continuing: {' '.join(cmd)}",
                flush=True,
            )

    run_qmd("collection", "add", str(tmp_dir), "--name", collection, "--mask", "**/*.md")
    run_qmd("context", "add", str(tmp_dir), "Benchmark corpus for Qdrant vs QMD")
    run_qmd("update", timeout=180)
    run_qmd("embed", timeout=180, required=False)
    return tmp_dir


def parse_qmd_sources(raw: str, limit: int) -> List[str]:
    out: List[str] = []
    seen = set()
    for line in (raw or "").splitlines():
        line = line.strip()
        if not line.startswith("qmd://"):
            continue
        source = canonicalize_source(line)
        if source in seen:
            continue
        seen.add(source)
        out.append(source)
        if len(out) >= limit:
            break
    return out


def query_qmd(index_name: str, collection: str, query: str, top_k: int) -> BackendQueryResult:
    start = time.perf_counter()
    cmd = ["qmd", "--index", index_name, "query", query, "-c", collection]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    merged = (proc.stdout or "") + "\n" + (proc.stderr or "")
    sources = parse_qmd_sources(merged, top_k)
    if not sources:
        cmd = ["qmd", "--index", index_name, "search", query, "-c", collection]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        merged = (proc.stdout or "") + "\n" + (proc.stderr or "")
        sources = parse_qmd_sources(merged, top_k)

    latency = (time.perf_counter() - start) * 1000.0
    err = None
    if not sources and proc.returncode != 0:
        err = (proc.stderr or "qmd query failed").strip()
    return BackendQueryResult(sources=sources, latency_ms=latency, error=err)


def ensure_qdrant_collection_http(
    base_url: str,
    collection: str,
    docs: List[CorpusDoc],
    dim: int,
    embedder: HashingEmbedder,
) -> None:
    base = base_url.rstrip("/")
    http_request(
        f"{base}/collections/{urllib.parse.quote(collection)}",
        method="PUT",
        payload={"vectors": {"size": dim, "distance": "Cosine"}},
    )

    points = []
    for idx, doc in enumerate(docs):
        points.append(
            {
                "id": idx + 1,
                "vector": embedder.encode(f"{doc.title}\n{doc.body}"),
                "payload": {
                    "source": doc.source,
                    "canonical_source": doc.canonical_source,
                    "title": doc.title,
                    "body": doc.body,
                },
            }
        )

    http_request(
        f"{base}/collections/{urllib.parse.quote(collection)}/points?wait=true",
        method="PUT",
        payload={"points": points},
        timeout=60.0,
    )


def delete_qdrant_collection_http(base_url: str, collection: str) -> None:
    try:
        http_request(f"{base_url.rstrip('/')}/collections/{urllib.parse.quote(collection)}", method="DELETE")
    except Exception:
        pass


def query_qdrant_http(base_url: str, collection: str, query: str, top_k: int, embedder: HashingEmbedder) -> BackendQueryResult:
    start = time.perf_counter()
    payload = {
        "vector": embedder.encode(query),
        "limit": top_k,
        "with_payload": ["canonical_source"],
    }
    try:
        response = http_request(
            f"{base_url.rstrip('/')}/collections/{urllib.parse.quote(collection)}/points/search",
            method="POST",
            payload=payload,
            timeout=30.0,
        )
        results = response.get("result", [])
        sources: List[str] = []
        for row in results:
            src = (row.get("payload") or {}).get("canonical_source")
            if src:
                sources.append(src)
        latency = (time.perf_counter() - start) * 1000.0
        return BackendQueryResult(sources=sources[:top_k], latency_ms=latency)
    except Exception as exc:
        latency = (time.perf_counter() - start) * 1000.0
        return BackendQueryResult(sources=[], latency_ms=latency, error=str(exc))


def ensure_qdrant_collection_local(
    client: "QdrantClient",
    collection: str,
    docs: List[CorpusDoc],
    dim: int,
    embedder: HashingEmbedder,
) -> None:
    if any(x is None for x in [Distance, PointStruct, VectorParams]):
        raise RuntimeError("qdrant-client models are unavailable")

    if client.collection_exists(collection_name=collection):
        client.delete_collection(collection_name=collection)
    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

    points = []
    for idx, doc in enumerate(docs):
        points.append(
            PointStruct(
                id=idx + 1,
                vector=[],
                payload={
                    "source": doc.source,
                    "canonical_source": doc.canonical_source,
                    "title": doc.title,
                    "body": doc.body,
                },
            )
        )

    # Vector values are assigned in upsert call to avoid re-allocating payload objects twice.
    vectors = [embedder.encode(f"{doc.title}\n{doc.body}") for doc in docs]
    for point, vector in zip(points, vectors):
        point.vector = vector

    client.upsert(collection_name=collection, points=points, wait=True)


def delete_qdrant_collection_local(client: "QdrantClient", collection: str) -> None:
    try:
        client.delete_collection(collection_name=collection)
    except Exception:
        pass


def query_qdrant_local(
    client: "QdrantClient",
    collection: str,
    query: str,
    top_k: int,
    embedder: HashingEmbedder,
) -> BackendQueryResult:
    start = time.perf_counter()
    try:
        response = client.query_points(
            collection_name=collection,
            query=embedder.encode(query),
            limit=top_k,
            with_payload=["canonical_source"],
        )
        rows = response.points
        sources: List[str] = []
        for row in rows:
            payload = row.payload or {}
            src = payload.get("canonical_source")
            if src:
                sources.append(src)
        latency = (time.perf_counter() - start) * 1000.0
        return BackendQueryResult(sources=sources[:top_k], latency_ms=latency)
    except Exception as exc:
        latency = (time.perf_counter() - start) * 1000.0
        return BackendQueryResult(sources=[], latency_ms=latency, error=str(exc))


def rank_of(expected: str, results: List[str]) -> Optional[int]:
    for i, src in enumerate(results, start=1):
        if src == expected:
            return i
    return None


def evaluate_backend(name: str, query_results: List[Tuple[Dict[str, str], BackendQueryResult]], k: int) -> Dict[str, object]:
    hits = 0
    rr = 0.0
    latencies = []
    errors = 0
    per_query = []

    for row, result in query_results:
        expected = row["expected_source"]
        rank = rank_of(expected, result.sources)
        hit = rank is not None and rank <= k
        if hit:
            hits += 1
            rr += 1.0 / rank
        latencies.append(result.latency_ms)
        if result.error:
            errors += 1

        per_query.append(
            {
                "name": row["name"],
                "expected": expected,
                "rank": rank,
                "hit": hit,
                "latency_ms": result.latency_ms,
                "error": result.error,
                "top_sources": result.sources,
            }
        )

    n = max(1, len(query_results))
    return {
        "backend": name,
        "queries": len(query_results),
        "hit_at_k": hits / n,
        "mrr": rr / n,
        "avg_latency_ms": mean(latencies) if latencies else 0.0,
        "p95_latency_ms": sorted(latencies)[max(0, math.ceil(len(latencies) * 0.95) - 1)] if latencies else 0.0,
        "errors": errors,
        "details": per_query,
    }


def print_human_report(qmd_report: Dict[str, object], qdrant_report: Dict[str, object], k: int) -> int:
    print("Qdrant vs QMD Benchmark")
    print(f"Quality metric: hit@{k}, MRR")
    print("")

    for report in [qmd_report, qdrant_report]:
        print(f"Backend: {report['backend']}")
        print(f"  hit@{k}: {report['hit_at_k']:.2f}")
        print(f"  MRR: {report['mrr']:.2f}")
        print(f"  avg latency: {report['avg_latency_ms']:.1f} ms")
        print(f"  p95 latency: {report['p95_latency_ms']:.1f} ms")
        print(f"  errors: {report['errors']}")
        print("")

    print("Per-query ranks:")
    qmd_by_name = {d["name"]: d for d in qmd_report["details"]}
    qdrant_by_name = {d["name"]: d for d in qdrant_report["details"]}
    names = [d["name"] for d in qmd_report["details"]]
    for name in names:
        qmd_rank = qmd_by_name[name]["rank"]
        qdr_rank = qdrant_by_name[name]["rank"]
        print(f"- {name}: qmd_rank={qmd_rank} qdrant_rank={qdr_rank}")

    qdrant_better = (
        qdrant_report["hit_at_k"] > qmd_report["hit_at_k"]
        or (
            qdrant_report["hit_at_k"] == qmd_report["hit_at_k"]
            and qdrant_report["mrr"] >= qmd_report["mrr"]
        )
    )

    if qdrant_better:
        print("\nRESULT: Qdrant meets or exceeds QMD on this benchmark.")
        return 0

    print("\nRESULT: Qdrant underperforms QMD on this benchmark.")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare Qdrant and QMD retrieval on labeled scenarios.")
    parser.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--vector-dim", type=int, default=384)
    parser.add_argument(
        "--qdrant-local",
        action="store_true",
        help="Use qdrant-client local engine (:memory:) instead of HTTP service.",
    )
    parser.add_argument("--start-qdrant-docker", action="store_true")
    parser.add_argument("--qdrant-container-name", default="tr-qdrant-bench")
    parser.add_argument("--keep-temp", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    query_rows, docs = build_corpus()
    collection = docs[0].collection if docs else "benchmark"

    qmd_index = f"tr-qmd-vs-qdrant-{int(time.time())}"
    qdrant_collection = f"tr-bench-{int(time.time())}"

    started_docker = False
    tmp_qmd_dir: Optional[Path] = None
    qdrant_client_local = None

    qdrant_embedder = HashingEmbedder(dim=args.vector_dim)
    qdrant_embedder.fit([f"{d.title}\n{d.body}" for d in docs])

    try:
        if args.qdrant_local:
            if QdrantClient is None:
                raise RuntimeError(
                    "qdrant-client is not installed. Install it in a venv and rerun with --qdrant-local."
                )
            qdrant_client_local = QdrantClient(":memory:")
        else:
            if not qdrant_ready(args.qdrant_url):
                if args.start_qdrant_docker:
                    start_qdrant_docker(args.qdrant_container_name)
                    started_docker = True
                    deadline = time.time() + 25
                    while time.time() < deadline:
                        if qdrant_ready(args.qdrant_url):
                            break
                        time.sleep(0.5)
                if not qdrant_ready(args.qdrant_url):
                    raise RuntimeError(
                        f"Qdrant not reachable at {args.qdrant_url}. Start it or use --start-qdrant-docker."
                    )

        tmp_qmd_dir = ensure_qmd_corpus(qmd_index, collection, docs)

        # Build Qdrant index
        if qdrant_client_local is not None:
            ensure_qdrant_collection_local(
                qdrant_client_local, qdrant_collection, docs, args.vector_dim, qdrant_embedder
            )
        else:
            ensure_qdrant_collection_http(
                args.qdrant_url, qdrant_collection, docs, args.vector_dim, qdrant_embedder
            )

        # Warmup once
        _ = query_qmd(qmd_index, collection, query_rows[0]["query"], args.top_k)
        if qdrant_client_local is not None:
            _ = query_qdrant_local(
                qdrant_client_local, qdrant_collection, query_rows[0]["query"], args.top_k, qdrant_embedder
            )
        else:
            _ = query_qdrant_http(
                args.qdrant_url, qdrant_collection, query_rows[0]["query"], args.top_k, qdrant_embedder
            )

        qmd_results: List[Tuple[Dict[str, str], BackendQueryResult]] = []
        qdrant_results: List[Tuple[Dict[str, str], BackendQueryResult]] = []

        for row in query_rows:
            qmd_results.append((row, query_qmd(qmd_index, collection, row["query"], args.top_k)))
            if qdrant_client_local is not None:
                qdrant_results.append(
                    (
                        row,
                        query_qdrant_local(
                            qdrant_client_local,
                            qdrant_collection,
                            row["query"],
                            args.top_k,
                            qdrant_embedder,
                        ),
                    )
                )
            else:
                qdrant_results.append(
                    (
                        row,
                        query_qdrant_http(
                            args.qdrant_url,
                            qdrant_collection,
                            row["query"],
                            args.top_k,
                            qdrant_embedder,
                        ),
                    )
                )

        qmd_report = evaluate_backend("qmd", qmd_results, args.top_k)
        qdrant_report = evaluate_backend("qdrant", qdrant_results, args.top_k)

        output = {
            "config": {
                "top_k": args.top_k,
                "vector_dim": args.vector_dim,
                "qmd_index": qmd_index,
                "qdrant_collection": qdrant_collection,
                "qdrant_url": args.qdrant_url,
                "qdrant_mode": "local" if qdrant_client_local is not None else "http",
            },
            "qmd": qmd_report,
            "qdrant": qdrant_report,
        }

        if args.json:
            print(json.dumps(output, indent=2))
            return 0

        return print_human_report(qmd_report, qdrant_report, args.top_k)

    finally:
        if not args.keep_temp:
            if tmp_qmd_dir is not None:
                shutil.rmtree(tmp_qmd_dir, ignore_errors=True)
            if qdrant_client_local is not None:
                delete_qdrant_collection_local(qdrant_client_local, qdrant_collection)
            else:
                delete_qdrant_collection_http(args.qdrant_url, qdrant_collection)
            subprocess.run(["qmd", "--index", qmd_index, "collection", "remove", collection], capture_output=True, text=True)

        if started_docker:
            stop_qdrant_docker(args.qdrant_container_name)


if __name__ == "__main__":
    raise SystemExit(main())
