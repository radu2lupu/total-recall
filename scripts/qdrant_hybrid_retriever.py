#!/usr/bin/env python3
"""Hybrid Qdrant retriever for real memory corpora (dense + sparse + RRF + rerank)."""

from __future__ import annotations

import hashlib
import math
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from qdrant_client import QdrantClient, models

try:
    from fastembed import TextEmbedding
except Exception:  # pragma: no cover
    TextEmbedding = None


WORD_RE = re.compile(r"[a-z0-9_./:-]+")
STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "for", "in", "on", "with", "by", "from",
    "session", "decision", "bug", "pattern", "fixed", "added", "updated", "implemented",
}

TYPE_PRIORITY = {
    "bugs": 4,
    "decisions": 3,
    "patterns": 2,
    "sessions": 1,
}


@dataclass
class MemoryDoc:
    id: int
    source: str
    path: str
    title: str
    content: str
    date: Optional[datetime]


@dataclass
class RetrievalHit:
    source: str
    title: str
    path: str
    content: str
    fusion_score: float
    rerank_score: float


@dataclass
class RetrievalResult:
    hits: List[RetrievalHit]
    retrieval_ms: float
    rerank_ms: float
    total_ms: float


class DenseEncoder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model_name = model_name
        self._fast = None
        self._dim = 384
        if TextEmbedding is not None:
            self._fast = TextEmbedding(model_name=model_name, lazy_load=True)
            # Trigger model load once; needed for dim discovery.
            sample = list(self._fast.embed(["dimension probe"]))[0]
            self._dim = int(len(sample))

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: Sequence[str]) -> List[List[float]]:
        if self._fast is not None:
            vectors = [np.asarray(v, dtype=np.float32) for v in self._fast.embed(list(texts))]
            return [self._normalize(v).tolist() for v in vectors]

        vectors = [self._hash_embed(text) for text in texts]
        return [self._normalize(v).tolist() for v in vectors]

    def _hash_embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self._dim, dtype=np.float32)
        for token in tokenize(text):
            digest = hashlib.md5(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % self._dim
            sign = 1.0 if (digest[4] & 1) == 0 else -1.0
            vec[idx] += sign
        return vec

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            return vec / norm
        return vec


def tokenize(text: str) -> List[str]:
    return WORD_RE.findall((text or "").lower())


def significant_tokens(text: str) -> List[str]:
    return [t for t in tokenize(text) if t not in STOPWORDS and len(t) > 2]


def parse_memory_date(path: str, content: str) -> Optional[datetime]:
    m = re.search(r"(\d{4}-\d{2}-\d{2})", path)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y-%m-%d")
        except ValueError:
            pass

    m = re.search(r"\*\*Date:\*\*\s*(\d{4}-\d{2}-\d{2})", content)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y-%m-%d")
        except ValueError:
            return None
    return None


def load_project_docs(project: str, shared_root: Optional[str] = None) -> List[MemoryDoc]:
    shared_root = shared_root or os.path.expanduser(os.getenv("TOTAL_RECALL_SHARED_ROOT", "~/.ai-memory/knowledge"))
    project_dir = Path(shared_root).expanduser().resolve() / project
    if not project_dir.exists():
        return []

    docs: List[MemoryDoc] = []
    next_id = 1
    for file in sorted(project_dir.rglob("*.md")):
        rel = file.relative_to(project_dir).as_posix()
        text = file.read_text(encoding="utf-8", errors="ignore")
        title = ""
        for line in text.splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
                break
        if not title:
            title = Path(rel).stem.replace("-", " ")

        source = f"tr://{project}/{rel}"
        docs.append(
            MemoryDoc(
                id=next_id,
                source=source,
                path=rel,
                title=title,
                content=text,
                date=parse_memory_date(rel, text),
            )
        )
        next_id += 1
    return docs


class SparseBM25:
    def __init__(self, docs: Sequence[MemoryDoc], k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.token_to_index: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.doc_len: Dict[int, int] = {}
        self.avgdl = 1.0
        self.doc_tf: Dict[int, Dict[str, int]] = {}
        self.doc_sparse: Dict[int, models.SparseVector] = {}
        self._build(docs)

    def _build(self, docs: Sequence[MemoryDoc]) -> None:
        doc_freq: Dict[str, int] = {}
        lengths: List[int] = []

        for doc in docs:
            tokens = significant_tokens(f"{doc.title}\n{doc.content}")
            tf: Dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            self.doc_tf[doc.id] = tf
            self.doc_len[doc.id] = len(tokens)
            lengths.append(max(1, len(tokens)))
            for token in tf.keys():
                doc_freq[token] = doc_freq.get(token, 0) + 1

        n_docs = max(1, len(docs))
        self.avgdl = sum(lengths) / max(1, len(lengths))

        for idx, token in enumerate(sorted(doc_freq.keys())):
            self.token_to_index[token] = idx
            df = doc_freq[token]
            self.idf[token] = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)

        for doc in docs:
            self.doc_sparse[doc.id] = self.doc_to_sparse(doc.id)

    def doc_to_sparse(self, doc_id: int) -> models.SparseVector:
        tf = self.doc_tf.get(doc_id, {})
        dl = max(1, self.doc_len.get(doc_id, 1))
        indices: List[int] = []
        values: List[float] = []

        for token, freq in tf.items():
            idf = self.idf.get(token, 0.0)
            num = freq * (self.k1 + 1.0)
            den = freq + self.k1 * (1.0 - self.b + self.b * dl / self.avgdl)
            weight = idf * (num / den)
            if abs(weight) <= 1e-9:
                continue
            indices.append(self.token_to_index[token])
            values.append(float(weight))

        return models.SparseVector(indices=indices, values=values)

    def query_to_sparse(self, query: str) -> models.SparseVector:
        q_tf: Dict[str, int] = {}
        for token in significant_tokens(query):
            if token in self.token_to_index:
                q_tf[token] = q_tf.get(token, 0) + 1

        indices: List[int] = []
        values: List[float] = []
        for token, freq in q_tf.items():
            indices.append(self.token_to_index[token])
            values.append(float(self.idf.get(token, 0.0) * (1.0 + math.log(freq))))

        return models.SparseVector(indices=indices, values=values)

    def score(self, query: str, doc_id: int) -> float:
        tf = self.doc_tf.get(doc_id, {})
        dl = max(1, self.doc_len.get(doc_id, 1))
        q_terms: Dict[str, int] = {}
        for t in significant_tokens(query):
            q_terms[t] = q_terms.get(t, 0) + 1

        score = 0.0
        for token, qf in q_terms.items():
            f = tf.get(token, 0)
            if f <= 0:
                continue
            idf = self.idf.get(token, 0.0)
            num = f * (self.k1 + 1.0)
            den = f + self.k1 * (1.0 - self.b + self.b * dl / self.avgdl)
            score += idf * (num / den) * (1.0 + math.log(qf))
        return float(score)


class QdrantHybridRetriever:
    def __init__(
        self,
        docs: Sequence[MemoryDoc],
        collection_name: str,
        qdrant_url: Optional[str] = None,
        dense_model: str = "BAAI/bge-small-en-v1.5",
        weight_dense: float = 0.45,
        weight_sparse: float = 0.40,
        weight_title: float = 0.10,
        weight_recent: float = 0.05,
    ):
        self.docs = list(docs)
        self.docs_by_id = {d.id: d for d in self.docs}
        self.collection_name = collection_name
        self.encoder = DenseEncoder(model_name=dense_model)
        self.sparse = SparseBM25(self.docs)
        self.query_dense_cache: Dict[str, np.ndarray] = {}

        if qdrant_url:
            self.client = QdrantClient(url=qdrant_url)
        else:
            self.client = QdrantClient(":memory:")

        self.doc_dense: Dict[int, np.ndarray] = {}
        self.weight_dense = weight_dense
        self.weight_sparse = weight_sparse
        self.weight_title = weight_title
        self.weight_recent = weight_recent

    def build(self) -> None:
        if self.client.collection_exists(collection_name=self.collection_name):
            self.client.delete_collection(collection_name=self.collection_name)

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "dense": models.VectorParams(
                    size=self.encoder.dim,
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams()
            },
        )

        dense_vectors = self.encoder.encode([f"{d.title}\n{d.content}" for d in self.docs])
        points: List[models.PointStruct] = []

        for doc, dense in zip(self.docs, dense_vectors):
            dense_np = np.asarray(dense, dtype=np.float32)
            self.doc_dense[doc.id] = dense_np
            points.append(
                models.PointStruct(
                    id=doc.id,
                    vector={
                        "dense": dense,
                        "sparse": self.sparse.doc_sparse[doc.id],
                    },
                    payload={
                        "source": doc.source,
                        "path": doc.path,
                        "title": doc.title,
                        "content": doc.content,
                        "date": doc.date.isoformat() if doc.date else None,
                    },
                )
            )

        self.client.upsert(collection_name=self.collection_name, points=points, wait=True)

    def _query_dense(self, query: str) -> np.ndarray:
        cached = self.query_dense_cache.get(query)
        if cached is not None:
            return cached
        dense = np.asarray(self.encoder.encode([query])[0], dtype=np.float32)
        self.query_dense_cache[query] = dense
        return dense

    def search(
        self,
        query: str,
        top_k: int = 8,
        prefetch_k: int = 40,
        rerank_k: int = 24,
        diversity: bool = False,
    ) -> RetrievalResult:
        start = time.perf_counter()
        q_dense = self._query_dense(query)
        q_sparse = self.sparse.query_to_sparse(query)

        prefetch = [
            models.Prefetch(query=q_dense.tolist(), using="dense", limit=prefetch_k),
            models.Prefetch(query=q_sparse, using="sparse", limit=prefetch_k),
        ]

        response = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            with_payload=["source", "path", "title", "content", "date"],
            limit=rerank_k,
        )
        retrieval_ms = (time.perf_counter() - start) * 1000.0

        rerank_start = time.perf_counter()
        rows = response.points
        sparse_scores: List[float] = []
        dense_scores: List[float] = []
        title_scores: List[float] = []
        recent_scores: List[float] = []
        scored_rows: List[Tuple[models.ScoredPoint, float, float, float, float]] = []

        q_terms = set(significant_tokens(query))
        now = datetime.utcnow()

        for row in rows:
            doc_id = int(row.id)
            payload = row.payload or {}
            title = str(payload.get("title", ""))
            dense_doc = self.doc_dense.get(doc_id)
            dense_score = float(np.dot(q_dense, dense_doc)) if dense_doc is not None else 0.0
            sparse_score = self.sparse.score(query, doc_id)

            title_terms = set(significant_tokens(title))
            title_overlap = (len(q_terms & title_terms) / max(1, len(q_terms))) if q_terms else 0.0

            date_raw = payload.get("date")
            if date_raw:
                try:
                    d = datetime.fromisoformat(str(date_raw))
                    age_days = max(0.0, (now - d).days)
                    recent = math.exp(-age_days / 90.0)
                except Exception:
                    recent = 0.0
            else:
                recent = 0.0

            sparse_scores.append(sparse_score)
            dense_scores.append(dense_score)
            title_scores.append(title_overlap)
            recent_scores.append(recent)
            scored_rows.append((row, dense_score, sparse_score, title_overlap, recent))

        def normalize(values: List[float]) -> List[float]:
            if not values:
                return []
            vmin = min(values)
            vmax = max(values)
            if abs(vmax - vmin) < 1e-9:
                return [0.5 for _ in values]
            return [(v - vmin) / (vmax - vmin) for v in values]

        dense_n = normalize(dense_scores)
        sparse_n = normalize(sparse_scores)
        title_n = normalize(title_scores)
        recent_n = normalize(recent_scores)

        hits: List[RetrievalHit] = []
        for i, (row, _, _, _, _) in enumerate(scored_rows):
            rerank = (
                self.weight_dense * dense_n[i]
                + self.weight_sparse * sparse_n[i]
                + self.weight_title * title_n[i]
                + self.weight_recent * recent_n[i]
            )
            payload = row.payload or {}
            hits.append(
                RetrievalHit(
                    source=str(payload.get("source", "")),
                    title=str(payload.get("title", "")),
                    path=str(payload.get("path", "")),
                    content=str(payload.get("content", "")),
                    fusion_score=float(row.score),
                    rerank_score=float(rerank),
                )
            )

        hits.sort(
            key=lambda h: (
                h.rerank_score,
                TYPE_PRIORITY.get(source_type(h.path), 0),
            ),
            reverse=True,
        )
        if diversity:
            hits = select_diverse_hits(hits, top_k=top_k)
        else:
            hits = hits[:top_k]
        rerank_ms = (time.perf_counter() - rerank_start) * 1000.0
        total_ms = (time.perf_counter() - start) * 1000.0
        return RetrievalResult(hits=hits, retrieval_ms=retrieval_ms, rerank_ms=rerank_ms, total_ms=total_ms)


def generate_real_queries(docs: Sequence[MemoryDoc], limit: int = 24) -> List[Tuple[str, str, str]]:
    """Return (name, query, expected_source)."""
    candidates = [d for d in docs if any(d.path.startswith(p) for p in ("bugs/", "decisions/", "patterns/", "sessions/"))]
    rows: List[Tuple[str, str, str]] = []

    for doc in candidates:
        title = doc.title
        title = re.sub(r"^(session|decision|bug|pattern):\s*", "", title, flags=re.I)
        tokens = []
        for t in significant_tokens(title):
            if t not in tokens:
                tokens.append(t)
        if len(tokens) < 3:
            continue
        query = " ".join(tokens[:8])
        rows.append((Path(doc.path).name, query, doc.source))
        if len(rows) >= limit:
            break

    return rows


def canonicalize_source(source: str) -> str:
    m = re.match(r"^(tr://[^\s:]+(?:/[^\s:]+)+)", source.strip())
    return m.group(1) if m else source.strip()


def source_type(path: str) -> str:
    p = (path or "").strip().lower()
    if "/" in p:
        return p.split("/", 1)[0]
    return p


def _topic_key(path: str, title: str) -> str:
    stem = Path(path or "").stem.lower()
    stem = re.sub(r"^\d{4}-\d{2}-\d{2}-", "", stem)
    stem = re.sub(r"-\d+$", "", stem)
    stem = re.sub(r"[^a-z0-9]+", "-", stem).strip("-")
    if stem:
        return stem
    t = re.sub(r"[^a-z0-9]+", "-", (title or "").lower()).strip("-")
    return t[:80]


def _token_set(text: str) -> set[str]:
    return set(significant_tokens(text))


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / max(1, len(left | right))


def select_diverse_hits(
    hits: Sequence[RetrievalHit],
    top_k: int,
    max_per_topic: int = 1,
    near_dup_jaccard: float = 0.82,
) -> List[RetrievalHit]:
    if not hits:
        return []

    selected: List[RetrievalHit] = []
    topic_counts: Dict[str, int] = {}
    selected_tokens: List[set[str]] = []

    for hit in hits:
        topic = _topic_key(hit.path, hit.title)
        if topic and topic_counts.get(topic, 0) >= max_per_topic:
            continue

        this_tokens = _token_set(f"{hit.title}\n{hit.content}")
        if any(_jaccard(this_tokens, prev) >= near_dup_jaccard for prev in selected_tokens):
            continue

        selected.append(hit)
        selected_tokens.append(this_tokens)
        if topic:
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
        if len(selected) >= top_k:
            break

    if len(selected) < top_k:
        seen = {canonicalize_source(h.source) for h in selected}
        for hit in hits:
            src = canonicalize_source(hit.source)
            if src in seen:
                continue
            selected.append(hit)
            seen.add(src)
            if len(selected) >= top_k:
                break

    return selected[:top_k]


def salient_excerpt(content: str, max_lines: int = 12) -> str:
    lines: List[str] = []
    for line in content.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("- ") or s.startswith("## ") or "`" in s:
            lines.append(s)
        if len(lines) >= max_lines:
            break

    if not lines:
        lines = [ln.strip() for ln in content.splitlines() if ln.strip()][:max_lines]

    return "\n".join(lines)


def hits_to_raw_blocks(hits: Sequence[RetrievalHit], backend_label: str = "qdrant") -> str:
    blocks = []
    for i, h in enumerate(hits, start=1):
        score = max(1, min(99, int(round(h.rerank_score * 100))))
        source = f"{canonicalize_source(h.source)}:1 #h{i:02d}"
        body = salient_excerpt(h.content, max_lines=12)
        blocks.append(
            f"{source}\n"
            f"Title: {h.title}\n"
            f"Context: Real project memory ({backend_label})\n"
            f"Score:  {score}%\n\n"
            f"{body}\n"
        )
    return "\n\n".join(blocks)
