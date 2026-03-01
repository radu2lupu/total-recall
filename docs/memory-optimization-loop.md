# Memory Optimization Loop

This document translates cognitive-memory principles into an engineering loop for `total-recall`.

## 1) Human Memory Principles to Model

- Retrieval cues beat brute-force recall: memories are easier to retrieve when indexed by meaningful cues (task, bug, decision, file, outcome).
- Recency matters, but is not enough: recent memories are easier to access, but high-importance older memories should survive.
- Utility over volume: humans prioritize actionable and context-relevant memories, not exhaustive logs.
- Compression + gist: people preserve "gist" before details; detailed traces are pulled only when needed.
- Forgetting is adaptive: pruning low-value memories improves retrieval speed and reduces interference.

## 2) Implementation Plan (One Improvement at a Time)

1. Query-time shortlist + token budget (implemented)
- Rank candidate memories by semantic score, query overlap, recency decay, and actionability.
- Output only top candidates within a configurable token budget.

2. Cue enrichment on write (implemented)
- Add structured tags in generated notes (area, intent, outcome, confidence).
- Improve retrieval cue quality without increasing note length.

3. Importance-aware persistence (implemented)
- Promote durable memories (architectural decisions, repeated failures, verified fixes) into `decisions/` or `patterns/`.
- De-prioritize noisy session chatter.

4. Adaptive retrieval strategy (implemented)
- If shortlist relevance is weak, expand search breadth.
- If shortlist is strong, keep retrieval narrow to minimize tokens.

## 3) Test Criteria for Each Iteration

- Token efficiency: optimized output stays within budget (default 900 tokens).
- Retrieval relevance: query-term coverage remains high after compression.
- Actionability: at least one selected memory includes implementation/decision/fix details.
- Determinism: repeated runs on same input yield the same shortlist order.

Run evaluator:

```bash
scripts/evaluate_memory_optimizer.py \
  --project total-recall \
  --query "memory optimization token limits retrieval quality" \
  --token-budget 900
```

Run write-path evaluator:

```bash
python3 scripts/evaluate_memory_write.py --max-session-tokens 420
```

Run agent replay integration benchmark:

```bash
python3 scripts/evaluate_agent_replay.py --token-budget 900
```

Run real coding gauntlet (actual `codex exec` cold vs warm):

```bash
python3 scripts/evaluate_agent_gauntlet.py \
  --mode agent \
  --token-budget 900 \
  --agent-timeout-seconds 420

# Durable suite (multi-scenario, repeated, includes irrelevant-memory control track):
python3 scripts/evaluate_agent_gauntlet_suite.py \
  --mode agent \
  --repeats 2 \
  --order-policy alternate \
  --agent-timeout-seconds 240 \
  --json

# Deterministic harness sanity check:
python3 scripts/evaluate_agent_gauntlet_suite.py --mode reference --repeats 1 --json
```

Run backend comparison benchmark (Qdrant vs QMD) on labeled replay scenarios:

```bash
# Requires qdrant-client installed in .venv-bench
.venv-bench/bin/python scripts/evaluate_qdrant_vs_qmd.py --qdrant-local
```

Run hybrid real-corpus benchmark (Qdrant dense+sparse+RRF+rerank vs QMD):

```bash
.venv-bench/bin/python scripts/evaluate_qdrant_hybrid_real.py \
  --project total-recall \
  --sample-size 20 \
  --top-k 5

# Bounded runtime + compact/procedural warm-memory cues for gauntlet replay:
TOTAL_RECALL_GAUNTLET_TIMEOUT=220 .venv-bench/bin/python scripts/evaluate_qdrant_hybrid_real.py \
  --project total-recall \
  --sample-size 6 \
  --top-k 5 \
  --qmd-timeout 20 \
  --qmd-search-timeout 8 \
  --gauntlet-memory-k 2 \
  --gauntlet-token-budget 420 \
  --run-gauntlet-replay

# Repeat gauntlet runs and inspect median/faster-rate stability:
TOTAL_RECALL_GAUNTLET_TIMEOUT=220 .venv-bench/bin/python scripts/evaluate_qdrant_hybrid_real.py \
  --project total-recall \
  --sample-size 2 \
  --top-k 5 \
  --qmd-timeout 20 \
  --qmd-search-timeout 8 \
  --gauntlet-memory-k 2 \
  --gauntlet-token-budget 420 \
  --gauntlet-repeats 2 \
  --run-gauntlet-replay \
  --json
```

Human-memory research mapping:

```bash
cat docs/human-memory-research-notes.md
```

## 4) Iterative Decision Loop

For each improvement:

1. Hypothesis
- Example: "Adding actionability weighting improves usefulness without exceeding token budget."

2. Implement minimally
- Keep scope narrow and isolated to one behavior.

3. Assert
- Run evaluator and compare against prior run.

4. Learn
- Record what improved/regressed.

5. Decide
- If metrics improved or held steady: iterate on same idea.
- If metrics regressed and no clear fix: scrap and move to next idea.

## 5) Gauntlet Trust Gates

- Multi-scenario coverage: minimum 3 scenario families with different bug structures.
- Repeatability: run at least `N=5` repeats with alternating run order.
- Stability metrics: require median warm-vs-cold delta and CI, not single-run wins.
- Control comparison: warm memory must also beat control (irrelevant memory) often enough to show semantic usefulness, not just token priming.
- Patch-quality proxy: compare patch minimality/precision (changed files + changed lines) so gains are not judged by speed alone.
- Memory alignment gate: inject warm memory only when cue/file hints align with task focus; suppress irrelevant memory.
- Pass-rate floor: warm and control should keep high pass rates; speed gains that come from quality drops do not count.

## Current Iteration Status

- Iteration 1 complete: query-time shortlist + token budget + evaluation harness.
- Iteration 2 complete: structured write cues + durable memory promotion into decision/bug/pattern lanes.
- Iteration 3 complete: adaptive retrieval expansion + durable-lane score boosts + low-utility tail filtering.
- Iteration 4 complete: stronger query-overlap weighting and scope checks to reduce cross-repo leakage in shortlist results.
- Iteration 5 complete: automatic repeated-topic synthesis into canonical `patterns/*-synthesized.md` notes.
- Iteration 6 complete: integration benchmark added for cold-vs-warm replay solving effectiveness and speed.
- Iteration 7 complete: real-agent gauntlet benchmark with a tricky failing codebase and measured warm-memory speedup.
- Iteration 8 complete: encoding-specific cue fields + cue/intent-aligned ranking + MMR shortlist diversity + action playbook extraction.
- Iteration 9 complete: project-isolated qmd indices (`--index <prefix>-<project>`) wired through CLI/server, plus lazy collection bootstrap for new indices to preserve query reliability after migration.
- Iteration 10 complete: resilient retrieval handling for qmd non-zero exits that still emit valid `qmd://` hits (CLI/server/evaluator) to avoid false "no memory" results.
- Iteration 11 complete: added `evaluate_qdrant_vs_qmd.py` backend comparison harness with objective quality metrics (hit@k, MRR) and latency; supports Qdrant local engine and dedicated-service mode.
- Iteration 12 complete: implemented `qdrant_hybrid_retriever.py` (dense + sparse BM25 vectors + Qdrant RRF fusion + rerank) and `evaluate_qdrant_hybrid_real.py` over real project memories, including optional real-memory gauntlet replay tracks.
- Iteration 13 complete: added diversity-aware selection helpers, bounded qmd timeouts for stable benchmark loops, and compact/procedural warm-memory cues (action-script format). In a gauntlet replay run, warm memory became faster than cold while still passing tests (cold 77.03s, warm-qmd 63.97s, warm-qdrant 75.52s).
- Iteration 14 complete: added multi-run gauntlet aggregation (`--gauntlet-repeats`) with median timings, warm-vs-cold faster-rate, and all-pass-rate. Early repeated run showed high variance and one warm-qmd timeout; warm-qdrant remained comparatively stable and had better median than cold in that sample.
- Iteration 15 complete: introduced durable multi-scenario gauntlet suite (`scripts/evaluate_agent_gauntlet_suite.py`) backed by `benchmarks/gauntlet/scenarios.json`, added two new tricky scenarios (pricing boundary bugs, token bucket precision bugs), bootstrap CI on median deltas, and a control track with irrelevant memory to prevent false positives from prompt priming. Suite data now exposes when warm memory is slower or no better than control, improving trust in conclusions.
- Iteration 16 complete: added distinctiveness-weighted retrieval scoring (cue-overload mitigation), procedural-confidence gating (`TOTAL_RECALL_PROCEDURAL_MIN_CONFIDENCE`), compact if-then procedural recipes, and stricter evaluator support for procedural outputs. Also aligned gauntlet memory blocks with real structured cues (`intent`, `cue_*`, `files`), and made raw excerpt injection optional (`--include-memory-excerpt`, default off) to reduce prompt overhead.
- Iteration 17 complete: fixed over-compression of procedural fix hints (preserve threshold/rounding context), expanded fix-hint extraction for boundary/rounding patterns, and tightened recipe phrasing to "direct deltas." In a full suite run (`--repeats 1 --token-budget 420`) all gates passed (`macro_warm_win_rate=0.67`, `macro_warm_beats_control_rate=0.67`). However, targeted repeated pricing runs (`--scenario pricing_engine --repeats 2`) remain noisy (`warm_win_rate=0.5`, `warm_beats_control_rate=0.5`), so durability at higher N is not yet proven.
- Iteration 18 complete: added patch-quality instrumentation to gauntlet tracks (`changed_files_count`, `changed_lines`, `focus_only`) and a macro precision gate (`macro_warm_more_precise_than_control_rate`) to avoid relying on timing-only wins. Artifact files (`__pycache__`, `.pyc`, `.pytest_cache`) are excluded from patch metrics.
- Iteration 19 complete: added focus-alignment memory gating and warm-start execution policy in gauntlet prompts. Warm memory is now injected only when shortlist cues/file paths align with scenario focus files; control memory remains unaligned and is suppressed. Added per-run alignment metrics.
- Iteration 20 complete: refined procedural fix hints to prefer code deltas (and skip cue-noise lines), tightened warm prompt policy (patch-first, minimal exploration, concise final response), and balanced run order across scenarios (`alternate` now rotates by scenario+repeat). In one full durability run (`repeats=2`, token budget 420), suite passed all gates with strong warm gains on inventory/token-bucket and improved control separation (`macro_warm_win_rate=0.67`, `macro_warm_beats_control_rate=0.83`, `macro_warm_more_precise_than_control_rate=1.0`), though a single warm timeout remained in pricing.
- Next iteration candidate: add first-attempt correctness and test-run-count metrics from agent stdout to reduce sensitivity to wall-clock variance, then raise durability bar to `N>=5`.
