# Human Memory Research Notes (Applied to Total Recall)

## Objective

Improve LLM memory retrieval quality under token limits by borrowing mechanisms validated in human memory research.

## Research Principles and Translation

1. Spacing effect (retain useful information longer when revisited on spaced intervals)
- Source: [Cepeda et al., 2006](https://pubmed.ncbi.nlm.nih.gov/16719566/)
- Applied translation: maintain concise durable memories (`decisions/`, `patterns/`, `bugs`) and synthesize repeated-topic sessions into canonical notes, reducing noisy repetition.

2. Testing effect / retrieval practice (retrieval itself strengthens future recall utility)
- Source: [Roediger & Karpicke, 2006](https://pubmed.ncbi.nlm.nih.gov/16507066/)
- Applied translation: query output now includes a compact "Top action playbook" so the agent can immediately execute the previously successful fix path and verification command.

3. Encoding specificity + transfer-appropriate processing (best recall when retrieval cues match encoding cues/task)
- Sources:
  - [Tulving & Thomson, 1973 (DOI listing)](https://philpapers.org/rec/TULESA)
  - [Morris, Bransford, Franks, 1977 metadata](https://www.proquest.com/openview/1bf0ed70f1fee7339588af4e9228a143/1?cbl=1819609&pq-origsite=gscholar)
- Applied translation:
  - Memory notes now store explicit cue fields (`cue_problem`, `cue_action`, `cue_verify`).
  - Retrieval ranking boosts memories whose intent/cues align with current query intent.

4. Cue overload and interference (non-distinct cues retrieve wrong memories)
- Sources:
  - [Watkins & Watkins cue-overload discussion](https://pmc.ncbi.nlm.nih.gov/articles/PMC5664228/)
  - [Interference overview citing cue-overload](https://pmc.ncbi.nlm.nih.gov/articles/PMC3389825/)
- Applied translation:
  - Retrieval adds scope-aware penalties to reduce cross-project bleed.
  - MMR-style diversity ranking reduces redundant shortlist items competing for attention.

5. Prioritized replay (choose memories with highest near-term value: need + gain)
- Source: [Mattar & Daw, 2018](https://www.nature.com/articles/s41593-018-0232-z)
- Applied translation:
  - Utility combines overlap ("need"), actionability and durable signal ("gain"), recency, and cue/intent alignment.
  - Token budgeting enforces high-value replay only.

6. Implementation intentions (if-then plans improve execution under pressure)
- Source: [Gollwitzer & Sheeran meta-analysis, 2006](https://pubmed.ncbi.nlm.nih.gov/16770967/)
- Applied translation:
  - Procedural retrieval output now emits compact `if cue -> action -> verify` recipes.
  - This reduces planning overhead in the agent loop and increases direct actionability.

7. Working-memory limits (keep actionable chunks small)
- Source: [Cowan, 2001](https://pubmed.ncbi.nlm.nih.gov/11271757/)
- Applied translation:
  - Procedural output is confidence-gated and compressed into a small step list.
  - Verbose raw memories are only used when confidence is low and shortlist mode is safer.

8. Cue competition and retrieval-induced interference
- Source: [Anderson et al., 1994](https://pubmed.ncbi.nlm.nih.gov/7938335/)
- Applied translation:
  - Ranking now adds a distinctiveness term (query-term rarity across candidate memories).
  - This helps prefer diagnostic memories over generic-but-similar distractors.

## What Changed in Code

- Retrieval optimizer: intent/cue alignment, MMR ranking, action-playbook extraction, scope filtering.
- Retrieval optimizer: distinctiveness-weighted scoring, procedural-confidence gate, compact if-then recipes.
- Memory writer: structured cue fields and durable synthesis support.
- Benchmarks: replay integration + real `codex exec` gauntlet to measure cold-vs-warm effectiveness.

## Practical Metric to Watch

- Warm-vs-cold gauntlet speedup and pass parity:
  - warm must pass at least as often as cold
  - warm should reduce time-to-green under equivalent task difficulty
