# ADR 0002 — Episode indices generated once and saved to disk

**Status:** accepted (2026-07-17)

**Context.** The project compares Stage 1 baselines against Stage 2/3 Flow Matching variants. The comparison is only valid if every method is evaluated on the exact same support/query sets.

**Decision.** Episode definitions (support and query sample indices per episode) are generated once per (dataset, protocol, seed) and saved under `results/artifacts/episodes/`. All evaluation code — now and in stages 2–3 — loads these files instead of re-sampling.

**Consequences.** Cross-stage comparisons are exact, not just statistically comparable; re-runs are reproducible independent of RNG library changes. The index files are small (integers only) and are committed to git.
