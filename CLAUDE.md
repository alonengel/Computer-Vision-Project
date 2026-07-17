# CLAUDE.md

Course project: few-shot classification with Flow Matching, 4 stages (see `_docs/bigPicture.txt`).
Stage 1 (baselines) is complete; Stage 2 (Flow Matching as final layer) is next.

## Environment — non-negotiable

- Run ALL Python with the prebuilt ROCm venv:
  `C:\Users\Alon\Desktop\cv-ex2\rocm_win312\Scripts\python.exe`
- **Never pip-install into that venv** — it holds a hand-built ROCm-on-Windows torch stack (AMD RX 7900 XTX). If a package is missing, tell the user; do not install.
- Set `KMP_DUPLICATE_LIB_OK=TRUE` before torch imports (the `clip` package crashes otherwise). `src/utils.py` and `tasks.ps1` already do this.
- CLIP models must be forced `.float()` — fp16 misbehaves on ROCm.
- Prefer `.\tasks.ps1 <setup|extract|run|figures|notebook|check>` over raw commands.

## Data & experiment integrity (ADRs are binding)

- **Fixed episodes (ADR 0002):** every method — in every stage — evaluates on the committed index files in `results/artifacts/episodes/`. Never resample. Episode files carry a `pool_fingerprint` asserted against feature caches at eval time; a mismatch means regenerate, not bypass.
- **No eval-split statistics as training targets (ADR 0003):** Stage 2/3 Flow-Matching targets are per-episode support prototypes or CLIP text embeddings only. Train-split prototypes exist for train-split use.
- **Backbones stay frozen (ADR 0001).** Embeddings come from the caches in `results/features/` (regenerable via `tasks.ps1 extract`).
- `timm/mini-imagenet` is pinned to revision `bd8779f9d33c061ea6e75fdd3bce4e43dd679060` in `src/data.py`; do not unpin.
- Hyperparameters are fixed a priori (report §2.5). Never tune anything on test data — including reacting to smoke-run numbers (smoke runs use test episodes).
- All tunables live in `config/config.json`, never hardcoded.

## Statistics & claims

- Episodic: mean ± 95% CI (1.96·std/√n over per-episode accuracies). Simple protocol: mean ± sample std over seeds.
- Any "A beats B" claim uses **paired** per-episode difference CIs from the raw arrays in `results/metrics/raw/` — never marginal-CI-overlap eyeballing.
- ResNet-50 on Mini-ImageNet is ImageNet-label-contaminated: always flag it (†), never present it as an honest few-shot result.
- Committed raw arrays are canonical; CPU/GPU re-runs can flip float-tie queries.
- After changing metrics or tables, `tasks.ps1 check` (repro_check) must pass before committing.

## Figures & documents

- Figure text uses the display-name/style maps in `src/visualize.py` (`METHOD_STYLES`, `DATASET_NAMES`) — raw pipeline IDs (`proto_cos__clip_vitb32`, `mnist`) must never appear in figure text or report tables. Keep colors/markers consistent via those maps; legends outside the axes.
- Every experiment produces styled figures in `results/figures/`; the professor grades visual examples and formal documentation heavily.
- `docs/LAB_NOTEBOOK.md` is chronological and updated at every step: what/why/commands/errors+fixes. Never rewrite history in it.
- Reports follow formal academic structure (abstract, numbered sections, numbered+captioned embedded figures, references). Number-bearing tables are generated programmatically from the CSVs, not typed by hand.
- Non-obvious decisions get a short ADR in `docs/adr/`.
- The presentation notebook is built from `notebooks/nb_sections/s*.py` via `build_notebook.py` — edit sections, never the .ipynb directly.
- **The notebook must be self-contained.** Reviewers (professor, teammates, external LLMs) often see *only* the executed notebook — no `src/`, no `results/`, no report. Any evidence, policy, or claim that affects grading must therefore appear in the notebook itself: protocol-integrity rules (ADR summaries), hyperparameter provenance, paired statistical comparisons, equivalence benchmarks, and every caveat attached to a reported number. If a review finds a "gap" that is already solved elsewhere in the repo, that is a notebook packaging bug — fix the notebook.

## Workflow

- Commit at every milestone; push to `origin` (private GitHub `alonengel/Computer-Vision-Project`, SSH).
- Review gates before declaring a milestone done: methodology review before expensive runs; adversarial number-check and presentation review before finalizing deliverables (`.claude/agents/cv-expert.md`, `.claude/agents/critical-reviewer.md`).
- Long downloads/extractions run in background; HF stalls are resumable (cache keeps completed shards). `HF_TOKEN` is set as a user-level env var.
- Temp/scratch scripts go to the session scratchpad, never into the repo.
