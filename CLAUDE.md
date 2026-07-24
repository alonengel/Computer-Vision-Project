# CLAUDE.md

Course project: few-shot / low-shot classification with Flow Matching, three stages.

**`_docs/stage_1.pdf` is the single source of truth for Stage 1 (ADR 0004).** Where
`_docs/bigPicture.txt`, older ADRs, or anything in git history conflicts with it, the
PDF wins. The pre-spec implementation (MNIST/CIFAR-10/Mini-ImageNet, episodic
protocol) is archived at tag `stage1-v1-episodic`, branch `archive/stage1-v1-episodic`,
and `D:\_backups\Computer-Vision-Project_stage1-v1-episodic_20260724` — never cite its
numbers as current.

## Environment — non-negotiable

- Run ALL Python with the prebuilt ROCm venv:
  `C:\Users\Alon\Desktop\cv-ex2\rocm_win312\Scripts\python.exe`
- **Never pip-install into that venv** — it holds a hand-built ROCm-on-Windows torch stack (AMD RX 7900 XTX). If a package is missing, tell the user; do not install.
- Set `KMP_DUPLICATE_LIB_OK=TRUE` before torch imports (the `clip` package crashes otherwise). `src/utils.py` and `tasks.ps1` already do this.
- CLIP models must be forced `.float()` — fp16 misbehaves on ROCm.
- Prefer `.\tasks.ps1 <setup|data|extract|run|smoke|tables|figures|notebook|check>` over raw commands.
- `Remove-Item` is sandbox-blocked inside the repo; delete tracked files with `git rm`.

## Protocol — what the spec requires

- **Datasets:** DTD (official partition 1), FGVC-Aircraft (`variant` annotation level), Oxford Flowers-102 — all classes, **official train/validation/test splits**. Train and validation are never merged. Validation is for model selection only; the test split is touched only for the final numbers. The spec requires two datasets; we run all three (ADR 0005) and mark the third with ‡.
- **Training sizes:** K ∈ {5, 10, full} images per class. K ∈ {5, 10} use *balanced* subsets of the official training split with seeds {0, 1, 2}, saved as index files under `results/artifacts/subsets/` so every encoder and head sees identical images. "full" is the complete official training split.
- **Encoders (frozen, ADR 0001):** ResNet-18 ImageNet-1K (512-d, pre-classifier) on all datasets; DINOv2 ViT-S/14 (final class token) on FGVC-Aircraft only; CLIP RN50 for the **zero-shot branch only** — never as a linear-probe or image-prototype backbone. Each checkpoint uses its own associated preprocessing. Features are extracted and cached once (`results/features/`); classifiers only ever see caches.
- **Baselines:** linear probe (required) + both prototype branches (ADR 0005) — image-derived prototypes and zero-shot CLIP. The spec's exactly-compliant subset is linear probe + one branch.
- **Linear probe:** `s = Wz + b`, softmax cross-entropy, AdamW, lr 1e-3, weight decay 1e-4, batch 64, ≤200 epochs, checkpoint = highest validation accuracy. Deviations from this suggested configuration must be justified from validation results and reported.
- **Runs:** 3 per training-set size. At K ∈ {5, 10} the three runs are the three subset seeds (initialization fixed) so the spread measures subset sampling; at "full" they are three classifier-initialization seeds (training set fixed). Image prototypes at "full" and zero-shot CLIP are single deterministic runs.
- **Metric:** top-1 accuracy on the complete official test split; 5/10-shot and full-probe reported as mean ± std over the 3 runs.
- **Prototype formula (exact):** `mu_c = normalize( mean_{i in S_c} normalize(z_i) )`, classify by `argmax_c cos(z, mu_c)`. Prototypes come from the selected *training* subset only.
- Never select or tune anything on test results. Hyperparameters are the spec's suggested values, fixed a priori.
- All tunables live in `config/config.json`, never hardcoded.

## Statistics & claims

- Report mean ± sample std over the 3 runs; single-run settings carry no std (do not print "± 0.00" — say it is a single deterministic run).
- Any "A beats B" claim must be checkable against `results/metrics/runs.csv` and the raw arrays in `results/metrics/raw/`.
- After changing metrics or tables, `tasks.ps1 check` (repro_check) must pass before committing.
- Number-bearing tables are generated programmatically (`scripts/make_tables.py`) from the CSVs, never typed by hand.

## Figures & documents

- The five required presentation items: accuracy table; accuracy vs. training-set size with error bars (zero-shot as a horizontal reference line); linear-probe training + validation loss curves; **row-normalized** confusion matrices; 2-D feature visualizations of ~8–10 classes.
- Feature visualizations: same selected classes, same test examples and same class colours across encoders of a dataset; show image features with image-derived prototypes (ResNet-18 / DINOv2) or text prototypes (CLIP); when prototypes are shown the projection is fitted **jointly** to features and prototypes; treat 2-D plots as qualitative only.
- Figure text uses the display-name maps in `src/visualize.py` (`DATASET_NAMES`, `ENCODER_NAMES`, `HEAD_NAMES`) — raw pipeline IDs (`fgvc_aircraft`, `resnet18`, `linear_probe`) must never appear in figure text or report tables. Keep colours consistent via those maps; legends outside the axes.
- `docs/LAB_NOTEBOOK.md` is chronological and updated at every step: what/why/commands/errors+fixes. Never rewrite history in it — it spans both the archived v1 and the current spec-compliant work.
- Reports follow formal academic structure (abstract, numbered sections, numbered+captioned embedded figures, references).
- Non-obvious decisions get a short ADR in `docs/adr/`.
- The presentation notebook is built from `notebooks/nb_sections/s*.py` via `build_notebook.py` — edit sections, never the .ipynb directly.
- **The notebook must be self-contained.** Reviewers (professor, teammates, external LLMs) often see *only* the executed notebook — no `src/`, no `results/`, no report. Any evidence, policy, or claim that affects grading must therefore appear in the notebook itself: the protocol, hyperparameter provenance, run/seed structure, and every caveat attached to a reported number. If a review finds a "gap" that is already solved elsewhere in the repo, that is a notebook packaging bug — fix the notebook.

## Workflow

- Commit at every milestone; push to `origin` (private GitHub `alonengel/Computer-Vision-Project`, SSH).
- Review gates before declaring a milestone done: methodology review before expensive runs; adversarial number-check and presentation review before finalizing deliverables (`.claude/agents/cv-expert.md`, `.claude/agents/critical-reviewer.md`).
- Long downloads run in background; they are resumable (torchvision/HF keep completed archives). `HF_TOKEN` is set as a user-level env var.
- Temp/scratch scripts go to the session scratchpad, never into the repo.
