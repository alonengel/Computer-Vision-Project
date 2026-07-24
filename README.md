# CVLAB Summer Project — Classification Baselines and Flow Matching

Three-stage course project. **`_docs/stage_1.pdf` is the single source of truth for Stage 1** (ADR 0004).

1. **Stage 1 (this repo state)** — a reproducible classification pipeline on **frozen** pretrained encoders: a linear probe (required) plus both prototype branches (image-derived class prototypes and zero-shot CLIP), on DTD, FGVC-Aircraft and Oxford Flowers-102, with training-set sizes K ∈ {5, 10, full}.
2. **Stage 2** — Flow Matching as the decision layer, transporting embeddings toward the selected branch's class representation.
3. **Stage 3** — Flow Matching between the frozen encoder and the linear probe, trained jointly with cross-entropy (encoder stays frozen).

> The earlier pre-specification implementation (MNIST / CIFAR-10 / Mini-ImageNet, episodic few-shot protocol) is archived at tag `stage1-v1-episodic` and branch `archive/stage1-v1-episodic`. Its numbers are **not** current — see ADR 0004.

## Protocol at a glance

| | |
|---|---|
| Datasets | DTD (official partition 1), FGVC-Aircraft (`variant`), Oxford Flowers-102 ‡ |
| Splits | all classes, official train / validation / test; train and validation never merged |
| Training sizes | K ∈ {5, 10, full}; 5/10-shot are balanced subsets with seeds {0, 1, 2} |
| Encoders (frozen) | ResNet-18 ImageNet-1K (512-d) on all datasets; DINOv2 ViT-S/14 on FGVC-Aircraft; CLIP RN50 for the zero-shot branch only |
| Linear probe | AdamW, lr 1e-3, wd 1e-4, batch 64, ≤200 epochs, checkpoint = best validation accuracy |
| Runs | 3 per training size (subset seeds at 5/10-shot, initialization seeds at full) |
| Metric | top-1 accuracy on the **complete official test split**, mean ± std over 3 runs |

‡ beyond the spec's required pair (DTD + FGVC-Aircraft); marked as such in every table. Group choices are recorded in [docs/adr/0005-group-choices-within-the-spec.md](docs/adr/0005-group-choices-within-the-spec.md).

## Environment

All code runs with the pre-built ROCm virtual environment (Python 3.12, torch 2.9.1+rocm, AMD RX 7900 XTX):

```
C:\Users\Alon\Desktop\cv-ex2\rocm_win312\Scripts\python.exe
```

**Do not pip-install into this venv** — it holds a hand-built ROCm-on-Windows torch stack. `tasks.ps1` wires the interpreter automatically.

## Usage

```powershell
.\tasks.ps1 setup     # GPU smoke test + runtime provenance dump
.\tasks.ps1 data      # download datasets, verify official splits, build K-shot subsets
.\tasks.ps1 extract   # cache train/val/test features for every (dataset, encoder) + CLIP text prototypes
.\tasks.ps1 run       # full experiment grid -> results/metrics/
.\tasks.ps1 smoke     # same grid with a few epochs, for a fast pipeline check
.\tasks.ps1 tables    # accuracy table -> results/metrics/accuracy_table.md
.\tasks.ps1 figures   # all figures -> results/figures/
.\tasks.ps1 notebook  # build + execute the presentation notebook
.\tasks.ps1 check     # repro check: re-derive every summary number from raw artifacts
```

`extract`, `run` and `smoke` accept dataset names to restrict the work, e.g. `python scripts/run_experiments.py dtd`.

## Repository layout

| Path | Purpose |
|---|---|
| [config/config.json](config/config.json) | All tunables: datasets, encoders, K values, seeds, probe hyperparameters, prompts |
| [src/](src/) | Library code: data, embeddings, classifiers, evaluation, visualization |
| [scripts/](scripts/) | Entry points (prepare, extract, run, tables, figures, repro check) |
| [notebooks/](notebooks/) | Modular notebook sections + builder → presentation notebook |
| [docs/REPORT_STAGE1.md](docs/REPORT_STAGE1.md) | Formal Stage 1 report |
| [docs/LAB_NOTEBOOK.md](docs/LAB_NOTEBOOK.md) | Chronological log: every step, command, error and fix |
| [docs/adr/](docs/adr/) | Decision records (0004 = spec adoption, 0005 = group choices) |
| results/features/ | Cached frozen features (gitignored, reproducible via `extract`) |
| results/artifacts/ | K-shot subset indices, CLIP text prototypes, probe training curves, predictions |
| results/metrics/, results/figures/ | Committed experiment outputs |

Binding project rules (environment, protocol, statistics, review gates) live in [CLAUDE.md](CLAUDE.md).
