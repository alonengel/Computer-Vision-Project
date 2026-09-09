# CVLAB Summer Project — Classification Baselines and Flow Matching

Three-stage course project. The professor's PDFs are the single source of truth per stage: `_docs/stage_1.pdf` (ADR 0004), `_docs/stage_2.pdf` (ADR 0007), `_docs/stage_3.pdf` (ADR 0008). **All three stages are complete.**

1. **Stage 1** — a reproducible classification pipeline on **frozen** pretrained encoders: a linear probe (required) plus both prototype branches (image-derived class prototypes and zero-shot CLIP), on DTD, FGVC-Aircraft and Oxford Flowers-102 ‡, K ∈ {5, 10, full}.
2. **Stage 2** — Flow Matching as the decision layer: transport frozen features toward the class prototypes (standard vs rolled-out training, T ∈ {4, 12}), compared against the Stage-1 prototype baseline on identical subsets/seeds; + the CLIP‑text branch ‡ and a full raw-feature comparison grid.
3. **Stage 3** — an FM transformation before the **frozen** Stage-1 linear probe (`z → FM → ẑ → frozen probe`), initialized to exact identity; end-to-end rolled-out CE training vs classifier-guided-target training, against the pinned probe; + an optional joint fine-tuning extension with a classifier-only attribution control.

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

‡ beyond our selected dataset pair — the specification allows any two of the three; we selected DTD + FGVC-Aircraft and additionally ran the third, marked as such in every table. Group choices are recorded in [docs/adr/0005-group-choices-within-the-spec.md](docs/adr/0005-group-choices-within-the-spec.md).

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
.\tasks.ps1 tests     # unit tests (FM forward/reverse transport)

# Stage 2:  run2 / run2raw / smoke2 / tables2 / figures2 / notebook2
# Stage 3:  run3 / smoke3 / tables3 / figures3 / notebook3
#           (+ scripts/run_stage3_joint.py for the optional joint extension)
# Stage 3 talk notebook (presentation-focused companion of stage3_presentation.ipynb; same tables of record):
#           python notebooks/build_talk_notebook.py
#           python -m jupyter nbconvert --to notebook --execute --inplace notebooks/stage3_presentation_talk.ipynb
# Stage 3 talk notebook v2 (English-only presentation; private Hebrew notes: notebooks/stage3_speaker_notes_he_v2.md):
#           python notebooks/build_talk_notebook.py v2
#           python -m jupyter nbconvert --to notebook --execute --inplace notebooks/stage3_presentation_talk_v2.ipynb
```

`extract`, `run` and `smoke` accept dataset names to restrict the work, e.g. `python scripts/run_experiments.py dtd`.

## Repository layout

| Path | Purpose |
|---|---|
| [config/config.json](config/config.json) | All tunables: datasets, encoders, K values, seeds, probe hyperparameters, prompts |
| [src/](src/) | Library code: data, embeddings, classifiers, evaluation, visualization |
| [scripts/](scripts/) | Entry points (prepare, extract, run, tables, figures, repro check) |
| [notebooks/](notebooks/) | Modular notebook sections + builder → one presentation notebook per stage; `stage3_presentation_talk.ipynb` (built by `build_talk_notebook.py` from `nb3_talk_sections/`) is the first 25–30-min talk version of Stage 3 (collapsed Hebrew speaker notes inside); `stage3_presentation_talk_v2.ipynb` (built by `build_talk_notebook.py v2` from `nb3_talk_v2_sections/`) is the English-only 13-section presentation, with the private Hebrew preparation notes in `stage3_speaker_notes_he_v2.md` |
| [docs/REPORT_STAGE1.md](docs/REPORT_STAGE1.md) · [REPORT_STAGE2.md](docs/REPORT_STAGE2.md) · [REPORT_STAGE3.md](docs/REPORT_STAGE3.md) | Formal per-stage reports |
| [docs/LAB_NOTEBOOK.md](docs/LAB_NOTEBOOK.md) | Chronological log: every step, command, error and fix |
| [docs/adr/](docs/adr/) | Decision records (0004 spec adoption · 0005/0006 group choices · 0007 Stage-2 · 0008 Stage-3 pre-registration) |
| results/features/ | Cached frozen features (gitignored, reproducible via `extract`) |
| results/artifacts/ | K-shot subset indices, CLIP text prototypes, probe training curves, predictions |
| results/metrics/, results/figures/ | Committed experiment outputs |

Binding project rules (environment, protocol, statistics, review gates) live in [CLAUDE.md](CLAUDE.md).
