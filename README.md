# Few-Shot Classification with Flow Matching

Course project in four stages (see [_docs/bigPicture.txt](_docs/bigPicture.txt)):

1. **Stage 1 (this repo state)** — Few-shot classification baselines on MNIST, CIFAR-10 and Mini-ImageNet using frozen pretrained embeddings: prototype classifier, `nn.Linear` probe (CrossEntropyLoss), and zero-shot CLIP.
2. **Stage 2** — Flow Matching as the final decision layer (embedding → class prototype / CLIP text embedding transport).
3. **Stage 3** — Flow Matching between the frozen encoder and the linear classifier, trained end-to-end.

Stage 1 deliberately produces **reusable artifacts** — cached embeddings, class prototypes, CLIP text embeddings, and fixed episode index files — so later stages compare against identical support/query sets.

## Environment

All code runs with the pre-built ROCm virtual environment (Python 3.12, torch 2.9.1+rocm, AMD RX 7900 XTX):

```
C:\Users\Alon\Desktop\cv-ex2\rocm_win312\Scripts\python.exe
```

**Do not pip-install into this venv** — it holds a hand-built ROCm-on-Windows torch stack. `tasks.ps1` wires the interpreter automatically.

## Usage

```powershell
.\tasks.ps1 setup     # GPU smoke test + runtime provenance dump
.\tasks.ps1 extract   # cache all dataset × backbone embeddings + CLIP text embeddings
.\tasks.ps1 run       # full experiment grid -> results/metrics/
.\tasks.ps1 figures   # regenerate all figures -> results/figures/
.\tasks.ps1 notebook  # build + execute the presentation notebook
.\tasks.ps1 check     # repro check: re-derive headline numbers from saved artifacts
```

## Repository layout

| Path | Purpose |
|---|---|
| [config/config.json](config/config.json) | All tunables: paths, datasets, backbones, episode grid, seeds |
| [src/](src/) | Library code: data, embeddings, classifiers, evaluation, visualization |
| [scripts/](scripts/) | Entry points (extract, run, figures, repro check) |
| [notebooks/](notebooks/) | Modular notebook sections + builder → presentation notebook |
| [docs/LAB_NOTEBOOK.md](docs/LAB_NOTEBOOK.md) | Chronological log: every step, command, error and fix |
| [docs/REPORT_STAGE1.md](docs/REPORT_STAGE1.md) | Formal Stage 1 report |
| [docs/adr/](docs/adr/) | Short decision records |
| results/features/ | Cached embeddings (gitignored, reproducible via `extract`) |
| [results/artifacts/](results/artifacts/) | Prototypes, CLIP text embeddings, episode indices (stage 2/3 inputs) |
| [results/metrics/](results/metrics/), [results/figures/](results/figures/) | Committed experiment outputs |

## Evaluation protocols

- **Episodic**: 5-way, K ∈ {1, 5}, 15 queries, 600 episodes, mean accuracy ± 95% CI. Mini-ImageNet test classes (Ravi & Larochelle split); also run on MNIST/CIFAR-10 for comparison.
- **Simple K-shot**: all classes, K ∈ {1, 5, 10} support per class, 10 seeds, full test set, mean ± std.

Episode support/query indices are generated once per (dataset, protocol, seed) and saved under `results/artifacts/episodes/` — stages 2–3 must load the same files.
