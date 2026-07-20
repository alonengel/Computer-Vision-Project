# Few-Shot Classification with Flow Matching

Course project in three stages (see [_docs/bigPicture.txt](_docs/bigPicture.txt)):

1. **Stage 1 (complete)** — Few-shot classification baselines on MNIST, CIFAR-10 and Mini-ImageNet over frozen pretrained embeddings (CLIP ViT-B/32, DINOv2 ViT-S/14, ResNet-50): prototype classifier (cosine/Euclidean + a multi-prototype k-means ablation), linear probe (CrossEntropyLoss), and zero-shot CLIP as a semantic reference baseline — every head evaluated on every backbone.
2. **Stage 2** — Flow Matching as the final decision layer (embedding → per-episode support prototype / CLIP text embedding transport, ADR 0003).
3. **Stage 3** — Flow Matching between the frozen encoder and the linear classifier, trained jointly with CE (encoder stays frozen).

Stage 1 produces **reusable artifacts** so later stages compare against identical support/query sets: cached embeddings, fixed episode index files (fingerprint-verified), train-split prototypes, CLIP text embeddings, and — crucially — [results/artifacts/best_baselines.json](results/artifacts/best_baselines.json): the **final embedding selection** per (dataset, classifier head), chosen on *validation* episodes only (Mini-ImageNet R&L val classes / train-split episodes), one configuration across all K, ties to the smaller embedding. A Stage-2/3 variant of a head counts as an improvement only if it beats *that* configuration, paired on the same test episodes.

Binding project rules (environment, integrity, statistics, review gates) live in [CLAUDE.md](CLAUDE.md).

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

Additional scripts (run with the venv interpreter): `scripts/select_baselines.py` (validation-based embedding selection → `best_baselines.json`), `scripts/embedding_metrics.py` (quantitative embedding quality), `scripts/bench_probe.py` (batched-vs-sequential probe equivalence benchmark), `scripts/make_architecture_figs.py` (architecture diagrams).

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
| [results/artifacts/](results/artifacts/) | Episode indices, train-split prototypes, CLIP text embeddings, `best_baselines.json` (Stage 2/3 inputs) |
| [results/metrics/](results/metrics/), [results/figures/](results/figures/) | Committed experiment outputs |

## Evaluation protocols

- **Episodic**: 5-way, K ∈ {1, 5}, 15 queries, 600 episodes, mean accuracy ± 95% CI. Mini-ImageNet test classes (Ravi & Larochelle split); also run on MNIST/CIFAR-10 for comparison.
- **Simple K-shot**: all classes, K ∈ {1, 5, 10} support per class, 10 seeds, full test set, mean ± std.

Episode support/query indices are generated once per (dataset, protocol, seed) and saved under `results/artifacts/episodes/` — stages 2–3 must load the same files.
