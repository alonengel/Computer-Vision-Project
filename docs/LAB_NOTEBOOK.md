# Lab Notebook — Stage 1

Chronological record of every step: what was done, why, exact commands, errors and fixes.

---

## 2026-07-17 — Repo moved to D:, scaffold

**Repo relocation.** The project directory was moved from `C:\Users\Alon\Desktop\Computer-Vision-Project` to `D:\Computer-Vision-Project` (361 GB free on D:) so datasets and cached features can live next to the code. All code uses paths relative to the repo root (`src/utils.py:repo_path`), so nothing else changed.

**Environment verification.** Reusing the prebuilt ROCm venv (never pip-install into it):

```
C:\Users\Alon\Desktop\cv-ex2\rocm_win312\Scripts\python.exe -c "import torch; ..."
→ Python 3.12.3 | torch 2.9.1+rocm7.2.1 | cuda available | AMD Radeon RX 7900 XTX
```

**Scaffold created.** `config/config.json` (all tunables), `src/utils.py` (seeding, device, ROCm guards, provenance dumps), `tasks.ps1` (one-word task runner pinned to the venv interpreter), README, .gitignore, review agents (`.claude/agents/cv-expert.md`, `critical-reviewer.md`), ADRs 0001–0002.

## 2026-07-17 — Data pipeline, sampler, classifiers, evaluation

**Mini-ImageNet source decision.** Compared HF candidates: `GATE-engine/mini_imagenet` has the R&L split sizes (38,400/9,600/12,000) but only bare integer labels — unusable for CLIP text prompts. Chose **`timm/mini-imagenet`** (100 wnid-labeled classes) and re-partition by *class* ourselves using the canonical Ravi & Larochelle 64/16/20 split lists (fetched from `yaoyao-liu/mini-imagenet-tools`, Ravi split CSVs) plus wnid→readable-name mapping from `tensorflow/models` `imagenet_metadata.txt`. Committed as `config/mini_imagenet_splits.json`. Verified 64/16/20 classes and sensible names (e.g. n02110341 → dalmatian).

**Code written.** `src/data.py` (pools + episode/support sampling, saved indices per ADR 0002), `src/embeddings.py` (3 frozen backbones + feature caching + CLIP text embeddings with prompt ensembles), `src/classifiers.py` (prototype cos/eucl, vectorized-Adam `nn.Linear` probe with CE, zero-shot CLIP), `src/evaluation.py` (episodic mean ± 95% CI, simple mean ± std, raw arrays saved for independent re-derivation), `src/visualize.py` (all figure types), scripts (prepare_data, extract_features, run_experiments, make_figures, repro_check).

**Synthetic sanity tests (no datasets)** — all passed:

```
sampler: deterministic, disjoint support/query, class-consistent
simple support: balanced K per class
proto_cos / proto_eucl / linear on separable Gaussians: 1.000
zero-shot with oracle text embeddings: 1.000
shuffled-label control: 0.193 (≈ 0.20 chance) ✓
```

**Downloads.** MNIST fast; CIFAR-10 mirror slow (~75 kB/s, ~35 min); Mini-ImageNet (HF) after that. `HF_HOME` pointed at `data/hf` so everything stays on D:.

## 2026-07-17 — Datasets downloaded, episode files fixed, sample grids rendered

**Download incident + fix.** Mini-ImageNet (HF, unauthenticated) stalled at shard 13/13 — 0-byte `.incomplete` file, no activity for 23 min. Killed the process and re-ran `scripts/prepare_data.py`; HF resumed from cache (5.8 GB kept), remaining ~1.5 GB completed normally. Lesson: set `HF_TOKEN` (no token was configured on this machine) for higher rate limits; a stall watchdog is worth arming for long unauthenticated downloads.

**Data verified.**
- MNIST 60k/10k, CIFAR-10 50k/10k (torchvision, on D:).
- Mini-ImageNet via `timm/mini-imagenet` re-partitioned by the canonical R&L class split: test pool = 13,000 images, 20 classes, exactly 650 per class.

**Episode/support files (ADR 0002): 66 files** under `results/artifacts/episodes/` — episodic 5-way K∈{1,5} seed 42 for all 3 datasets; simple-protocol support sets K∈{1,5,10} × seeds 0–9 for MNIST/CIFAR-10. Committed to git.

**Figures.** `episode_grid_{mnist,cifar10,mini_imagenet}.png` — 5×(support|query) grids, visually verified (correct classes, disjoint sets).

## 2026-07-17 — cv-expert methodology review (pre-launch) + fixes

Ran the methodology review agent on the full design *before* the expensive runs, per plan. Verdict: **minor revision** — 4 MAJOR findings, all fixed the same day:

1. **Linear-probe batched loss mis-scaled** — mean-reduction CE divided per-episode gradients by B=600, so "identical to independent heads" was false (Adam eps floor reached 600× sooner; weight decay would break equivalence badly). *Fix:* `reduction="sum"/S` — each head now gets exactly its independent-head mean-CE gradient. Also: `predict()` always refits (no stale-weight guard), docstring/report claims corrected, non-default init documented.
2. **Silent label corruption on episode/feature misalignment** — `remap` argmax mapped unknown labels to class 0 silently; `timm/mini-imagenet` was unpinned, so an upstream revision change would reorder the pool and produce plausible garbage. *Fix:* HF revision pinned (`bd8779f9`), `pool_fingerprint` (length + sha256 of labels) stored in episode/support files and asserted against feature caches at evaluation time, remap now asserts every label is in the episode class set.
3. **Hyperparameter provenance undocumented.** *Fix:* new §2.4b in the report — all hyperparameters fixed a priori (probe budget from standard linear-probe practice, prompts from the published OpenAI CLIP lists, cosine-primary as modern default), nothing tuned on any test metric; Mini-ImageNet val classes untouched in Stage 1.
4. **Test-split prototypes cached as stage-2 FM targets = future leakage.** *Fix:* ADR 0003 — permissible stage-2 targets are per-episode support prototypes or CLIP text embeddings only; full-split prototypes now cached from *train* splits only (mnist/cifar10).

Minor findings also applied: classifier names `proto_cos__clip_vitb32` (no brackets in filenames), zero-shot reported once per dataset in the simple protocol (K/seed-independent, so duplicated "± 0.00" rows dropped; drawn as reference lines in figures), config↔episode-file asserts, CLIP text-cache template check, paired-statistics commitment in §2.3, protocol asymmetry (episodic support from test split) stated explicitly, repro_check handles single-value rows.

Synthetic test suite re-run after all fixes: **all passed** (identical results, incl. chance control 0.193).

## 2026-07-17 — Features cached, fingerprints stamped, smoke run green

**Extraction complete** (one stall-free resume): 15 feature caches (3 backbones × 5 splits, ~1.6 GB), CLIP text embeddings ×3 datasets, prototypes from **train splits only** (test-split prototype files produced by the pre-fix script deleted per ADR 0003). Note: HF download stalled once mid-extraction earlier in the day; resume-from-cache worked as designed.

**Fingerprints**: all 66 episode/support files stamped with their pool fingerprint (from cache labels, which extraction asserts equal to pool labels) after validating every index in range and every episode label ∈ its episode class set.

**Smoke run (20 episodes / 2 seeds) — all sanity targets hit:**
- CLIP zero-shot CIFAR-10 all-classes: 88.3 / 88.8 (ens) % — published ≈ 89% ✓
- CLIP zero-shot MNIST: ~48% — expected weakness, planned discussion point ✓
- All episodic accuracies ≫ 20% chance; probe > prototype at K=5 on MNIST ✓
- **Finding for the report:** ResNet-50 prototype hits 96.7% on 1-shot Mini-ImageNet — the R&L test classes are ImageNet-1k classes, so supervised ResNet-50 saw them (labeled) during pretraining. Not few-shot-fair for that backbone; must be flagged in the backbone-comparison discussion.

**t-SNE first look**: CLIP/DINOv2 cluster CIFAR-10 cleanly; ResNet-50 visibly smeared — consistent with the smoke accuracies. (Fixed a missing-glyph ★ in the suptitle.)

**Known Windows/ROCm quirks carried over from cv-ex2** (guards already in place):
- `KMP_DUPLICATE_LIB_OK=TRUE` before torch import — otherwise the `clip` package triggers an OpenMP duplicate-runtime crash.
- CLIP model forced to `.float()` — fp16 weights misbehave on the ROCm stack.
- SSL unverified-context fallback available for dataset downloads (`utils.allow_insecure_downloads`).
- Free GPU memory (`torch.cuda.empty_cache()`) between backbones during extraction.
