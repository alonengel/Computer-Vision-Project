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

## 2026-07-17 — Full runs, figures, notebook, report

**Full experiment grid** (600 episodes / 10 seeds) ran clean off cached embeddings in a few minutes. Headlines: zero-shot CLIP beats every CLIP-embedding head on natural-image episodic tasks (CIFAR-10 93.2% 5-shot / 93.4% 1-shot, Mini-ImageNet 99.1%; only label-contaminated ResNet-50 prototypes nominally exceed it) but collapses on MNIST (48–60%); linear probe beats prototype at every dataset × K — gap grows with K on MNIST (+3.40→+8.59) and shrinks on CIFAR-10/Mini-ImageNet (+3.94→+1.66, +2.26→+0.56, paired CIs); prompt ensemble *hurts* MNIST; ResNet-50's 97.4% 1-shot Mini-ImageNet flagged as ImageNet-label-overlap, not few-shot skill; its 18.7-pt cosine-vs-Euclidean gap at K=1 supports cosine-primary.

**repro_check: PASSED** — 100/100 table rows re-derived exactly from raw per-episode/per-seed artifacts.

**All 19 figures regenerated** from full results (t-SNE glyph fix included). **Presentation notebook** built from nb_sections (17 cells) and executed end-to-end via nbconvert without error. **Report** §3 (results tables, generated programmatically from the CSVs — no manual transcription) and §4 (discussion) filled in.

## 2026-07-17 — Adversarial review (critical-reviewer) + final corrections

Final adversarial pass over all deliverables. **Numeric pipeline verified watertight**: all 100 CSV rows and 96 report table cells matched independent recomputation from raw artifacts (atol 1e-9); paired diffs, the 18.7-pt gap, ensemble deltas and zero-shot numbers reproduced exactly; all episode files re-verified leakage-free (zero support∩query overlap, correct fingerprints, K-per-class balance); sampler determinism confirmed by regenerating committed episode tensors `torch.equal`; no test-split prototypes; repro_check re-ran green.

**Two discussion overclaims caught and corrected** (wording errors under our own paired-CI standard):
1. "gap growing in K" was true only for MNIST — CIFAR-10/Mini-ImageNet gaps *shrink* (+3.94→+1.66, +2.26→+0.56); rewritten with 1-shot paired numbers.
2. "zero-shot beats every support-based head" — falsified by (label-contaminated) ResNet-50 prototypes at 5-shot Mini-ImageNet (99.51 vs 99.10, paired +0.40±0.11); claim scoped to CLIP-embedding heads with the contamination caveat inline.

Minor fixes: 93.4% quote correctly attributed to 1-shot episodes; published-CLIP comparison softened to "consistent with ≈89–90%" with citations; §2.5 note on CPU/GPU float-tie non-bit-exactness (committed arrays are canonical). Notebook s6 discussion aligned with the corrected claims and rebuilt.

## 2026-07-17 — Presentation-quality review (cv-expert) + polish pass; pushed to GitHub

Ran the deferred cv-expert *presentation* review (all 19 figures inspected visually, report prose, notebook narrative). Verdict: minor revision. Applied in full:
- **Figures**: global method→(display name, color, marker, linestyle) map in `src/visualize.py` — raw pipeline IDs (`proto_cos__clip_vitb32`) no longer appear in any figure text; legends moved outside the axes; colors now consistent across every figure; CVD-safe via marker/linestyle/hatch differentiation; dataset display names (MNIST, not `mnist`) everywhere; confusion titles note "seed 0" + colorbar label; CLIP zero-shot panels now show *least confident* predictions with a labeled probability axis; failure galleries show top-confidence error per (true, pred) pair; t-SNE legend rebuilt with proxy handles (normal-size star) centered under panels.
- **Report**: added Abstract, Author line, References; figures embedded as numbered Figures 1–9 with takeaway captions and cited by number; per-column best bolded († = ImageNet-contaminated ResNet-50 Mini-ImageNet entries); paired-differences table added to §4 (1-shot values verified against raw arrays: MNIST +3.40±0.29, CIFAR-10 +3.94±0.34, Mini-ImageNet +2.26±0.30); §2.4b renumbered to §2.5; grammar/register fixes.
- **Notebook**: display-name mapping applied to result tables; one-line "what to look at" narrative cells added before each figure block (now 20 cells); rebuilt and executed end-to-end.

Also this session: margin-based hardest-failure selection implemented (was first-12); HF_TOKEN configured (user-level env var); repo pushed to private GitHub `alonengel/Computer-Vision-Project`.

## 2026-07-17 — CLAUDE.md + probe equivalence/speed benchmark

**CLAUDE.md** added: binding rules for future sessions (venv, ADRs, statistics standards, figure style maps, review gates).

**Benchmark: sequential vs batched probe training** (`scripts/bench_probe.py`, user-requested to document the optimization). First run exposed a subtle flaw in my own check: predictions matched only ~98–99% because the sequential reference drew *different* per-episode inits from the shared RNG stream (accuracy deltas ≤ 0.07 pts — converged solutions differ microscopically, near-tie queries flip). Fixed by slicing the identical seeded `[600, C, D]` init draw for both paths (optional `W0` arg on `LinearProbe.fit`; default path values unchanged). Second run, all 6 configs:

```
identical predictions 45,000/45,000 in every config
accuracies match committed episodic.csv linear rows exactly
speedup 373–455x | grid total: 9.1 min sequential -> 1.3 s batched
```

Cause of the gap: kernel-launch overhead — 180,000 microscopic GPU steps (sequential) vs 300 steps on [600, C, 512] tensors (batched). Documented as §6 engineering benchmark + Figure 10 in the report; results in `results/metrics/bench_probe.json`.

**Known Windows/ROCm quirks carried over from cv-ex2** (guards already in place):
- `KMP_DUPLICATE_LIB_OK=TRUE` before torch import — otherwise the `clip` package triggers an OpenMP duplicate-runtime crash.
- CLIP model forced to `.float()` — fp16 weights misbehave on the ROCm stack.
- SSL unverified-context fallback available for dataset downloads (`utils.allow_insecure_downloads`).
- Free GPU memory (`torch.cuda.empty_cache()`) between backbones during extraction.
