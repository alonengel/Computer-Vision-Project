# Lab Notebook — Stage 1

Chronological record of every step: what was done, why, exact commands, errors and fixes.

> **Note.** Entries dated 2026-07-17 to 2026-07-20 describe the **pre-specification** implementation (MNIST / CIFAR-10 / Mini-ImageNet, episodic few-shot protocol), archived at tag `stage1-v1-episodic`. From 2026-07-24 the course document `_docs/stage_1.pdf` is the source of truth (ADR 0004) and the experiment is a different one. History is kept, never rewritten.

---

## 2026-07-24 — Specification received: Stage 1 rebuilt from `_docs/stage_1.pdf`

The official assignment document arrived and supersedes the outline the project had been built from. It specifies a materially different experiment; the work was rebuilt rather than patched.

**Backup first** (before touching anything): git tag `stage1-v1-episodic` + branch `archive/stage1-v1-episodic`, both pushed to the remote, plus a physical copy at `D:\_backups\Computer-Vision-Project_stage1-v1-episodic_20260724` (130 MB, full `.git` history, excludes regenerable `data/` and `results/features/`). Verified the copy afterwards — robocopy's `/XD results\features` did not match a relative path, so the feature caches were copied and then pruned.

**What changed** (ADR 0004): datasets MNIST/CIFAR-10/Mini-ImageNet → DTD (partition 1) / FGVC-Aircraft (`variant`) / Flowers-102; sampled episodes → official train/val/test splits, never merged; K ∈ {1,5,10} episodic → K ∈ {5,10,full} images per class with balanced subsets (seeds 0,1,2); encoders CLIP ViT-B/32 + DINOv2 + ResNet-50 → ResNet-18 (all datasets) + DINOv2 ViT-S/14 (FGVC-Aircraft) + CLIP RN50 (zero-shot only); metric mean ± 95% CI over episodes → top-1 on the complete official test split, mean ± std over 3 runs; probe Adam/300 steps → AdamW 1e-3 / wd 1e-4 / batch 64 / ≤200 epochs with best-validation-accuracy checkpointing.

**Group choices** (ADR 0005, confirmed with the user): all three datasets (spec pair = DTD + FGVC-Aircraft, Flowers-102 marked ‡), **both** prototype branches so the Stage-2 branch can be chosen on evidence, DINOv2 on FGVC-Aircraft only.

**Rebuilt:** `config/config.json` (+ `config/flowers102_classes.json` — torchvision exposes no class names for Flowers-102 and the zero-shot prompts need them), `src/{data,embeddings,classifiers,evaluation,visualize}.py`, `scripts/{prepare_data,extract_features,run_experiments,make_tables,make_figures,repro_check}.py`, all notebook sections, README, CLAUDE.md, ADRs 0001/0004/0005 (0002 marked superseded, 0003 revised). Obsolete v1 artifacts removed from the working tree via `git rm` (`Remove-Item` is sandbox-blocked in the repo).

**Downloads.** DTD verified exactly as expected: 1880/1880/1880 images, 47 classes, 40 per class in every split. FGVC-Aircraft (2.75 GB) and Flowers-102 ran in the background at ~0.6–1.7 MB/s.

**Pipeline validated on DTD before the other datasets finished downloading** — added dataset filters to `extract_features.py` / `run_experiments.py` for this. Full DTD grid (9 probe runs × 200 epochs + 7 prototype runs + zero-shot) takes **32 seconds**:

```
linear probe    5-shot 45.60 ± 0.64 | 10-shot 54.31 ± 0.28 | full 62.84 ± 0.45
image prototype 5-shot 46.51 ± 0.64 | 10-shot 53.37 ± 1.04 | full 58.78
zero-shot CLIP RN50                                          39.79
```

Sanity: published ResNet-18 linear probe on DTD is ≈ 62–67% and CLIP RN50 zero-shot DTD ≈ 41.7% with prompt ensembles (we use the spec's single prompt) — both in range. The training curves show the textbook picture the spec asks for: training loss → 0 while validation loss bottoms near epoch 30 and then rises, with the checkpoint landing at the validation minimum. `repro_check` green.

**Fixes during validation:** zero-shot was being printed in the "full train split" column of the accuracy table (misleading — it uses no training images); it now has its own "no training images" column. Seaborn's `whitegrid` theme drew gridlines over the confusion-matrix and feature-projection images; `ax.grid(False)` added.

**All datasets verified** against their official splits: DTD 1880/1880/1880 (47 classes, 40 per class), FGVC-Aircraft 3334/3333/3333 (100 classes, 33–34 per class), Flowers-102 1020/1020/6149 (102 classes, exactly 10 train per class; the test split is class-imbalanced 20–238, as published). 18 balanced K-shot subset files committed. As predicted, the Flowers-102 K=10 subset contains 1020 images — the entire training split — so its 10-shot and full settings coincide and its three 10-shot runs are identical by construction.

**Feature extraction:** 21 caches (3 datasets × 3 splits × their encoders) + 3 CLIP text-prototype files, ~46k images through the encoders.

**Full experiment grid: 2.1 minutes** (36 linear-probe runs at ≤200 epochs, 28 prototype runs, 3 zero-shot). `repro_check` green on all 27 summary rows.

Headline numbers (top-1, complete official test split):

```
DTD            probe 45.60±0.64 / 54.31±0.28 / 62.84±0.45   proto 46.51±0.64 / 53.37±1.04 / 58.78   zero-shot 39.79
FGVC ResNet-18 probe 19.90±1.72 / 27.36±0.83 / 36.62±0.27   proto 16.04±0.83 / 19.85±0.47 / 25.20   zero-shot 17.04
FGVC DINOv2    probe 36.47±0.70 / 51.30±0.95 / 67.21±0.11   proto 23.11±1.36 / 27.80±1.17 / 34.41
Flowers-102 ‡  probe 75.58±0.84 / 83.22±0.00 / 83.28±0.11   proto 70.19±0.14 / 75.22±0.00 / 75.22   zero-shot 63.64
```

External sanity: CLIP RN50 zero-shot is commonly reported at ≈41.7 / 19.3 / 65.9 for these datasets; ours (39.8 / 17.0 / 63.6) sits just below, consistent with the spec's single prompt vs a published prompt ensemble.

**Findings.** (1) The encoder dominates the head — DINOv2 lifts the FGVC-Aircraft full-split probe by 30.6 points over ResNet-18, and the DINOv2 probe with 5 images/class (36.47) matches the ResNet-18 probe on the entire split (36.62). (2) Image prototypes beat the probe at exactly one grid cell (DTD K=5) and lose by a widening margin elsewhere. (3) Training curves show the probe overfitting at every training size, absorbed by validation-accuracy checkpointing; on FGVC/DINOv2 the validation *loss* rises from ~epoch 25 while best validation *accuracy* lands at epoch 187 — loss and accuracy are not interchangeable selection criteria.

**Is the 200-epoch budget adequate?** 8 of 36 probe runs peak at epoch ≥190, so the cap mildly binds. Measured from the saved curves, validation-accuracy gain over the last 100 epochs is +0.63 / +0.81 / +0.98 points — within ~1 point of plateau and comparable to run-to-run spread. The spec's suggested configuration was therefore **kept unchanged**, and this check is reported rather than silently ignored.

## 2026-07-24 — Two review passes and the fixes they forced

Ran the spec-compliance review (cv-expert) and the adversarial number/leakage review (critical-reviewer) in parallel.

**Compliance:** all 47 requirements enumerated from the PDF are satisfied — verdict "compliant with minor gaps", nothing required missing. The adversarial pass independently reproduced all 27 summary rows, all 27 accuracy-table cells, and re-implemented the prototype and zero-shot classifiers from the caches: DTD prototype full = 58.7766 exactly, zero-shot 39.7872 / 17.0417 / 63.6364 exactly. It also confirmed the prototype normalization order is the specified one by showing the two plausible wrong orderings give different answers (59.0957 and 51.5426).

**Real defects found and fixed:**
1. **Factual numeric error.** §4 said zero-shot on Flowers-102 "lands within 12 points of a ResNet-18 probe trained on the full split". The gap to the *full* probe is 19.6 points; 12 is the gap to the *5-shot* probe. Corrected, and a speculative clause about "a couple of images per class" (no K<5 run exists) removed.
2. **Overclaim at DTD K=10.** "Beaten everywhere else" treated a cell that is statistically indistinguishable at n=3 (paired −0.94 ± 0.83; per-seed −1.60 / 0.00 / −1.22, one exact tie) as a clean loss. Now stated as indistinguishable.
3. **No paired statistics anywhere**, despite the project's own rule. Added `paired_heads_table.md` (generated): the one prototype win, DTD K=5, is **+0.90 ± 0.14 with 3/3 seeds agreeing** — far stronger than the overlapping marginal intervals it was previously argued from.
4. **Overfitting claim over-generalized.** Curves are saved only for K=10, and Flowers-102/ResNet-18 does *not* overfit (validation loss decreases monotonically to epoch 199). Scoped to the three combinations that actually overfit.
5. **The `pool_fingerprint` guard could not do what it claimed.** DTD's three partitions and all three splits have byte-identical label sequences (40 per class, sorted), so a partition change passed the assertion silently. Fingerprint now hashes image identities as well: dtd train/val/test are `LF…4d95 / af7f / 0eae` and partition 2 gives `7a04`, all distinct. All 18 subset files upgraded **in place with indices untouched**, and 21 feature caches stamped; re-ran the grid and every number is unchanged.
6. **"No encoder was ever chosen by looking at test accuracy" was false** — the confusion-matrix encoder was picked by test accuracy. Now picked by validation accuracy, and the two presentation decisions genuinely made after the test read-out (which matrix to show, the Stage-2 branch recommendation) are flagged rather than hidden; validation gives the same ordering (DINOv2 68.62 vs ResNet-18 38.36 on FGVC), so the recommendation does not depend on test.
7. **Caption/labelling errors:** "the DINOv2 5-shot probe *equals* the ResNet-18 full probe" overstated a 0.15-point deficit (now "matches within run-to-run noise"); Figure 10 attributed the 30.6-point *probe* gap to a *prototype* figure (the prototype gap is 9.2); "recovers barely half" holds for DINOv2 but not ResNet-18 (68.8%); confusion titles quoted the 3-seed mean over a seed-0 matrix; one of six DTD confusions was omitted; figure subtitles promised error bars on single-run points.
8. **Table bolding was hand-applied and inconsistent** (DTD K=10's winner unmarked) — moved into `make_tables.py` and defined in the caption.
9. **repro_check was shallower than advertised** — extended to also verify all 27 prediction files against `runs.csv` and to regenerate the three markdown tables from artifacts. Now: 27 summary rows + 27 prediction files + 3 tables, all green.
10. **Notebook self-containment:** added the Stage-2 branch decision (previously only in the report), paired statistics, the epoch-budget check, the published-CLIP sanity comparison, limitations, the deviations table, per-figure takeaways, and display-name mapping for the last table that still printed raw pipeline IDs. Now 30 cells.

Also: training curves now plot validation **accuracy** on a twin axis, since the report's sharpest curve claim is about accuracy diverging from loss; feature plots switched to a colourblind-safe palette with varying marker shapes; the CLIP modality gap is explained where the CLIP panels appear; DTD confusion axes use class names; duplicate ADR 0001 file removed; ADR 0004's "removed from the working tree" claim narrowed to tracked files (the gitignored v1 feature caches remain on local disk).

**One reported finding was a false positive:** the compliance reviewer flagged CLAUDE.md as still describing the episodic v1 protocol. The file on disk had already been rewritten; the agent was reading the stale copy cached in its own system prompt from session start. Verified directly — no occurrences of `episodes`, `best_baselines`, `95% CI` or `mini-imagenet` remain.

## 2026-07-26 — Notebook review pass: projections, wording, validation-based handoff

User-directed review of `stage1_presentation.ipynb` against the spec. Audited the PCA/t-SNE pipeline first: L2 normalization ✓, label/prototype alignment ✓ (sorted-class remap, prototypes indexed identically, palette by position), fixed seeds ✓ (`seed=0`, `class_seed=0`), identical classes/examples/colours across encoders ✓ (`viz_selection` runs once per dataset), joint fitting ✓. Two gaps fixed: t-SNE parameters were only *implicitly* identical (the perplexity formula evaluated to 30 for every panel) — now a single explicit `TSNE_PARAMS` constant with a guard assert — and the CLIP text-prototype class order was asserted in `run_experiments` but not at figure time — assert added in `feature_charts` (passes for all three datasets).

**Presentation changes** (no numbers touched):
- PCA is now labelled and framed as the **primary** view (deterministic, linear, globally interpretable); t-SNE as **supplementary**, read for local neighbourhoods only — panel titles, figure subtitles, notebook §5.5 and report §3.5 all updated.
- Modality-gap wording softened: the text-prototype/image separation is "*consistent with* the modality gap (Liang et al., 2022)" — a 2-D projection cannot establish it alone; confirmation would need distances in the original 1024-d space. (The separation does appear in the PCA view too, not only t-SNE.)
- "The spec's required pair" → "**our selected pair**" everywhere (README, report, ADR 0005, notebook, generated table footnotes) — the specification allows any two of the three datasets.

**Branch selection cleaned of test dependence.** `run_experiments.py` now records the prototype head's **validation** accuracy per run (deterministic, no training; test numbers verified byte-identical before/after the rerun). The Stage-2 branch decision is restated in report §4 and notebook §6 as resting on three methodological reasons plus validation-measured headroom: probe − prototype on validation (full split) = 5.3 pts (DTD/ResNet-18), **35.7 pts** (FGVC-Aircraft/DINOv2), 8.5 pts (Flowers-102/ResNet-18). Test accuracy plays no role in the selection.

**New final handoff table** (`scripts/make_tables.py handoff_table()` → `results/metrics/handoff_table.md`, displayed as notebook §7 and in report §4): per dataset — encoder selected by validation (DTD → ResNet-18; FGVC-Aircraft → DINOv2; Flowers ‡ → ResNet-18), Stage-2 prototype target (class-mean μ_c of the selected training subset on that encoder), validation headroom, and the Stage-3 linear-probe baseline as the one-time test read-out (62.84 ± 0.45 / 67.21 ± 0.11 / 83.28 ± 0.11). Caught and fixed a bug in the first version: the Stage-3 baseline column ignored the selected encoder and showed FGVC's ResNet-18 probe (36.62) instead of DINOv2's (67.21).

**Verification:** notebook rebuilt — 34 cells, all 15 code cells executed, zero error outputs; feature figures regenerated (class-order asserts green); extended repro check green (27 summary rows + 27 prediction files + 4 generated tables); `git diff` on the metrics confirms the only change beyond the new `handoff_table.md` and the proto `val_acc` column is one footnote sentence — **every reported number is unchanged**.

## 2026-08-03 — Dataset pair confirmed post-results: DTD + FGVC-Aircraft (ADR 0006)

User decision after re-reading `_docs/stage_1.pdf`'s degrees of freedom (two datasets, one prototype branch): **keep the full three-dataset / two-branch superset in the Stage-1 deliverable, but make both selections explicit, post-results decisions** rather than a-priori flags. ADR 0006 written recording both: the pair **DTD + FGVC-Aircraft** (ADR 0005's provisional `spec_selected` pair, now confirmed on evidence) and the branch **Option A** (already decided in report §4 / notebook §6; the ADR consolidates it into the decision trail).

The pair evidence is deliberately **structural, never a test-accuracy ranking** (avoids any selection-on-test smell): (i) Flowers-102's 10-images-per-class training split makes 10-shot ≡ full — visible in the runs as the deterministic prototype head scoring identically at both settings and the probe's three 10-shot "subset seeds" carrying exactly zero spread — so its accuracy-vs-training-size curve has two distinct points where the others have three, and that axis is what Stage 2 argues along; (ii) DTD/FGVC give three genuinely distinct K settings; (iii) FGVC carries DINOv2 + the 35.7-pt validation headroom; (iv) Flowers' class-imbalanced test split makes its top-1 the least clean single number.

**Where recorded:** ADR 0006 (+ status pointer in ADR 0005); report — §1 forward-pointer, new §4 paragraph "Which two datasets carry forward", §6 deviations row updated; notebook §6 — new "Which two datasets carry forward — decision" markdown + evidence cell (train imgs/class and the full−10-shot gaps / 10-shot spread from `dataset_splits.csv` + `summary.csv`, computed live, ResNet-18 rows for cross-dataset comparability), deviations row updated. Flowers-102 stays ‡ everywhere; no numbers, figures or config changed (`spec_selected` flags already encoded the pair).

## 2026-08-09 — Stage 2 spec received; FM infrastructure built, reviewed, grid launched

`_docs/stage_2.pdf` received (+ `fm_training.png` / `fm_inference.png` algorithm cards): add an FM decision layer that transports frozen features toward class prototypes; standard vs rolled-out training; T ∈ {4,12}; same datasets/prototypes/subsets/seeds/test sets as Stage 1; report ΔAcc vs the Stage-1 prototype baseline. User decisions: run **both** target branches (image prototypes = spec branch per ADR 0006 selection; CLIP text prototypes as marked ‡ extension — Stage 1 established both), datasets = the selected pair only (DTD + FGVC-Aircraft).

**ADR 0007 written before any training run**, fixing: FM operates on L2-normalized features (the cosine classifier's space; disclosed as a deviation from the literal spec's raw-feature ẑ₀ = z); prototypes recomputed per (K, seed) from the committed subset files with a **T=0 integrity guard** (no-transport classification must reproduce the Stage-1 baseline from runs.csv to 1e-6, asserted per setting before training); Stage-1 probe recipe verbatim (AdamW 1e-3 / wd 1e-4 / batch 64 / 200 epochs); final-epoch models, no validation checkpointing, with a pre-registered stability criterion (final loss ≤ 1.05× running min, uniform min-loss-epoch fallback); standard FM trained once per setting (T-independent) and evaluated at both T; rolled-out one model per T.

**Implementation:** `src/flow_matching.py` (VelocityMLP dim+1→512→512→dim SiLU; FlowMatchingHead: standard + rollout fit, Euler transport with trajectory capture, cosine prediction, save/load), `scripts/run_stage2.py` (grid runner, per-seed-paired ΔAcc), `make_tables_stage2.py`, `make_figures_stage2.py` (acc-vs-K with baseline, loss curves + stability sweep, joint-PCA 3-panel feature comparison, flow trajectories in the same PCA plane), repro_check extended to every Stage-2 artifact incl. re-deriving baselines/deltas from Stage-1 runs.csv, `nb2_sections/` + parameterized `build_notebook.py 2`, config `stage2` block, tasks run2/smoke2/tables2/figures2/notebook2.

**cv-expert methodology review (pre-launch): minor revision.** Applied before launch: smoke curves were writing to real-run filenames (tag added — a smoke rerun would have silently clobbered real 200-epoch curves); `load_fm_head` now restores the saved architecture instead of trusting current config; summary gains `delta_std` (paired per-seed delta spread); ADR §7 contingency tightened to an exact pre-registered rule. Review also mandated: normalization deviation disclosed in report + notebook (done), CLIP‡ captions must state Δ ≠ "FM helps" (in table generator), repro-check coverage (done). Smoke run green (T=0 guard exact). Full grid (135 trainings) launched in background; interim first-grid readings (later superseded by the checkpoint-fallback re-run, see next entry): DTD/ResNet-18 standard FM ≈ baseline at full, below at 5/10-shot; FGVC/ResNet-18 5-shot positive (+2.3 standard). Infrastructure committed mid-grid (`cbceb53`) — git confirms the ADR 0007 §7 contingency text and `final_epoch` policy predate the completion of any full-split (subsequently-divergent) run.

## 2026-08-09 — Stage 2 contingency triggered, grid re-run, results finalized, both reviews passed

**Contingency.** The first grid's stability sweep (ADR 0007 §7, final loss ≤ 1.05× running min over all 135 curves) fired: 3 genuine divergences, all rolled-out at K = full — FGVC/CLIP‡ T12 seed 1 at **62×** its minimum (test accuracy collapsed to ~6%, recorded at trigger time), FGVC/ResNet-18 T4 seeds 0/1 at 2.9×/1.26× — plus 20 marginal 1.05–1.13× flags (18 standard-FM last-epoch noise from random-t resampling, 2 rolled-out near-threshold). As pre-registered: uniform fallback to **min-training-loss checkpointing for every model** (`checkpoint_selection: "min_train_loss"`; training-set-only criterion), full grid re-run (~1.5 h), first grid's numbers never published. The divergent cell went from 21.29 ± 13.25 to 28.64 ± 0.21.

**Final results (fallback grid; runs_stage2.csv, 180 rows, repro-checked).** FM gains track Stage-1 headroom exactly: FGVC/DINOv2 **+10.4..+23.2** (3/3 seeds everywhere; full 34.41 → 57.59); FGVC/ResNet-18 standard +1.6..+6.7 (3/3); DTD standard +0.8 at full but −2.4..−4.7 at 5/10-shot (0/3), rollout −7..−12.9. Standard ≥ rolled-out on the spec branch (DINOv2 K∈{5,10} a statistical tie); all divergences were rolled-out. CLIP‡: supervised transport beats zero-shot in every cell (+1.5..+23.9); rollout T4 doubles standard at FGVC full (+14.5 vs +6.9) — endpoint-only objective wins where the straight-line velocity is most misspecified (modality gap; trajectory figures show the curvature). Best FM-to-prototype (57.59) still 9.62 pts below the Stage-1 probe (67.21) on the same features.

**Deliverables.** Tables `stage2_{image_prototype,clip_text,paired_delta}_table.md` (‡ on all CLIP rows); 14 figures (acc-vs-K ×5, curves ×2, joint-PCA feature comparisons ×5 — one PCA per setting fitted jointly to all three feature sets + prototypes — trajectories ×5 in the same plane); `REPORT_STAGE2.md` (abstract → deviations, embedded generated tables); `stage2_presentation.ipynb` 27 cells executed clean, self-contained (protocol, hyperparameter provenance, normalization deviation, contingency story, CLIP‡ caveat, T=0 guard demo, live Stage-1 reference-value cell).

**Reviews.** cv-expert methodology review pre-launch (minor revision, applied). critical-reviewer adversarial pass post-results: every quoted number reproduced from artifacts; leakage checks clean; found stale final-epoch wording in notebook §6 + curve-figure captions (fixed, regenerated), abstract overclaim "beats throughout" (softened — DINOv2 K∈{5,10} is a tie with rollout nominally ahead), notebook self-containment gap for Stage-1 reference values (fixed with a live-computed cell), misattributed 2/20 marginal flags (fixed), FGVC range misquote (fixed), pre-registration wording tightened (commit was mid-grid; §7 text predates all divergent runs), repro_check now **content-compares** committed tables against regenerated ones instead of checking existence. `tasks.ps1 check` green end-to-end (27 + 60 summary rows, 27 + 60 prediction files, 7 tables content-verified).

## 2026-08-10 — Animated flow trajectories added to the Stage-2 notebook

User-requested (spec: `_docs/stage2_flow_animation_prompt.md`): an inline animation of the **actual learned transport** — no synthetic trajectories — added as `nb2_sections/s5b_flow_animation.py` immediately after the static trajectory section. Representative setting FGVC-Aircraft/DINOv2, image prototypes, K=10 seed 0, T=12, Standard vs Rolled-out in two synchronized panels. Implementation reuses `_rep_setting`/`_joint_pca` from `make_figures_stage2.py`, so the classes, four examples (737-700, A320, A340-200, DHC-1), colours and the jointly-fitted PCA plane are identical to the static figures **by construction**. Pre-display assertions verify: 13 Euler states per trajectory; first state = normalized pipeline input; final states match the transported features of the feature-comparison figure (atol 1e-5); both models share start features and prototypes; single PCA transform; no gradients (`transport()` is `no_grad`). `FuncAnimation` → inline `to_jshtml()` player (no ffmpeg) + `stage2_flow_animation_fgvc_dinov2_T12.gif` via PillowWriter; first/last frames held. Markdown caveats below the animation: transport happens in 384-d, PCA is qualitative, projected lengths/curvature are not measurements.

**Errors + fixes:** (1) mathtext legend labels raised `SyntaxWarning: invalid escape '\\h'` into the cell output — switched to raw strings, rebuilt; (2) suptitle/legend clipped in animation frames (animations render without `bbox_inches="tight"`) — moved all fixed text inside the canvas via `subplots_adjust`. Final state: notebook 30 cells, executed top-to-bottom, 0 errors/0 warnings; repro check green; git status confirms the only new artifact is the GIF — no existing table, metric, model or figure changed.

## 2026-08-10 — Animation reworked per user feedback: per-example zoom + velocity arrows, GIF embedding

User feedback on the first version: (a) the animation did not display in their notebook viewer — root cause: `to_jshtml()` outputs a JavaScript player, which VS Code's notebook renderer blocks; (b) one combined view of 4 points was too small to read — wanted **one example per animation, zoomed**; (c) wanted a **direction vector at every iteration**.

Rework (same section `s5b`, all integrity assertions unchanged): four separate two-panel (Standard | Rolled-out) animations, one per representative example, axis limits fixed per example to its two trajectories + its prototype (identical limits in both panels, equal aspect); a black quiver arrow at each frame shows the projected direction of the current Euler step $\\tfrac{1}{T}v_θ(\\hat z_k, k/T)$ (hidden at the final state); display switched from the JS player to **base64-embedded GIFs** (`<img src="data:image/gif...">`) — plays in VS Code and Jupyter with no JavaScript and no path dependence, with the JS player kept only as a fallback if PillowWriter fails. Old combined GIF `git rm`'d; new artifacts `stage2_flow_anim_fgvc_aircraft_dinov2_vits14_T12_ex{0..3}.gif`. Caveat text extended: the arrow is the *projection* of the true 384-d step. Notebook rebuilt + executed top-to-bottom: 30 cells, 0 errors/0 warnings, all `[OK]` verifications green; no existing artifact changed.

**Follow-up (same day) — class-palette separation.** User: the colourblind palette's near-duplicates are hard to tell apart (three orange-family, two pinks) — make one pink purple and one orange red. Added `class_palette()` in `src/visualize.py` as the **single source of truth** for class colours: colourblind base with `#D55E00` (vermillion) → `#D62728` (red) and `#CC78BC` (medium pink) → `#6A3D9A` (deep purple); `feature_projection`, `flow_trajectory_chart`, and the animation section all now use it, so class colours agree across every figure and animation of both stages (the spec's same-colours rule). Regenerated all class-coloured figures — Stage-1 `features_*.png` (7) and Stage-2 features/trajectories (10) + the four animation GIFs — and rebuilt + executed **both** notebooks (36 + 30 cells, 0 errors). Numbers untouched; repro check green.

**Second follow-up — remaining orange pair.** User caught that `#DE8F05` (orange) vs `#CA9161` (light tan) still read as two oranges. Third swap in `class_palette()`: `#CA9161` → `#8C564B` (dark brown — dark-vs-bright contrast with the orange). All 17 class-coloured figures + 4 GIFs regenerated, both notebooks rebuilt and executed (0 errors), repro check green.

## 2026-08-22 — External audit (user prompt) + teammate feedback: raw-feature control run, checkpoint-benignity evidence, prose corrections

**Audit** (`_docs/stage2_ai_audit_prompt.md`, executed by cv-expert against the PDF + executed notebook, with independent re-derivation): verdict *mostly ready with minor fixes*; 20/21 PDF requirements Pass (the optional reverse-flow exploration not attempted); no leakage, no protocol violation, no number failing re-derivation. Two checkable-but-wrong prose sentences found (both survived the earlier reviews): (1) "rollout nominally higher in 3 of 4 DINOv2 low-shot cells" — at matched T it is 2 of 4 (ahead at both T=12, behind at both T=4); (2) the blanket "paired spread far tighter than marginal spreads" — true on FGVC (±0.06 vs ±0.79/0.83), **inverted on DTD** where per-seed deltas anticorrelate (±1.61 vs ±0.62/1.04). Teammate feedback (relayed by user) overlapped: no raw-features experiment, min-train-loss could pick a "lucky" epoch, headroom claim is n=3, single-file notebook not re-runnable.

**Fixes applied (prose + evidence, no published number changed):**
- **Raw-vs-normalized control** (the audit's one real gap): `normalize=False` option in `FlowMatchingHead` (exists only for this control) + `scripts/run_stage2_raw_ablation.py` — seed-0 slice, 3 image-branch settings × K × {standard, rollout}, T=12, identical recipe/targets/subsets; only the input normalization differs. Result: **normalized ≥ raw in 18/18 settings**, gaps +4.89..+20.49 pts (median +10.19); at low K raw-FM often falls below the Stage-1 baseline itself. Table `stage2_raw_ablation.{csv,md}`, embedded in notebook §2 with a live-computed summary; ADR 0007 addendum; deviations rows added (control ‡ + optional-exploration-not-attempted).
- **Checkpoint benignity**: notebook §4 + report §3 now argue the "lucky epoch" objection and print the evidence — selection is training-data-only (no correlation with test noise); selected epochs sit at end of training (standard min 159 / median 193 / max 199; rollout min 88 = a diverged run / median 197); superseded final-epoch stable-run accuracies differ only by tenths (first-grid arrays preserved at `cbceb53`). The 3-vs-20 genuine/marginal split explicitly labelled post-hoc description.
- **Prose corrections** in notebook §3/§6 + report: matched-T count fixed; pairing claim now states the DTD anticorrelation; "+0.8 at full" → "+0.5 to +0.8"; "few tenths" → "< 0.7 pts"; headroom claim qualified as consistent-ordering-at-n=3; "45 assertions" count stated at the T=0 guard; animations note they show the success case and point to the DTD failure case; provenance paragraph added (repo is the unit of reproduction; standalone .ipynb is readable, not re-runnable — teammate's point).
- **Figure polish**: PC1/PC2 axis labels with explained-variance % on all Stage-2 feature/trajectory figures (joint-PCA variances now visible); rolled-out T=4/T=12 line-style differentiation + per-panel y-labels in the curve figures.

Notebook rebuilt + executed top-to-bottom: 32 cells, 0 errors/0 warnings; repro check green. Remaining instructor question (audit §9): whether normalized-input FM is acceptable as the primary formulation — the 18/18 control is the supporting evidence either way.

## 2026-08-22 — Two full versions: the complete raw-feature (literal-spec) grid

User request: not a control slice — **two complete versions**, with and without input normalization. Added `--raw` to `run_stage2.py` (tag `_raw`, `normalize=False`; identical protocol otherwise — every seed, both T, both branches, same prototypes/baselines/checkpoint rule) and re-ran the full 135-training grid on raw features (~1.5 h). The earlier seed-0 ablation (script + artifacts) was `git rm`'d as superseded; ADR 0007 addendum updated.

**Result — the two versions diverge exactly where the geometry says they should:** on the **image-prototype branch the normalized version is better in all 36/36 cells**, mostly by double digits (worst gap −25.3 on DINOv2 full rollout T4; DTD full standard 43.6 raw vs 59.6 normalized), 27/60 raw cells fall below their own Stage-1 baseline, and raw training is less stable (33 vs 23 curves trip the 1.05× criterion; raw DINOv2-full rollout spreads reach ±10.4). On the **CLIP‡ branch the versions are nearly equivalent** (|Δ| ≤ 4.6, mixed signs, raw slightly ahead in most rolled-out cells) — CLIP feature norms are far more uniform, so raw ≈ normalized there. Across all 60 cells: normalized ≥ raw in 45, median +8.2.

**Infrastructure:** `stage2_raw_*_table.md` generators (raw accuracy + per-cell raw−normalized difference); repro_check parametrized over variants — now verifies both grids end-to-end (summaries from raw arrays, deltas vs Stage-1, prediction files) + content-compares the raw tables; the published-grid stability sweeps exclude `_raw` curves (count stays 135); notebook §2 rewritten as "The two versions ‡" — both raw tables embedded, live-computed summary (45/60, median +8.17, 27 below baseline, 33/135 raw instability), honest branch-split reading; report §1 updated with the full-version numbers; tasks `run2raw`. Notebook: 33 cells, 0 errors/warnings; `tasks.ps1 check` green over both variants (2×60 summary rows, 2×180 runs, 2×60 prediction files, 5 tables content-verified).

## 2026-08-23 — Optional reverse-flow exploration implemented (user prompt, spec's optional item)

Per the user-supplied implementation prompt: the spec's optional "explore the learned flow in the reverse direction" is now done as a marked ‡ exploration — **no retraining, no change to any published number** (git status verified: only new artifacts + the rebuilt notebook).

**API:** `FlowMatchingHead.reverse_transport(Z1, T, return_traj=False)` — explicit reverse Euler `z_{k−1} = z_k − (1/T)·v_θ(z_k, k/T)`, k = T..1, first evaluation at t = 1 (canonical; the forward rollout never evaluates t = 1 — disclosed), input taken as an in-space state (prototypes are unit-norm; no re-normalization, matching forward), `no_grad`, deterministic; with `return_traj` returns [T+1] states + times in descending order [1, …, 0]. Documented as numerical backward integration of the learned *continuous* field — not the algebraic inverse of discrete forward Euler, not generative.

**Tests (new — repo previously had none; pytest absent from the ROCm venv, so plain-assert script):** `tests/test_flow_matching.py`, `tasks.ps1 tests` — 16/16: shapes, descending times, first state ≡ supplied prototype, finiteness, no gradients, determinism, zero-field invariance, constant-field endpoint z₁−c (+ forward re-adds c), forward-transport contract (normalized start, shapes, predict), and a forward-then-reverse reconstruction *diagnostic* (mean err 0.0034; deliberately not asserted — reverse Euler isn't the exact inverse).

**Second follow-up (user request): reverse on the CLIP‡ branch + clarify what the percentages are.** `reverse_charts` refactored to loop over both branches (artifacts renamed to include the target; old-named files `git rm`'d). **The CLIP result is the strongest finding of the section:** standard FM's reverse trajectories cross the image–text modality gap *backward* — from the isolated text prototypes into the image-feature cloud, landing on the class centroids at cosine **0.973 ± 0.006** (tightest agreement anywhere here) — and the reverse-endpoint classifier reaches **26.40%**, above that cell's forward FM (21.99%) and far above zero-shot CLIP (17.04%): backward integration converts each text prototype into a usable *image-space* class representative. (Supervised ‡ number — the flow used K=10 labels — not a zero-shot claim.) Rollout fails there too (cosine −0.06 ± 0.57, round-trip 6.27%, endpoints 3.45%). Also noted: the CLIP round trip sits slightly *below* baseline even for standard (16.89 vs 17.04) — the one-way backward map carries the value, not the there-and-back traversal. Percentage semantics now stated explicitly in §5c: all reverse Top-1 numbers are top-1 on the **complete** official test split from **single deterministic runs** (K=10, seed 0) — **no ± spread, not averaged over runs/seeds**; the only averaging in the section is the cosine/L2 table, which averages over the **ten selected classes**. A per-column "what is measured" table was added.

**Follow-up (user feedback):** (1) "the reverse arrows aren't the forward arrows" — correct and expected (reverse starts exactly at the prototype; reverse Euler ≠ inverse of the discrete forward steps; the standard field near t=1 is a class-conditional average); made it *visible* instead of implicit — each reverse panel now draws the forward class-centroid path (dashed, faint, same plane): standard's reverse path hugs it, rollout's departs immediately; explanation added to §5c. (2) Reverse-flow **Top-1 (%)** added on the complete test split (`stage2_reverse_top1_*.csv`, forward accuracy asserted against the published grid before writing): round-trip (fwd→rev→classify) — standard **31.17%** vs 29.13% baseline (information preserved), rollout **26.01%** (lost); reverse-endpoint classifier (backward images of all 100 prototypes as class representatives) — standard **25.68%**, rollout **11.07%** (collapse). Notebook §5c → 7 cells (40 total, 0 errors); report §4 updated; check + tests green.

**Experiment (representative setting, both modes, same joint-PCA plane fitted once — never refit):** `make_figures_stage2.py reverse` → `stage2_reverse_flow_fgvc_aircraft_dinov2_vits14_T12.png` (reverse trajectories from the 10 selected prototypes, arrowheads t=1→0, square endpoint markers) + intermediate-time comparison (reverse-prototype state vs forward same-class centroid at matching Euler times, cosine + L2 in 384-d): per-class CSV `stage2_reverse_intermediate_*.csv` + summary chart. **Finding:** the modes separate sharply and training-free — standard FM reverses gracefully (endpoint cosine 0.85 ± 0.06, paths stay in the data region) while rolled-out FM collapses in reverse (0.24 ± 0.20, paths shoot outside the data; L2 ~4×) — an independent confirmation of the per-path-supervision vs endpoint-only mechanism story. Notebook section §5c (after the animations, before the discussion) with the equation, both figures, live-computed endpoint table, results-based discussion and the six required limitations; s6/report deviations rows flipped from "not attempted" to "done ‡"; report §4 paragraph added. Notebook: 38 cells, 0 errors/warnings; `check` green over both grids; tests green.

---

## 2026-07-17 — Repo moved to D:, scaffold *(archived v1 — see note above)*

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

## 2026-07-17 — Teammate LLM-review response: probe × backbone grid + self-contained notebook

A co-member's LLM review of the notebook (`_docs/feedbackImprove.txt`, 8.5/10) was triaged: 5 relevant improvements, 4 points already solved in the repo but invisible in the notebook (the review saw *only* the notebook — packaging lesson), 1 hallucination ("Stage 4"). New CLAUDE.md rule: **the presentation notebook must be self-contained** — evidence living only in src/results/report doesn't exist for reviewers.

**The substantive fix — linear probe on all three backbones** (episodic + simple; grid now 11 heads). This overturned our headline claim: the probe beats the prototype **only on CLIP embeddings**. Paired per-episode CIs: DINOv2 → prototype wins 5/6 settings (e.g. CIFAR-10 1-shot −4.58±0.41; probe recovers only MNIST 5-shot +4.57±0.41); ResNet-50 split (MNIST pro-probe, CIFAR/Mini pro-prototype). Interpretation: fixed a-priori probe budget (lr 0.01/300 steps) is matched to CLIP's normalized 512-d features; at K=1 the probe overfits single examples. Report §4 rewritten with the full 3×6 paired table; abstract scoped accordingly. The reviewer's concern was validated *harder* by the data than it claimed.

**Also applied:** zero-shot CLIP reframed as a *semantic reference baseline* (class names + pretrained alignment ≠ labeled support images); Mini-ImageNet caveat extended to CLIP/DINOv2 ("frozen foundation-model embeddings" setting, not comparable to traditional FSL literature); t-SNE demoted to qualitative with new quantitative backing (`scripts/embedding_metrics.py`: cosine silhouette / 1-NN / within-between ratio → `embedding_quality.csv`, table in notebook §3 — backbone ranking matches classifier ranking); zero-shot "± 0.00" removed from the simple table (deterministic, no support sampling — note added); new notebook §6 "Protocol integrity & provenance" (ADR 0003 incl. the c_k support-only formula, hyperparameter provenance incl. init/wd/seed/reduction, live-computed paired-diff table, query-label hygiene). Notebook now 26 cells, 8 sections, executes end-to-end. repro_check green (124 rows).

## 2026-07-17 — Baseline selection reframe + architecture diagrams

**Reframe (user direction):** Stage 1's purpose is *selection, not ranking* — find the strongest encoder **per dataset** for each classifier function; Stages 2–3 improve against that. `scripts/select_baselines.py` derives the per-(dataset × head × K) winner from episodic.csv (contaminated ResNet-50 × Mini-ImageNet excluded, within-CI alternatives recorded) → committed `results/artifacts/best_baselines.json`, which Stage 2 loads. Pattern: linear→CLIP, prototype→DINOv2 (CLIP for MNIST 5-shot), zero-shot→ensemble (single prompt on MNIST). CLAUDE.md rule added: FM variants must beat the *selected* config of their head, per dataset, paired on the same episodes.

**Architecture diagrams** (`scripts/make_architecture_figs.py`): `arch_baselines.png` — the three heads (prototype / linear probe / zero-shot CLIP) as box-flow panels with formulas and training notes; `arch_stage2_advised.png` — advised Stage-2 (FM as decision layer: x₀ = f(x) transported to per-episode support prototypes or text embeddings, nearest-target classification) and Stage-3 (frozen encoder → FM → nn.Linear, trained jointly with CE) designs, each on its dataset's selected embedding. Placed as report Figures 2 and 11 (others renumbered) and in notebook §4 and §7. Notebook now 30 cells, executes end-to-end.

## 2026-07-17 — Second teammate LLM review: selection moved to validation data

Second LLM review (`_docs/feedbackImprove1.txt`, 9.5/10) caught one genuine methodological flaw plus a design improvement; user independently caught a third:

1. **Selection leakage (valid, serious):** `best_baselines.json` was selected on *test* episodic accuracy. Fixed with the standard select–freeze–evaluate protocol: selection episodes now come from data disjoint from all test evaluation — Mini-ImageNet's 16 R&L **validation classes** (features + val text embeddings extracted for all backbones; their canonical purpose) and **train-split** episodes for MNIST/CIFAR-10 (seed 123, 600 eps/K, saved as `*_valsel_*` files). Test numbers are now a one-time read-out for the selected config.
2. **One embedding per (dataset, head) across K (valid):** selection = mean val accuracy over K∈{1,5} + paired per-episode check vs the runner-up on the same val episodes; statistical tie → smaller embedding.
3. **Zero-shot K-dimension mistake (user catch):** K never applied to zero-shot (no support); its selection axis is the *prompt variant* per dataset.
4. Rejected: the review's "nh.Linear" typo claim — the figure correctly renders `nn.Linear` (LLM misread).

**Selection results (validation-based):** prototype → CLIP-eucl on MNIST (statistical tie +0.04±0.07, CLIP kept), DINOv2-cos on CIFAR-10 (+1.16±0.18) and Mini-ImageNet (+0.43±0.11); linear → CLIP everywhere (+2.51/+1.67/+0.67); zero-shot → single prompt on MNIST (+2.96), ensemble on CIFAR-10/Mini-ImageNet. Validation choice agrees with what test would have chosen — the selection generalizes. ResNet-50 stays excluded on Mini-ImageNet (val classes are ImageNet-1k too). All val accuracies in `results/metrics/selection_validation.csv`; docs' "val classes unused" claims corrected; CLAUDE.md rule updated ("never select or tune on test"). Notebook rebuilt (30 cells).

## 2026-07-20 — Multi-prototype (k-means) ablation

User-requested extension (matches the original `_docs/K-Mean Prototype.png` sketch): n_centers ∈ {1,2,3} spherical k-means centers per class from **support only** (deterministic init from first n support points, 10 iterations, empty clusters keep previous center); query scored by max cosine over class centers. Sanity: n=1 verified per-episode **identical** to proto_cos, K=1 clamp degenerates identically. Run only where K ≥ n_centers (episodic 5s, simple 5s/10s), all 3 backbones (grid now 166 repro-checked rows).

**Result — one center is enough at 5-shot, multi-modality pays at K=10:** paired vs the single prototype on each dataset's selected backbone (episodic 5s): MNIST tie (−0.32±0.39, −0.12±0.43), CIFAR-10 significantly worse (−0.94±0.20, −1.64±0.23), Mini-ImageNet marginally worse (−0.10±0.07, −0.16±0.08) — each center gets ~5/n samples, estimation noise dominates. MNIST simple 10-shot: **n=3 wins +1.18±1.08 paired** — digit styles are multi-modal, but need ~3+ shots per center. Stage-2 implication: class-mean targets are right at 5-shot; multi-center targets only at higher K. New `kmeans_ncenters.png` (report Figure 11, others renumbered); ablation heads excluded from main curves for legibility; repro_check green.

## 2026-07-20 — Per-classifier accuracy charts (readability)

User-requested split of the accuracy visualizations by classifier head, replacing the 11-series combined curves as the primary display: `acc_prototype_backbones.png` and `acc_linear_backbones.png` (accuracy vs support size, one line per embedding, panels per dataset with protocol labels — MNIST/CIFAR-10 simple ± std, Mini-ImageNet episodic ± CI with ResNet-50 † noted) and `zeroshot_variants.png` (single prompt vs ensemble per dataset, **no shots axis** — zero-shot uses class names, not support). One fixed color per embedding across every chart (`BACKBONE_COLORS` in `src/visualize.py`); k-means ablation kept in its own chart. Report Figures 5–7 swapped/inserted (others renumbered to 14); combined `acc_vs_k_*` files remain in `results/figures/` as counterparts. Notebook s5 rebuilt around the split charts (34 cells, executed).

## 2026-07-20 — Strict per-protocol figure organization (user catch)

User caught that the per-classifier charts mixed protocols in one figure (MNIST/CIFAR simple + Mini-ImageNet episodic) while sitting in the episodic section — misleading. Restructured to strict one-protocol-per-figure:

- **Episodic set** (`episodic.csv`, 5-way, K∈{1,5}, all 3 datasets): `ep_acc_prototype`, `ep_acc_linear`, `ep_zeroshot_variants` (episode queries), `ep_kmeans_ncenters` (renamed; moved into the episodic section since it uses 5w5s results).
- **Simple set** (`simple.csv`, all 10 classes, K∈{1,5,10}, MNIST/CIFAR only): `simple_acc_prototype`, `simple_acc_linear`, `simple_zeroshot_variants`, **new** `simple_kmeans_ncenters` (n∈{1,2,3} × K∈{5,10} — visually shows the MNIST K=10 n=3 win).
- Mixed-protocol figures deleted from the repo. Notebook §5/§5b rebuilt with explicit protocol banners and per-section figure sets (32 cells, executed); report §3.1/§3.2 headers state the data source, figures renumbered 1–18, k-means embed moved from §4 into §3.1 (text cross-references Figures 8/12).

**Known Windows/ROCm quirks carried over from cv-ex2** (guards already in place):
- `KMP_DUPLICATE_LIB_OK=TRUE` before torch import — otherwise the `clip` package triggers an OpenMP duplicate-runtime crash.
- CLIP model forced to `.float()` — fp16 weights misbehave on the ROCm stack.
- SSL unverified-context fallback available for dataset downloads (`utils.allow_insecure_downloads`).
- Free GPU memory (`torch.cuda.empty_cache()`) between backbones during extraction.
