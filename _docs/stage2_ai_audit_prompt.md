# Stage 2 Notebook Audit Prompt

Please perform a strict, evidence-based audit of the attached **Stage 2 requirements PDF** and **Stage 2 presentation notebook**.

Do not modify the notebook yet. First, inspect both files carefully and identify anything that is incorrect, incomplete, misleading, inconsistent, or insufficiently justified.

## What to check

1. Compare every requirement in the PDF against the implementation, experiments, tables, plots, equations, captions, and discussion in the notebook.
2. Verify the Standard Flow Matching and Rolled-out Flow Matching implementations mathematically and in code, including interpolation, velocity target, loss, Euler rollout, values of `T`, time conditioning, prototypes, and inference classification.
3. Check that Stage 1 and Stage 2 use the same datasets, encoders, prototypes, subsets, seeds, test splits, and baseline protocol wherever the specification requires this.
4. Audit the evaluation protocol: 5-shot, 10-shot, full-data experiments, number and meaning of runs, mean/std calculation, top-1 accuracy, delta computation, and fair baseline comparisons.
5. Check all accuracy tables, training curves, feature projections, trajectories, and animations for correctness, fair comparison, readable labeling, and consistency with the saved metrics.
6. Inspect the conclusions and confirm that every claim is supported by the reported results.
7. Look for data leakage, accidental use of validation/test information, inconsistent normalization, nondeterminism, cherry-picking, unstable training, or checkpoint-selection bias.

## Examine these two choices especially carefully

### A. L2-normalized features

The Flow Matching model operates on L2-normalized image features rather than the raw frozen features. Determine:

- whether this is allowed by the literal specification;
- how it changes the learned problem and geometry;
- whether the baseline and prototype comparison remains fair;
- whether the notebook explains it clearly enough;
- whether a raw-vs-normalized ablation or instructor approval is needed.

### B. Minimum-training-loss checkpoint

Because some runs diverged, the reported checkpoint is selected using the minimum training loss instead of always using the final epoch. Determine:

- whether this is scientifically defensible and free of validation/test leakage;
- whether it can still introduce optimistic selection or noise-based checkpoint bias;
- whether the rule was applied consistently and disclosed clearly;
- whether a safer solution should be used, such as a lower learning rate, gradient clipping, scheduler, fixed epoch, EMA, or validation-based early stopping.

## Required output

Return the audit in this order:

1. **Overall verdict:** Ready / Mostly ready with minor fixes / Requires major fixes.
2. **Requirement checklist:** each PDF requirement marked Pass, Partial, or Fail, with notebook evidence.
3. **Critical issues:** issues that may invalidate results or violate the specification.
4. **Important improvements:** issues that should be fixed before submission or presentation.
5. **Minor presentation improvements:** wording, labels, captions, plot placement, and animation clarity.
6. **Normalization verdict:** keep it, add an ablation, or change to raw features, with reasoning.
7. **Checkpoint verdict:** keep the current rule or replace it, with a concrete recommended protocol.
8. **Exact action list:** prioritized changes, naming the relevant notebook section or cell.
9. **Questions for the instructor:** only decisions that genuinely require clarification.

Be skeptical but fair. Do not assume that a choice is wrong merely because it differs from a common convention. Separate definite specification violations from acceptable design choices and from optional improvements. Quote or point to exact evidence from the files, and do not invent missing results.
