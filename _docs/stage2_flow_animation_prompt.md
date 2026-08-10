# Stage 2 Flow Animation - Implementation Prompt

Review `stage2_presentation.ipynb` and add a new section immediately after the existing static flow-trajectory section:

## Animated Flow Trajectories

Create a clear inline animation that visualizes the **actual learned Flow Matching transport** using the saved Stage 2 models, feature caches, prototypes, and test examples. Do not generate a synthetic or purely illustrative trajectory, and do not modify any existing experimental result.

## Representative Setting

Use the following representative experiment for the main animation:

- Dataset: FGVC-Aircraft
- Encoder: DINOv2 ViT-S/14
- Target branch: image-derived class prototypes
- Training-set size: `K = 10`
- Subset seed: `0`
- Number of Euler steps: `T = 12`
- Methods: Standard FM and Rolled-out FM

This setting should be used because it shows the clearest Stage 2 improvement and provides an informative comparison between the two training objectives.

## Data and Model Requirements

1. Load the real saved Standard FM and Rolled-out FM velocity networks for this exact setting.
2. Put both networks in evaluation mode and perform trajectory generation inside `torch.no_grad()`.
3. Reuse the same four representative test examples used in the existing static trajectory figure.
4. Reuse the same true-class labels, class colors, class prototypes, normalized starting features, and selected test-example indices.
5. Reuse the same PCA coordinate system as the existing Stage 2 feature-comparison and trajectory figures whenever possible.
6. If the PCA object was not saved, reconstruct it using exactly the same selected classes, test examples, original feature set, transported feature sets, and prototypes as the static figure. Fit PCA only once and use the same transform for both methods and every animation frame.
7. Do not fit PCA separately for different frames or methods.
8. Do not use t-SNE for the animation.

## Exact Flow Computation

For each selected test feature, generate every real Euler state using the implemented inference rule:

$$
\hat{z}_{k+1}
=
\hat{z}_k
+
\frac{1}{T}
v_\theta\left(\hat{z}_k,\frac{k}{T}\right),
\qquad k=0,\ldots,T-1.
$$

The animation must satisfy the following conditions:

- Start from the same normalized feature used by the Stage 2 inference pipeline.
- Generate exactly `T + 1` states: $\hat{z}_0,\ldots,\hat{z}_T$.
- Use the correct model-specific time value $t=k/T$ at every step.
- Do not renormalize features between Euler steps, matching the existing Stage 2 inference implementation.
- Use the Standard FM model for the Standard panel.
- Use the `T = 12` Rolled-out FM model for the Rolled-out panel.
- Project the already-computed high-dimensional states into PCA only after the trajectories have been generated.

## Animation Layout

Create one figure with two synchronized panels:

1. `Standard FM - T = 12`
2. `Rolled-out FM - T = 12`

Both panels must use identical:

- PCA coordinates;
- axis limits;
- aspect ratio;
- class colors;
- prototype markers;
- example ordering;
- background points;
- frame timing.

In every panel:

- Show the original selected test-feature background with low opacity.
- Show all relevant class prototypes as large star markers.
- Show each example's original feature $\hat{z}_0$ as a hollow circle.
- Show its current transported state as a large filled marker in its true-class color.
- Preserve previously visited states as a visible trajectory trail.
- Connect consecutive Euler states with thin line segments.
- Mark the final transported state $\hat{z}_T$ with an `X` when the final frame is reached.
- Optionally connect each corresponding prototype to the moving point using a faint dashed guide line, but do not obscure the real trajectory.
- Add an informative legend explaining the original point, current state, trajectory, final point, and prototype.
- Display the current step and time clearly, for example: `Euler step 5 / 12, t = 0.417`.

Use fixed axis limits throughout the animation to avoid visual movement caused by automatic rescaling.

## Animation Implementation

Use `matplotlib.animation.FuncAnimation`.

Display the result directly in the notebook using:

```python
from IPython.display import HTML
HTML(anim.to_jshtml())
```

Also attempt to save the animation as:

```text
results/figures/stage2_flow_animation_fgvc_dinov2_T12.gif
```

using `PillowWriter`. If GIF export is unavailable, the inline JavaScript animation must still work and the notebook should continue without an error.

Use a readable frame interval of approximately 500-700 ms per Euler step. Hold the initial and final frames slightly longer so the viewer can understand the start and outcome.

Do not require `ffmpeg` for the inline notebook animation.

## Programmatic Verification

Before displaying the animation, add assertions that verify:

1. Each trajectory contains exactly `T + 1` states.
2. The first state exactly matches the normalized original test feature used by the Stage 2 inference pipeline.
3. The final state matches the final transported feature used by the existing static trajectory figure within an appropriate floating-point tolerance.
4. Standard FM and Rolled-out FM start from identical features.
5. The correct corresponding prototype is used for every example.
6. Every projected state is produced by the same fitted PCA transform.
7. Neither model receives gradients during animation generation.
8. No existing result table, metric file, model checkpoint, prediction artifact, or figure is overwritten, except for creating the new animation artifact.

Print a short verification summary before the animation, such as:

```text
[OK] 4 examples loaded
[OK] 13 Euler states per trajectory
[OK] shared PCA transform
[OK] final states match stored trajectory results
```

## Markdown Explanation Below the Animation

Add a short explanation containing the following points:

- The markers follow real states generated by the learned velocity networks, not a manually drawn interpolation.
- Standard FM receives velocity supervision at random points along the ideal interpolation path.
- Rolled-out FM is trained through its own self-generated sequence and is supervised only through the final transported state.
- The actual transport occurs in the original 384-dimensional DINOv2 feature space.
- PCA displays only a qualitative two-dimensional projection of that motion.
- Projected path length, curvature, and distance should not be interpreted as exact high-dimensional measurements.
- The animation is intended to explain how repeated Euler updates accumulate into the final transported representation.

## Optional Extension

After the main animation works, optionally add a compact selector or a second animation for:

- DTD with ResNet-18 and image prototypes; or
- FGVC-Aircraft with CLIP RN50 transported toward text prototypes.

Clearly label CLIP transport as an extension and as supervised transport using labeled examples, not zero-shot classification.

## Final Validation

After adding the animation:

1. Restart and run the entire notebook from top to bottom.
2. Confirm that every cell executes without errors.
3. Confirm that all existing numerical tables and figures remain unchanged.
4. Confirm that the animation plays inline after reopening the notebook.
5. Keep the new section concise enough for presentation use.

