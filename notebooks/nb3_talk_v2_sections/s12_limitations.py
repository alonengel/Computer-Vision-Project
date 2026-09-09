# v2 talk — section 13: limitations and final takeaway.

MAIN = r"""
## <span style="color:#1f3a5f;">12 · Limitations and final takeaway</span>

- Only three subset seeds.
- Seed 0 was used for hyperparameter selection.
- Only one K, one T and one encoder per dataset were tested.
- PCA is a qualitative two-dimensional projection.

<div style="border-left:6px solid #0b7a75; background:#f2f8f7; color:#1a1a1a; padding:12px 18px; margin:1.0em 0; font-size:1.2em;"><b>Takeaway.</b> Flow Matching before a frozen classifier is not universally beneficial, but conservative classifier-guided targets provide a consistent improvement on the fine-grained FGVC-Aircraft setting.</div>
"""

CELLS = [("markdown", MAIN)]
