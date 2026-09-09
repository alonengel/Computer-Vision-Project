# v2 talk — section 12: conclusions.

MAIN = r"""
## <span style="color:#1f3a5f;">11 · Conclusions</span>

1. **Strategy 2 is the best tested configuration on FGVC-Aircraft:** a paired gain of +2.50 ± 0.50 points, with positive changes on all three seeds.
2. **DTD shows no clear improvement** at the tested operating point.
3. **The indirect classifier-guided strategy outperforms direct rolled-out CE training on both datasets.**

**Caveat.** Strategy 2 changes several components simultaneously — target construction, trust-region projection, lowest-CE acceptance and standard path regression. The experiment therefore supports the complete strategy but does not identify which individual component causes the advantage.
"""

CELLS = [("markdown", MAIN)]
