# v2 talk — section 7: main classification results.

HEAD = r"""
## <span style="color:#1f3a5f;">6 · Main classification results</span>

Top-1 accuracy on the complete official test split; K = 10, T = 4; mean ± sample standard deviation over three subset seeds.
"""

TABLE_CODE = r"""
# Main comparison, generated from the summary of record (results/metrics/summary_stage3.csv)
# and cross-checked against the generated table of record (stage3_main_table.md, Appendix A.4).
summary = pd.read_csv(MET / "summary_stage3.csv")
record = (MET / "stage3_main_table.md").read_text(encoding="utf-8")
METHOD = {"pinned_probe": "Pinned Stage-1 probe", "fm_s1": "Strategy 1", "fm_s2": "Strategy 2"}
rows = []
for ds, enc in cfg["stage3"]["settings"]:
    for head in ("pinned_probe", "fm_s1", "fm_s2"):
        r = summary[(summary["dataset"] == ds) & (summary["encoder"] == enc)
                    & (summary["head"] == head)].iloc[0]
        acc = f"{100 * r['mean_acc']:.2f} ± {100 * r['std_acc']:.2f}"
        if head == "pinned_probe":
            delta = "baseline"
        else:
            delta = f"{100 * r['delta_mean']:+.2f} ± {100 * r['delta_std']:.2f}".replace("-", "−")
        # Every displayed value must appear verbatim in the table of record.
        assert acc in record and (delta == "baseline" or delta.replace("−", "-") in record), (acc, delta)
        rows.append({"Dataset": f"{dataset_label(ds)} / {encoder_label(enc, short=True)}",
                     "Method": METHOD[head], "Top-1 accuracy (%)": acc,
                     "Paired change (points)": delta})
main_table = pd.DataFrame(rows)


def _emphasize(row):
    best = row["Dataset"].startswith("FGVC") and row["Method"] == "Strategy 2"
    css = "font-size:1.2em; padding:8px 18px; white-space:nowrap;"
    return [css + (" font-weight:bold; background-color:#f2f8f7; color:#1a1a1a;" if best else "")] * len(row)


display(main_table.style.apply(_emphasize, axis=1).hide(axis="index")
        .set_table_styles([{"selector": "th", "props":
                            "font-size:1.2em; text-align:left; padding:8px 18px; color:#1f3a5f;"}]))
"""

AFTER = r"""
<div style="border-left:6px solid #0b7a75; background:#f2f8f7; color:#1a1a1a; padding:10px 16px; margin:0.8em 0; font-size:1.15em;"><b>Takeaway.</b> Strategy 2 improves FGVC-Aircraft on all three observed seeds, while DTD shows no clear benefit at this operating point.</div>

- Values are mean ± sample standard deviation over three subset seeds; each change is **paired per seed** against the exact pinned probe of that seed.
- Seed 0 was used for hyperparameter selection and is therefore partly a development run.
- "Consistent" describes agreement in sign across the three observed seeds; no statistical significance is claimed.
- Per-seed changes, the $\lambda = 0$ pair and the full validation sweep: Appendix A.5–A.7.
"""

CELLS = [
    ("markdown", HEAD),
    ("code", TABLE_CODE),
    ("markdown", AFTER),
]
