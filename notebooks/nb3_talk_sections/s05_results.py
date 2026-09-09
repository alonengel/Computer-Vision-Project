# Talk section 5 — main results table (values read verbatim from the table of record).


def NOTES(body):
    return ('\n<details><summary><b>Speaker notes · הערות למרצה (עברית)</b></summary>\n'
            '<div dir="rtl" style="text-align:right; line-height:1.65; margin:0.6em 1em;">\n'
            + body.strip() + '\n</div></details>\n')


HEAD = r"""
## 5 · Main results — test top-1 (%), K = 10, T = 4, mean ± std over the 3 subset seeds
"""

TABLE_CODE = r"""
# Main comparison, read VERBATIM from the table of record
# (results/metrics/stage3_main_table.md) — presentation-only styling. No value
# is retyped or recomputed; the raw file with its full caption is in Appendix A.3.
_md = (MET / "stage3_main_table.md").read_text(encoding="utf-8")
_rows = [[c.strip() for c in ln.strip().strip("|").split("|")]
         for ln in _md.splitlines() if ln.startswith("|")]
_header = _rows[0]
_body = [r for r in _rows[1:] if not set("".join(r)) <= set("-: ")]
main_tbl = pd.DataFrame(_body, columns=_header)


def _row_style(row):
    fgvc = row.iloc[0].startswith("FGVC")
    win = fgvc and row.iloc[1].startswith("Strategy 2")
    css = "font-size:1.2em; padding:8px 18px; white-space:nowrap;"
    if fgvc:
        css += " background-color:#fff3e0; color:#1a1a1a;"
    if win:
        css += " font-weight:bold;"
    return [css] * len(row)


display(main_tbl.style.format(escape="html").apply(_row_style, axis=1)
        .hide(axis="index")
        .set_table_styles([{"selector": "th",
                            "props": "font-size:1.2em; text-align:left; padding:8px 18px;"}]))
"""

AFTER = r"""
**FGVC-Aircraft / DINOv2:** pinned probe **51.30 ± 0.95%** → Strategy 1 **52.18 ± 1.11%** (Δ **+0.88 ± 0.23**) → Strategy 2 **53.80 ± 0.95%** (Δ **+2.50 ± 0.50**, 3/3 seeds).

**DTD / ResNet-18:** every Δ lies within the seed spread (Strategy 2 +0.16 ± 0.45; Strategy 1 −0.69 ± 0.70).

> **Takeaway:** Strategy 2 provides a modest but consistent gain on FGVC-Aircraft, while DTD shows no clear benefit at this operating point.

- Δ is paired per seed against the exact pinned probe of that seed — the pipeline at initialization equals it exactly.
- Hyperparameters were selected on seed-0 validation only, so seed 0 is partly a development run; the 3-seed mean is a summary, not an independent confirmatory estimate.
- n = 3: "consistent" is descriptive (all three seeds agree in sign), not a statistical-significance claim. Full seed-0 sweep, λ-ablation pair and per-seed table: Appendix A.3–A.4.
"""

HEB = """
<p><b>הנקודה המרכזית:</b> ב-FGVC-Aircraft אסטרטגיה 2 משפרת ב-+2.50 ± 0.50 נקודות עם 3/3 seeds חיוביים; ב-DTD אין שיפור ברור — כל ההפרשים בתוך פיזור ה-seeds.</p>
<p><b>מונחים בפשטות:</b> <b>Top-1</b> — אחוז הדוגמאות שהתחזית הראשונה שלהן נכונה. <b>mean ± std</b> — ממוצע וסטיית תקן מדגמית על 3 seeds. <b>pts</b> — נקודות אחוז (הפרש בין שני אחוזים). <b>3/3 seeds &gt; 0</b> — בכל שלוש הדגימות ה-Δ חיובי.</p>
<p><b>מה לומר על הטבלה:</b> לקרוא את שלוש שורות FGVC (המודגשות): 51.30 → 52.18 → 53.80. להשתמש במילה "עקבי" (אותו סימן בכל seed) ולא "מובהק" — n = 3. ב-DTD: S2 +0.16 ± 0.45 ו-S1 −0.69 ± 0.70 — שניהם בתוך הרעש.</p>
<p><b>שאלה צפויה:</b> "האם +2.5 מובהק סטטיסטית?" — <b>תשובה:</b> עם n = 3 לא טוענים מובהקות. הטענה תיאורית: שלושת ה-seeds מסכימים בסימן (+2.76 / +1.92 / +2.82), והממוצע גדול פי ~5 מסטיית התקן של ה-Δ. גילוי נאות: seed 0 שימש גם לבחירת ההיפר-פרמטרים.</p>
<p><b>זמן משוער:</b> ~4 דקות.</p>
"""

CELLS = [
    ("markdown", HEAD),
    ("code", TABLE_CODE),
    ("markdown", AFTER + NOTES(HEB)),
]
