# Talk section 9 — limitations (four bullets) + final takeaway.


def NOTES(body):
    return ('\n<details><summary><b>Speaker notes · הערות למרצה (עברית)</b></summary>\n'
            '<div dir="rtl" style="text-align:right; line-height:1.65; margin:0.6em 1em;">\n'
            + body.strip() + '\n</div></details>\n')


MAIN = r"""
## 9 · Limitations and takeaway

- **n = 3 subset seeds** — spreads are sample standard deviations; no significance claims (the spread measures subset sampling only; probe-init and FM-init are fixed).
- **Seed 0 is partly a development run** — hyperparameters were selected on its validation split; the 3-seed mean is a summary, not an independent confirmatory estimate.
- **One K (10), one T (4), one encoder per dataset** — by the spec's own scoping; conclusions are about this operating point.
- **2-D PCA is qualitative** — the joint plane explains 15.6% (DTD) / 33.0% (FGVC-Aircraft) of the variance. Full limitations list: Appendix A.8.

> **Takeaway:** Flow Matching before a frozen classifier is not universally beneficial, but conservative classifier-guided targets produce a consistent improvement on the fine-grained FGVC-Aircraft setting.
"""

HEB = """
<p><b>הנקודה המרכזית:</b> המגבלות מגדירות מה מותר להסיק — ואז המסר הסופי, מילה במילה.</p>
<p><b>מונחים בפשטות:</b> <b>Sample std</b> — סטיית תקן מדגמית (חלוקה ב-n−1). <b>Operating point</b> — הצירוף הספציפי של K, T, encoder ומסווג. <b>Development run</b> — ריצה ששימשה גם לבחירת היפר-פרמטרים ולכן אינה אימות בלתי-תלוי.</p>
<p><b>מה לומר:</b> לעבור על ארבעת הבולטים בקצרה ולסיים במשפט המסר. לא להוסיף טענות מעבר לו: "לא תמיד מועיל, אבל מטרות שמרניות מונחות-מסווג נותנות שיפור עקבי ב-FGVC-Aircraft".</p>
<p><b>שאלה צפויה:</b> "מה הייתם עושים עם עוד זמן?" — <b>תשובה:</b> יותר seeds (להעריך את השונות), אבלציה של רכיבי S2 ושל רדיוס אזור האמון α = 0.1 (שהיה "רווי" — כמעט 100% מהמטרות על שפת הכדור, נספח A.5), ו-K / T נוספים.</p>
<p><b>זמן משוער:</b> ~2 דקות.</p>
"""

CELLS = [("markdown", MAIN + NOTES(HEB))]
