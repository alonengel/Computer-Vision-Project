# Talk section 8 — three conclusions + interpretation.


def NOTES(body):
    return ('\n<details><summary><b>Speaker notes · הערות למרצה (עברית)</b></summary>\n'
            '<div dir="rtl" style="text-align:right; line-height:1.65; margin:0.6em 1em;">\n'
            + body.strip() + '\n</div></details>\n')


MAIN = r"""
## 8 · Conclusions

1. **Strategy 2 is the best method on FGVC-Aircraft / DINOv2:** +2.50 ± 0.50 points over the pinned probe, positive on 3/3 seeds (+2.76 / +1.92 / +2.82); Strategy 1 gains +0.88 ± 0.23 (3/3).
2. **DTD / ResNet-18 shows no clear improvement:** Strategy 2 +0.16 ± 0.45 (2/3 seeds), Strategy 1 −0.69 ± 0.70 (1/3) — everything within the seed spread at this operating point.
3. **The indirect strategy beats the direct one on both datasets** (FGVC +2.50 vs +0.88; DTD +0.16 vs −0.69; the same ordering appeared on the validation sweeps before test was opened) — but the component responsible is **not isolated**.

Strategy 2 differs from Strategy 1 through several components at once: the source-centred trust region, lowest-CE (monotone) target selection, classifier-guided target construction, and standard path-regression training.

Both strategies memorize the training set, so the gap lies in what each objective does *off* the training points — and everything that differs there pushes Strategy 2 toward conservatism. Which component matters is an open question this experiment does not answer.
"""

HEB = """
<p><b>הנקודה המרכזית:</b> שלוש מסקנות — S2 הכי טוב ב-FGVC (+2.50 ± 0.50, 3/3 seeds); אין שיפור ברור ב-DTD; והשיטה העקיפה מנצחת את הישירה בשני הדאטהסטים — אבל בלי לבודד איזה רכיב אחראי.</p>
<p><b>מונחים בפשטות:</b> <b>Direct vs indirect</b> — S1 מקבל גרדיאנט CE ישירות; S2 רק דרך מטרות. <b>Path regression</b> — אימון על כל הקו בין z ל-ẑ′ ולא רק על נקודת הקצה. <b>Conservatism</b> — מגבלות קשיחות על גודל התזוזה ועל טיב המטרה (אזור אמון + קבלה מונוטונית).</p>
<p><b>מה לומר:</b> לקרוא את שלוש המסקנות עם המספרים. בפסקאות — להדגיש ש-S2 שונה מ-S1 בכמה רכיבים <u>בבת אחת</u>, ולכן אי אפשר לומר איזה מהם אחראי; שתי האסטרטגיות משננות את סט האימון, ולכן ההבדל הוא במה שקורה מחוץ לנקודות האימון. הדמיון לתוצאת שלב 2 (path supervision מכליל טוב יותר מ-endpoint) הוא דפוס חוזר, לא מנגנון שבודד.</p>
<p><b>שאלה צפויה:</b> "איזה רכיב של S2 אחראי לשיפור?" — <b>תשובה:</b> לא ידוע מהניסוי הזה; זה דורש אבלציה (למשל S1 עם אזור אמון קשיח, או S2 בלי בחירה מונוטונית). זו הצעה מפורשת לעבודת המשך.</p>
<p><b>זמן משוער:</b> ~3 דקות.</p>
"""

CELLS = [("markdown", MAIN + NOTES(HEB))]
