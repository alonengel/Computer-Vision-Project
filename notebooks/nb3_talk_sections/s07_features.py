# Talk section 7 — feature-space joint PCA (figures of record, shown larger; FGVC first).


def NOTES(body):
    return ('\n<details><summary><b>Speaker notes · הערות למרצה (עברית)</b></summary>\n'
            '<div dir="rtl" style="text-align:right; line-height:1.65; margin:0.6em 1em;">\n'
            + body.strip() + '\n</div></details>\n')


HEAD_FGVC = r"""
## 7 · Feature space — what the FM does to the class structure (joint PCA, seed 0)

**FGVC-Aircraft / DINOv2** — the setting with the +2.50-point gain:
"""

FGVC_CODE = r"""
display(Image(str(FIG / "stage3_features_fgvc_aircraft_dinov2_vits14.png"), width=1500))
"""

HEAD_DTD = r"""
**DTD / ResNet-18** — the setting with no clear benefit:
"""

DTD_CODE = r"""
display(Image(str(FIG / "stage3_features_dtd_resnet18.png"), width=1500))
"""

AFTER = r"""
- **Same ten classes, same test examples, same class colours** as Stages 1–2; one PCA fitted **jointly** on $[z,\ \hat{z}_{S1},\ \hat{z}_{S2}]$ in raw space, so the three panels share axes and are directly comparable.
- **The first two PCs change only subtly** — even on FGVC-Aircraft, despite its +2.50-point gain. The transport acts in the full 384-d / 512-d space, along classifier-relevant directions this plane barely captures (it explains 15.6% of the variance on DTD, 33.0% on FGVC).
- Consistent with the measured displacements (≈ 5–13% of the mean feature norm): small, targeted adjustments rather than a visible reorganization. **2-D projections are qualitative only.**
"""

HEB = """
<p><b>הנקודה המרכזית:</b> התזוזה שה-FM עושה כמעט לא נראית בשני הרכיבים הראשיים — היא קטנה ומכוונת, בכיוונים שרלוונטיים למסווג במרחב הגבוה-ממדי.</p>
<p><b>מונחים בפשטות:</b> <b>PCA</b> — הטלה ליניארית לשני הכיוונים עם השונות הגדולה ביותר. <b>Joint PCA</b> — ההטלה מחושבת פעם אחת על [z, ẑ_S1, ẑ_S2] יחד, ולכן שלושת הפאנלים חולקים צירים ואפשר להשוות ביניהם. <b>Explained variance</b> — כמה מהשונות הכוללת המישור הדו-ממדי תופס: 15.6% ב-DTD (8.2% + 7.4%), 33.0% ב-FGVC (20.6% + 12.4%).</p>
<p><b>מה לומר על הגרפים:</b> אותן 10 מחלקות, אותן דוגמאות ואותם צבעים כמו בשלבים 1–2. להצביע על כך שהמבנה כמעט זהה בשלושת הפאנלים למרות +2.5 נקודות ב-FGVC; להזכיר שהתזוזות הנמדדות הן כ-5–13% מנורמת הפיצ'ר. להתחיל מ-FGVC (הסיפור העיקרי) ולעבור ל-DTD בקצרה.</p>
<p><b>שאלה צפויה:</b> "אם לא רואים שינוי, איך יש שיפור?" — <b>תשובה:</b> המישור מסביר רק 33% מהשונות ב-FGVC; המסווג פועל על 384 ממדים, ותזוזה קטנה בכיוון שמשנה את הלוגיטים לא חייבת להיות נראית ב-PC1/PC2. זו אבחנה איכותית בלבד — לא ראיה כמותית.</p>
<p><b>זמן משוער:</b> ~3 דקות.</p>
"""

CELLS = [
    ("markdown", HEAD_FGVC),
    ("code", FGVC_CODE),
    ("markdown", HEAD_DTD),
    ("code", DTD_CODE),
    ("markdown", AFTER + NOTES(HEB)),
]
