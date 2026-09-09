# Talk section 4 — setup and guards (short confirmation; full evidence in the Appendix).


def NOTES(body):
    return ('\n<details><summary><b>Speaker notes · הערות למרצה (עברית)</b></summary>\n'
            '<div dir="rtl" style="text-align:right; line-height:1.65; margin:0.6em 1em;">\n'
            + body.strip() + '\n</div></details>\n')


MAIN = r"""
## 4 · Setup and guards

- **Data / encoders:** DTD → ResNet-18 (512-d) · FGVC-Aircraft → DINOv2 ViT-S/14 (384-d) — the Stage-1 validation-selected encoders, frozen, features cached once.
- **Protocol:** K = 10 images per class, T = 4, three balanced subset seeds {0, 1, 2}; every Δ is **paired per seed** against that seed's pinned probe.
- **Training:** AdamW (lr $10^{-3}$, weight decay $10^{-4}$), batch 64, **exactly 200 epochs**, best-validation-accuracy checkpoint; hyperparameters chosen on **seed-0 validation only**, then applied unchanged to seeds 1–2.
- **Test split sealed** during all training and selection; evaluated in one final pass per pre-registered phase, over already-locked checkpoints.

**Guards — all passed (re-executed in Appendix A.2):** exact identity at initialization holds on both settings (features bit-exact, logits and predictions identical to the direct probe); all six pinned probes reproduce the recorded Stage-1 validation and test accuracies within $10^{-12}$; no run ever hit NaN/Inf (every model at fallback level 0).
"""

HEB = """
<p><b>הנקודה המרכזית:</b> אותו פרוטוקול בדיוק לשתי האסטרטגיות, בחירת היפר-פרמטרים על ולידציה בלבד, test חתום, ושלוש בדיקות-שפיות שעברו.</p>
<p><b>מונחים בפשטות:</b> <b>Subset seed</b> — איזה 10 תמונות לכל מחלקה נדגמו (שלוש דגימות שונות → שלוש ריצות). <b>Paired Δ</b> — ההפרש נמדד לכל seed מול ה-probe של <u>אותו</u> seed. <b>Pinned probe</b> — המסווג של שלב 1 שאומן מחדש באותו מתכון ונשמר; כל השיטות מקבלות אותו מסווג קפוא. <b>Checkpoint</b> — נשמר המודל מה-epoch עם דיוק הוולידציה הגבוה ביותר. <b>Fallback ladder</b> — מדיניות שנקבעה מראש למקרה של NaN/Inf (grad-clip → lr נמוך → סטנדרטיזציה); לא נדרשה אף פעם.</p>
<p><b>מה לומר:</b> להדגיש "exactly 200 epochs" ו-"seed-0 validation only". שלוש הבדיקות: זהות מדויקת באתחול, שחזור ה-probes של שלב 1 (סטייה קטנה מ-10⁻¹² בוולידציה וב-test), ואפס NaN/Inf.</p>
<p><b>שאלה צפויה:</b> "האם ה-test שימש לבחירה כלשהי?" — <b>תשובה:</b> לא. ה-test נקרא פעם אחת בסוף כל שלב שנרשם מראש (הגריד החובה; אחר כך ההרחבה האופציונלית), על checkpoints שכבר ננעלו לפי ולידציה.</p>
<p><b>זמן משוער:</b> ~2 דקות.</p>
"""

CELLS = [("markdown", MAIN + NOTES(HEB))]
