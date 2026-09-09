# Talk section 1 — why Stage 3, in four bullets.


def NOTES(body):
    return ('\n<details><summary><b>Speaker notes · הערות למרצה (עברית)</b></summary>\n'
            '<div dir="rtl" style="text-align:right; line-height:1.65; margin:0.6em 1em;">\n'
            + body.strip() + '\n</div></details>\n')


MAIN = r"""
## 1 · Why Stage 3 — the setup in four bullets

- **Encoder and Stage-1 linear probe are frozen.** The probe is retrained per (dataset, subset seed) with the exact Stage-1 recipe and pinned: it is the baseline *and* the classifier inside the pipeline.
- **Only the FM is trained.** Velocity MLP $d{+}1 \to 512 \to 512 \to d$ (SiLU), $T = 4$ Euler steps, $\hat{z}_{k+1} = \hat{z}_k + \tfrac{1}{T}\,v_\theta(\hat{z}_k, k/T)$, in the **raw** feature space the probe was trained in.
- **Exact identity at initialization.** The final velocity layer starts at zero (weight *and* bias), so $\hat{z} = z$ bit-for-bit: the pipeline starts at exactly the probe's accuracy, and every Δ is measured from there.
- **Goal: better features for the existing classifier** — not a new classifier. Any gain must come from moving $z$ to a $\hat{z}$ that the fixed boundary $W x + b$ separates better.
"""

HEB = """
<p><b>הנקודה המרכזית:</b> ארבע עובדות שמגדירות את הניסוי — מה קפוא, מה מתאמן, מאיפה מתחילים (זהות מדויקת), ומה המטרה.</p>
<p><b>מונחים בפשטות:</b> <b>Velocity network</b> — MLP שמקבל (ẑ_k, t) ומחזיר כיוון תזוזה. <b>Euler steps</b> — T = 4 צעדים קטנים שמצטברים לתזוזה הכוללת. <b>Raw feature space</b> — עובדים על הפיצ'רים כפי שהם, בלי נרמול, כי המסווג אומן עליהם כך (בשלב 2 המסווג היה מבוסס קוסינוס ולכן שם עבדנו על הספירה). <b>Identity at init</b> — השכבה האחרונה של רשת המהירות מאותחלת לאפס ולכן ẑ = z בדיוק.</p>
<p><b>מה לומר על הנוסחה:</b> צעד Euler — כל צעד מוסיף 1/T כפול המהירות; ארבעה צעדים כאלה. ה-FM לא מחליף את המסווג אלא "מכין" לו קלט. ה-FM הוא רשת קטנה (~0.8M פרמטרים ב-DTD, ~0.66M ב-FGVC).</p>
<p><b>שאלה צפויה:</b> "למה זהות <u>מדויקת</u> ולא רק 'קרוב לזהות'?" — <b>תשובה:</b> כך ה-Δ נמדד מנקודת התחלה שהיא בדיוק ה-probe של שלב 1 (נבדק ברמת ביט, לוגיטים ותחזיות — נספח A.2), ואין שינוי ביצועים "חינם" שנובע מרעש אתחול.</p>
<p><b>זמן משוער:</b> ~2 דקות.</p>
"""

CELLS = [("markdown", MAIN + NOTES(HEB))]
