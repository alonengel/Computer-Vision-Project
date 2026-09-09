# Talk section 2 — Strategy 1.


def NOTES(body):
    return ('\n<details><summary><b>Speaker notes · הערות למרצה (עברית)</b></summary>\n'
            '<div dir="rtl" style="text-align:right; line-height:1.65; margin:0.6em 1em;">\n'
            + body.strip() + '\n</div></details>\n')


MAIN = r"""
## 2 · Strategy 1 — end-to-end rolled-out classification training

- Run the full $T$-step rollout $z \to \hat{z}$, pass $\hat{z}$ through the **frozen** classifier, and backpropagate the cross-entropy **through the whole rollout** — updating **only** the FM parameters.
- A **relative displacement regularizer** discourages unnecessarily large moves (scale-free, so it means the same thing on both datasets):

$$\mathcal{L}_{\mathrm{cls}} = \mathrm{CE}(W\hat{z} + b,\, y) \;+\; \lambda\,\mathrm{mean}_i\,\frac{\lVert\hat{z}_i - z_i\rVert^2}{\lVert z_i\rVert^2 + \varepsilon}$$

- $\lambda \in \{0, 1, 10, 100\}$ is selected on seed-0 validation, and the $\lambda = 0$ variant is always test-reported next to the winner (pre-registered pair). Selected: $\lambda = 1$ on both datasets.
"""

HEB = """
<p><b>הנקודה המרכזית:</b> הדרך <u>הישירה</u> — מריצים את כל ה-rollout, מעבירים במסווג הקפוא, ומחזירים את גרדיאנט ה-CE דרך כל הצעדים אל ה-FM בלבד.</p>
<p><b>מונחים בפשטות:</b> <b>Rollout</b> — הרצת T הצעדים ברצף מ-z ל-ẑ. <b>Backprop through the rollout</b> — הגרדיאנט עובר אחורה דרך כל ארבעת הצעדים. <b>Relative displacement</b> — גודל התזוזה ‖ẑ−z‖² מחולק ב-‖z‖², כלומר ביחידות יחסיות, ולכן ניתן להשוואה בין דאטהסטים עם נורמות שונות. <b>λ</b> — משקל הרגולריזציה.</p>
<p><b>מה לומר על הנוסחה:</b> האיבר הראשון הוא ה-CE של המסווג הקפוא על ẑ (W ו-b קבועים; רק ה-FM מתאמן). האיבר השני הוא קנס על תזוזה גדולה. λ נבחר על ולידציה של seed 0 מתוך {0, 1, 10, 100} — יצא λ = 1 בשני הדאטהסטים; גם λ = 0 מדווח על ה-test כזוג שנרשם מראש (נספח A.3).</p>
<p><b>שאלה צפויה:</b> "האם המסווג מתעדכן?" — <b>תשובה:</b> לא. W ו-b קפואים; הגרדיאנט רק זורם <u>דרכם</u> אל ה-FM.</p>
<p><b>זמן משוער:</b> ~3 דקות.</p>
"""

CELLS = [("markdown", MAIN + NOTES(HEB))]
