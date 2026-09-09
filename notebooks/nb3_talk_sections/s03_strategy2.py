# Talk section 3 — Strategy 2 + compact S1-vs-S2 comparison table.


def NOTES(body):
    return ('\n<details><summary><b>Speaker notes · הערות למרצה (עברית)</b></summary>\n'
            '<div dir="rtl" style="text-align:right; line-height:1.65; margin:0.6em 1em;">\n'
            + body.strip() + '\n</div></details>\n')


MAIN = r"""
## 3 · Strategy 2 — classifier-guided targets + standard FM training

Each epoch has two phases. The frozen classifier is used **only to build targets**; the FM itself is trained by ordinary velocity regression.

**Phase 1 — build a nearby, better target for every training feature.** Start from the current FM output $\hat{z}$, projected into a **trust region centred on the source** $z$; then take $m$ normalized CE-gradient steps of size $\eta = \beta\rho$, re-projecting after each:

$$u_0 = \Pi_{B(z,\rho)}(\hat{z}),\qquad \rho = 0.1\,\lVert z\rVert,\qquad u_{j+1} = \Pi_{B(z,\rho)}\!\left(u_j - \eta\,\frac{\nabla_{u_j}\mathrm{CE}}{\lVert\nabla_{u_j}\mathrm{CE}\rVert + \varepsilon}\right)$$

The target $\hat{z}'$ is the **lowest-CE candidate** among $\{u_0,\dots,u_m\}$ (monotone acceptance), **detached and cached** for the epoch.

**Phase 2 — standard Flow-Matching regression toward the cached targets:** $t \sim U(0,1)$, $z_t = (1-t)\,z + t\,\hat{z}'$,

$$\mathcal{L}_{\mathrm{FM}} = \bigl\lVert v_\theta(z_t, t) - (\hat{z}' - z)\bigr\rVert^2$$

- **No CE gradient ever reaches the FM** — that separation is the scientific contrast with Strategy 1.
- The trust region is centred on the source $z$, not on $\hat{z}$: this prevents cumulative target drift across epochs.
- Grid $\beta \in \{0.25, 0.5, 1\} \times m \in \{1, 3\}$ on seed-0 validation. Selected: DTD $\beta = 0.25,\ m = 1$; FGVC-Aircraft $\beta = 0.5,\ m = 3$.

| | Strategy 1 | Strategy 2 |
|---|---|---|
| Trained parameters | FM only | FM only |
| Training loss | CE through the full rollout + λ · relative displacement | standard FM velocity regression toward cached targets |
| Role of the frozen classifier | inside the loss — its gradient flows through the rollout into the FM | builds the targets — its gradient never reaches the FM |
| How movement is limited | soft penalty on ‖ẑ − z‖² / ‖z‖² | hard trust region ρ = 0.1‖z‖ around z, lowest-CE acceptance |
| Validation-selected setting | λ = 1 (both datasets) | DTD: β = 0.25, m = 1 · FGVC-Aircraft: β = 0.5, m = 3 |
"""

HEB = """
<p><b>הנקודה המרכזית:</b> הדרך <u>העקיפה</u> — המסווג משמש רק לבניית "מטרה" קרובה ומשופרת לכל דוגמה, וה-FM מתאמן ברגרסיית מהירות סטנדרטית. אף גרדיאנט CE לא מגיע ל-FM.</p>
<p><b>מונחים בפשטות:</b> <b>Trust region</b> — כדור ברדיוס ρ = 0.1‖z‖ סביב הפיצ'ר המקורי z. <b>Projection Π</b> — אם יוצאים מהכדור, מחזירים אל שפתו. <b>Normalized CE-gradient step</b> — צעד בכיוון שמוריד את ה-CE, בגודל קבוע η = β·ρ. <b>Monotone acceptance</b> — בוחרים את המועמד עם ה-CE הנמוך ביותר, כך שהמטרה לעולם אינה גרועה מנקודת ההתחלה. <b>Detach + cache</b> — המטרות מחושבות פעם אחת בתחילת כל epoch ונשמרות כקבועים.</p>
<p><b>מה לומר על הנוסחאות והטבלה:</b> הנוסחה הראשונה — נקודת ההתחלה מוקרנת לתוך אזור האמון, ואז צעד גרדיאנט מנורמל והקרנה חוזרת. השנייה — הפסד FM רגיל: לחזות את הווקטור (ẑ′ − z) לאורך הקו הישר בין z ל-ẑ′. בטבלה להדגיש את השורה "Role of the frozen classifier" — זה ההבדל המדעי בין השיטות.</p>
<p><b>שאלה צפויה:</b> "למה אזור האמון ממורכז ב-z ולא ב-ẑ?" — <b>תשובה:</b> כדי למנוע סחיפה מצטברת בין epochs. אם המרכז היה זז יחד עם ẑ, חישוב חוזר של מטרות היה יכול "לייצר" פיצ'רים נוחים למסווג ללא גבול.</p>
<p><b>זמן משוער:</b> ~4 דקות.</p>
"""

CELLS = [("markdown", MAIN + NOTES(HEB))]
