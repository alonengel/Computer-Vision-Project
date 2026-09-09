# Talk section 0 — title, research question, pipeline, hidden setup cell.


def NOTES(body):
    """Collapsed Hebrew speaker notes (visible content stays English)."""
    return ('\n<details><summary><b>Speaker notes · הערות למרצה (עברית)</b></summary>\n'
            '<div dir="rtl" style="text-align:right; line-height:1.65; margin:0.6em 1em;">\n'
            + body.strip() + '\n</div></details>\n')


TITLE = r"""
# Stage 3 · Flow Matching Before a Frozen Linear Classifier

**CVLAB Summer Project — few-shot classification with Flow Matching · DTD / ResNet-18 · FGVC-Aircraft / DINOv2**

<p style="font-size:1.5em; text-align:center; margin:1.1em 0 1.1em 0;">
Image → Frozen Encoder → <b>z</b> → <b>Flow Matching</b> → <b>ẑ</b> → Frozen Linear Classifier → Prediction
</p>

**Research question:** can a Flow-Matching transformation, inserted *before* the frozen Stage-1 linear classifier, turn frozen encoder features into a representation that the *existing* classifier separates better?

- The classifier is trained first, exactly as in Stage 1, then frozen; the FM starts as an exact identity, so before training the system *is* the Stage-1 probe.
- Two FM training strategies are compared against that probe on two datasets, with the test split sealed until one final read-out.

*Presentation version (≈ 28 min) of `stage3_presentation.ipynb`, the complete scientific report; every number here is read from the same tables of record. Reference material sits in the Appendix.*
"""

TITLE_NOTES = """
<p><b>הנקודה המרכזית:</b> שלב 3 שואל האם שכבת Flow Matching שמוכנסת <u>לפני</u> מסווג ליניארי קפוא יכולה "לשפר את הפיצ'רים" כך שאותו מסווג בדיוק יסווג טוב יותר.</p>
<p><b>מונחים בפשטות:</b> <b>Frozen encoder</b> — רשת מאומנת מראש (ResNet-18 / DINOv2) שרק מחלצת פיצ'רים ולא מתעדכנת. <b>z</b> — וקטור הפיצ'רים של תמונה. <b>Flow Matching (FM)</b> — רשת קטנה שלומדת "שדה מהירות" ומזיזה את z בכמה צעדי Euler קטנים אל ẑ. <b>Linear probe</b> — המסווג הליניארי s = Wz + b שאומן בשלב 1.</p>
<p><b>מה לומר על השקף:</b> לעבור על הצינור משמאל לימין ולהדגיש ששני הקצוות קפואים ורק ה-FM באמצע מתאמן. לפני האימון ה-FM הוא זהות מדויקת, ולכן נקודת ההתחלה היא בדיוק הביצועים של שלב 1 — כל שיפור נמדד ממנה.</p>
<p><b>שאלה צפויה:</b> "למה לא לאמן את המסווג יחד עם ה-FM?" — <b>תשובה:</b> זו הגדרת השלב במפרט: ייצוג טוב יותר עבור מסווג קיים. אימון משותף נבדק כהרחבה אופציונלית (נספח A.7) ולא הראה עדות ליתרון מעבר לאימון המסווג לבד.</p>
<p><b>זמן משוער:</b> ~2 דקות (סה"כ מתוכנן ~28 דקות ל-10 השקפים).</p>
"""

SETUP_CODE = r"""
# Hidden setup — paths, config, display-name maps. Everything downstream READS
# the tables, histories and figures of record under results/; nothing is
# recomputed or written back.
import io
import sys
from pathlib import Path

REPO = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(REPO))

import json

import numpy as np
import pandas as pd
from IPython.display import Image, Markdown, display

from src.utils import load_config                       # sets KMP_DUPLICATE_LIB_OK before torch
from src.visualize import dataset_label, encoder_label  # display-name maps (+ figure theme)

pd.set_option("display.precision", 2)
pd.set_option("display.max_rows", 200)
cfg = load_config()
MET = REPO / "results" / "metrics"
FIG = REPO / "results" / "figures"
"""

CELLS = [
    ("markdown", TITLE + NOTES(TITLE_NOTES)),
    ("code", SETUP_CODE),
]
