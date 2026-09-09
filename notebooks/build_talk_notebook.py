"""Assemble a Stage-3 TALK notebook — a presentation-focused companion of
``stage3_presentation.ipynb`` — from a directory of section modules.

Variants (``python notebooks/build_talk_notebook.py [v1|v2]``, default v1):

* ``v1``: ``nb3_talk_sections/``    -> ``stage3_presentation_talk.ipynb``
* ``v2``: ``nb3_talk_v2_sections/`` -> ``stage3_presentation_talk_v2.ipynb``

Same convention as ``build_notebook.py`` (each section module exports ``CELLS``,
a list of ``("markdown"|"code", source)`` tuples), extended with an optional
third element carrying cell metadata.  Defaults applied here:

* code cells start with their **input collapsed** (JupyterLab / VS Code honour
  ``jupyter.source_hidden``; nbconvert templates honour the ``hide-input`` tag)
  while their **outputs stay visible**;
* a markdown cell may ask to start **collapsed as a heading**
  (``jp-MarkdownHeadingCollapsed`` — JupyterLab >= 4; ``heading_collapsed`` —
  the classic collapsible-headings extension).

This script writes ONLY the selected variant's notebook.  The scientific
notebook, its sections (``nb3_sections/``) and ``build_notebook.py`` are
untouched — the talk notebooks re-read the tables, histories, figures and
models of record from ``results/`` and never recompute or rewrite any of them.

Execute the built notebook with
    python -m jupyter nbconvert --to notebook --execute --inplace notebooks/<name>.ipynb
"""
import importlib.util
import sys
from pathlib import Path

import nbformat

HERE = Path(__file__).resolve().parent
VARIANTS = {"v1": ("nb3_talk_sections", "stage3_presentation_talk.ipynb"),
            "v2": ("nb3_talk_v2_sections", "stage3_presentation_talk_v2.ipynb")}

CODE_META = {"jupyter": {"source_hidden": True}, "tags": ["hide-input"]}


def load_cells(sections_dir):
    cells = []
    for path in sorted((HERE / sections_dir).glob("s*.py")):
        spec = importlib.util.spec_from_file_location(path.stem, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        cells.extend(mod.CELLS)
        print(f"  {path.name}: {len(mod.CELLS)} cells")
    return cells


def main(variant="v1"):
    sections_dir, out_name = VARIANTS[variant]
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3",
                                 "language": "python"}
    for entry in load_cells(sections_dir):
        kind, source = entry[0], entry[1]
        extra = dict(entry[2]) if len(entry) > 2 else {}
        if kind == "markdown":
            cell = nbformat.v4.new_markdown_cell(source.strip())
        else:
            cell = nbformat.v4.new_code_cell(source.strip())
            cell.metadata.update({"jupyter": dict(CODE_META["jupyter"]),
                                  "tags": list(CODE_META["tags"])})
        cell.metadata.update(extra)
        nb.cells.append(cell)
    nbformat.validate(nb)
    out = HERE / out_name
    nbformat.write(nb, out)
    print(f"built {out} ({len(nb.cells)} cells)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "v1")
