"""Assemble the Stage-3 TALK notebook — the presentation-focused companion of
``stage3_presentation.ipynb`` — from ``notebooks/nb3_talk_sections/s*.py``.

Same convention as ``build_notebook.py`` (each section module exports ``CELLS``,
a list of ``("markdown"|"code", source)`` tuples), extended with an optional
third element carrying cell metadata.  Defaults applied here:

* code cells start with their **input collapsed** (JupyterLab / VS Code honour
  ``jupyter.source_hidden``; nbconvert templates honour the ``hide-input`` tag)
  while their **outputs stay visible**;
* a markdown cell may ask to start **collapsed as a heading**
  (``jp-MarkdownHeadingCollapsed`` — JupyterLab ≥ 4; ``heading_collapsed`` —
  the classic collapsible-headings extension).

This script writes ONLY ``notebooks/stage3_presentation_talk.ipynb``.  The
scientific notebook, its sections (``nb3_sections/``) and ``build_notebook.py``
are untouched — the talk notebook re-reads the tables, histories and figures of
record from ``results/`` and never recomputes or rewrites any of them.

Usage (from the repository root, with the project venv):
    python notebooks/build_talk_notebook.py
    python -m jupyter nbconvert --to notebook --execute --inplace notebooks/stage3_presentation_talk.ipynb
"""
import importlib.util
import sys
from pathlib import Path

import nbformat

HERE = Path(__file__).resolve().parent
SECTIONS_DIR = "nb3_talk_sections"
OUT_NAME = "stage3_presentation_talk.ipynb"

CODE_META = {"jupyter": {"source_hidden": True}, "tags": ["hide-input"]}


def load_cells():
    cells = []
    for path in sorted((HERE / SECTIONS_DIR).glob("s*.py")):
        spec = importlib.util.spec_from_file_location(path.stem, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        cells.extend(mod.CELLS)
        print(f"  {path.name}: {len(mod.CELLS)} cells")
    return cells


def main():
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3",
                                 "language": "python"}
    for entry in load_cells():
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
    out = HERE / OUT_NAME
    nbformat.write(nb, out)
    print(f"built {out} ({len(nb.cells)} cells)")


if __name__ == "__main__":
    main()
