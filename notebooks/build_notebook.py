"""Assemble a stage's presentation notebook from modular sections.

Stage 1: nb_sections/s*.py  -> stage1_presentation.ipynb  (default)
Stage 2: nb2_sections/s*.py -> stage2_presentation.ipynb  (`build_notebook.py 2`)

Each section module exports CELLS: a list of ("markdown"|"code", source) tuples.
Keeping sections as plain .py files makes the notebook lintable and diffable;
the built .ipynb is a generated artifact executed via `tasks.ps1 notebook[2]`.
"""
import importlib.util
import sys
from pathlib import Path

import nbformat

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

STAGES = {"1": ("nb_sections", "stage1_presentation.ipynb"),
          "2": ("nb2_sections", "stage2_presentation.ipynb")}


def load_cells(sections_dir):
    cells = []
    for path in sorted((HERE / sections_dir).glob("s*.py")):
        spec = importlib.util.spec_from_file_location(path.stem, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        cells.extend(mod.CELLS)
        print(f"  {path.name}: {len(mod.CELLS)} cells")
    return cells


def main(stage="1"):
    sections_dir, out_name = STAGES[stage]
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    for kind, source in load_cells(sections_dir):
        maker = nbformat.v4.new_markdown_cell if kind == "markdown" else nbformat.v4.new_code_cell
        nb.cells.append(maker(source.strip()))
    out = HERE / out_name
    nbformat.write(nb, out)
    print(f"built {out} ({len(nb.cells)} cells)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "1")
