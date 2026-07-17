"""Assemble notebooks/stage1_presentation.ipynb from modular sections.

Each nb_sections/s*.py module exports CELLS: a list of ("markdown"|"code", source)
tuples. Keeping sections as plain .py files makes the notebook lintable and diffable;
the built .ipynb is a generated artifact executed via `tasks.ps1 notebook`.
"""
import importlib.util
import sys
from pathlib import Path

import nbformat

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))


def load_cells():
    cells = []
    for path in sorted((HERE / "nb_sections").glob("s*.py")):
        spec = importlib.util.spec_from_file_location(path.stem, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        cells.extend(mod.CELLS)
        print(f"  {path.name}: {len(mod.CELLS)} cells")
    return cells


def main():
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    for kind, source in load_cells():
        maker = nbformat.v4.new_markdown_cell if kind == "markdown" else nbformat.v4.new_code_cell
        nb.cells.append(maker(source.strip()))
    out = HERE / "stage1_presentation.ipynb"
    nbformat.write(nb, out)
    print(f"built {out} ({len(nb.cells)} cells)")


if __name__ == "__main__":
    main()
