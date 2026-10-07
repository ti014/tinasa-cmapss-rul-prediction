"""Add the required student identity while preserving notebook outputs."""

from pathlib import Path

import nbformat

from cmapss.project import add_notebook_identity


def main() -> None:
    """Update every project notebook, including the manually authored notebooks."""
    root = Path(__file__).resolve().parents[1]
    for path in sorted((root / "notebooks").glob("*.ipynb")):
        notebook = nbformat.read(path, as_version=4)
        add_notebook_identity(notebook)
        nbformat.validate(notebook)
        nbformat.write(notebook, path)
        print(path.relative_to(root))


if __name__ == "__main__":
    main()
