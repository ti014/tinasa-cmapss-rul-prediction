"""Shared identity and canonical artifact locations for project deliverables."""

from pathlib import Path

AUTHOR_NAME = "Phan Nguyễn Mai Phương"
STUDENT_ID = "25002711"
AUTHOR_MARKDOWN = f"**Học viên:** {AUTHOR_NAME}  \n**MSHV:** {STUDENT_ID}"
IMPROVEMENT_RUN = Path("outputs/experiments/rul_improvement/20260928_rebuild")
FOLLOWUP_RUN = Path("outputs/experiments/followup/20260929")


def add_notebook_identity(notebook: dict) -> None:
    """Add identity below the first heading without changing executable cells."""
    first = notebook["cells"][0]
    if first["cell_type"] != "markdown":
        raise ValueError("The first notebook cell must be Markdown")
    source = first["source"]
    text = source if isinstance(source, str) else "".join(source)
    if AUTHOR_NAME not in text or STUDENT_ID not in text:
        heading, _, remainder = text.partition("\n")
        first["source"] = f"{heading}\n\n{AUTHOR_MARKDOWN}\n\n{remainder.lstrip()}"
    notebook.setdefault("metadata", {})["authors"] = [{"name": AUTHOR_NAME, "student_id": STUDENT_ID}]
