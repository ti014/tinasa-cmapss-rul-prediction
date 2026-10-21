"""Danh tính học viên dùng chung khi bàn giao."""

from __future__ import annotations

AUTHOR_NAME = "Phan Nguyễn Mai Phương"
STUDENT_ID = "25002711"
AUTHOR_MARKDOWN = f"**Học viên:** {AUTHOR_NAME}  \n**MSHV:** {STUDENT_ID}"


def add_notebook_identity(notebook: dict) -> None:
    """Thêm danh tính dưới heading đầu mà không đổi cell thực thi."""
    first = notebook["cells"][0]
    if first["cell_type"] != "markdown":
        raise ValueError("Cell notebook đầu tiên phải là Markdown")
    source = first["source"]
    text = source if isinstance(source, str) else "".join(source)
    if AUTHOR_NAME not in text or STUDENT_ID not in text:
        heading, _, remainder = text.partition("\n")
        first["source"] = f"{heading}\n\n{AUTHOR_MARKDOWN}\n\n{remainder.lstrip()}"
    notebook.setdefault("metadata", {})["authors"] = [{"name": AUTHOR_NAME, "student_id": STUDENT_ID}]
