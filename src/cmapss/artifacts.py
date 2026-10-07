"""Ghi bộ artifact qua staging, bảo vệ input và rollback khi publication lỗi.

Thay nhiều file không phải atomic transaction cho reader đồng thời. Helper
này bảo đảm lỗi ghi/publish được xử lý không để lại bộ artifact trộn phiên bản.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import warnings
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path


class ArtifactRollbackError(RuntimeError):
    """Publication và rollback đều lỗi; backup được giữ để phục hồi thủ công."""

    def __init__(self, recovery_dir: Path) -> None:
        self.recovery_dir = recovery_dir
        super().__init__(f"Không khôi phục đủ artifact; backup được giữ tại {recovery_dir}")


class ArtifactCleanupWarning(RuntimeWarning):
    """Artifact đã publish thành công nhưng workspace tạm chưa dọn được."""


def default_raw_dir() -> Path:
    """Neo Bronze vào checkout; wheel dùng CWD tại lúc pipeline lấy boundary."""
    checkout = Path(__file__).resolve().parents[2]
    if (checkout / "pyproject.toml").is_file() and (checkout / "src" / "cmapss").is_dir():
        return checkout / "data" / "raw"
    return Path("data/raw").resolve()


def _inside(path: Path, directory: Path) -> bool:
    """Kiểm tra path bằng hoặc nằm trong directory đã resolve."""
    return path == directory or directory in path.parents


def _validate_destinations(
    output: Path,
    paths: Sequence[Path],
    protected_dirs: Sequence[Path],
    protected_files: Sequence[Path],
) -> None:
    """Chặn path traversal, Bronze, input overwrite và symlink thoát output."""
    raw_dirs = {default_raw_dir().resolve(), *(path.resolve() for path in protected_dirs)}
    inputs = {path.resolve() for path in protected_files}
    if any(_inside(output, raw) for raw in raw_dirs):
        raise ValueError("Artifact output không được ghi vào thư mục raw (Bronze immutable)")
    if len(set(paths)) != len(paths) or not paths:
        raise ValueError("Artifact paths phải không rỗng và không trùng nhau")
    for relative in paths:
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise ValueError(f"Artifact path phải tương đối, nằm trong output: {relative}")
        target = (output / relative).resolve()
        if any(_inside(target, raw) for raw in raw_dirs):
            raise ValueError(f"Artifact đích nằm trong raw: {target}")
        if target in inputs:
            raise ValueError(f"Artifact output không được ghi đè input: {target}")
        if not _inside(target, output):
            raise ValueError(f"Artifact symlink/junction thoát thư mục output: {target}")
        if (output / relative).is_symlink():
            raise ValueError(f"Artifact đích không được là symlink file: {output / relative}")
        if (output / relative).exists() and not (output / relative).is_file():
            raise ValueError(f"Artifact đích phải là file: {output / relative}")
    for relative in paths:
        if any(other in relative.parents for other in paths):
            raise ValueError("Artifact file không được làm thư mục cha của artifact khác")


def _remove_workspace(workspace: Path, parent: Path) -> None:
    """Chỉ xóa workspace riêng đã xác minh nằm trong parent dự kiến."""
    resolved = workspace.resolve()
    if resolved.parent != parent.resolve() or not resolved.name.startswith(".cmapss-write-"):
        raise ValueError(f"Workspace cleanup nằm ngoài phạm vi: {resolved}")
    shutil.rmtree(resolved)


def _mkdir_recorded(directory: Path, created_dirs: list[Path]) -> None:
    """Tạo thư mục và ghi lại các thư mục mới để dọn khi rollback."""
    missing = []
    cursor = directory
    while not cursor.exists():
        missing.append(cursor)
        cursor = cursor.parent
    for path in reversed(missing):
        path.mkdir()
        created_dirs.append(path)


def _publish(output: Path, workspace: Path, paths: Sequence[Path]) -> None:
    """Backup trước mọi replace; rollback các file đã publish nếu có lỗi."""
    staged = workspace / "new"
    backup = workspace / "backup"
    existing: set[Path] = set()
    for relative in paths:
        if not (staged / relative).is_file() or (staged / relative).is_symlink():
            raise ValueError(f"Staging chưa có artifact file hoàn chỉnh: {relative}")
        destination = output / relative
        if destination.exists():
            saved = backup / relative
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(destination, saved)
            existing.add(relative)

    installed: list[Path] = []
    created_dirs: list[Path] = []
    try:
        for relative in paths:
            destination = output / relative
            _mkdir_recorded(destination.parent, created_dirs)
            os.replace(staged / relative, destination)
            installed.append(relative)
    except BaseException as publication_error:
        failures = []
        for relative in reversed(installed):
            try:
                if relative in existing:
                    os.replace(backup / relative, output / relative)
                else:
                    (output / relative).unlink()
            except OSError as error:
                failures.append(error)
        for directory in reversed(created_dirs):
            try:
                directory.rmdir()
            except OSError as error:
                failures.append(error)
        if failures:
            recovery_error = ArtifactRollbackError(workspace)
            for error in failures:
                recovery_error.add_note(str(error))
            raise recovery_error from publication_error
        raise


@contextmanager
def staged_artifact_bundle(
    output_dir: str | Path,
    relative_paths: Sequence[str | Path],
    *,
    protected_dirs: Sequence[str | Path] = (),
    protected_files: Sequence[str | Path] = (),
) -> Iterator[Path]:
    """Ghi staging rồi publish đúng thứ tự, metadata phải đứng cuối.

    Args:
        output_dir: Thư mục artifact. Các file ngoài relative_paths được giữ.
        relative_paths: File writer sở hữu, theo thứ tự publication.
        protected_dirs: Bronze thực tế, bổ sung cho Bronze mặc định của project.
        protected_files: Input files và manifest không được ghi đè.

    Yields:
        Thư mục staging để caller ghi đủ artifact trước publication.

    Raises:
        ValueError: Đích vi phạm boundary hoặc staging chưa đầy đủ.
        OSError: Ghi, backup hoặc publication lỗi; rollback đã hoàn tất.
        ArtifactRollbackError: Rollback cũng lỗi, workspace được giữ lại.

    Example:
        >>> with staged_artifact_bundle(output, ['data.parquet', 'run_info.json']) as stage:
        ...     write_data_and_manifest(stage)

    Không bảo đảm reader đồng thời, nhiều writer đồng thời hoặc mất điện đột
    ngột. Backup được giữ nếu rollback lỗi; không nuốt lỗi publication.
    """
    output = Path(output_dir).resolve()
    paths = tuple(Path(path) for path in relative_paths)
    raw_dirs = tuple(Path(path).resolve() for path in protected_dirs)
    inputs = tuple(Path(path).resolve() for path in protected_files)
    _validate_destinations(output, paths, raw_dirs, inputs)
    output.parent.mkdir(parents=True, exist_ok=True)
    workspace = Path(tempfile.mkdtemp(prefix=".cmapss-write-", dir=output.parent))
    staged = workspace / "new"
    staged.mkdir()
    retain_backup = False
    pending_error: BaseException | None = None
    try:
        for relative in paths:
            (staged / relative).parent.mkdir(parents=True, exist_ok=True)
        yield staged
        _validate_destinations(output, paths, raw_dirs, inputs)
        _publish(output, workspace, paths)
    except ArtifactRollbackError:
        retain_backup = True
        raise
    except BaseException as error:
        pending_error = error
        raise
    finally:
        if not retain_backup:
            try:
                _remove_workspace(workspace, output.parent)
            except OSError as cleanup_error:
                if pending_error is None:
                    warnings.warn(
                        f"Artifact đã publish thành công; chưa dọn workspace {workspace}: {cleanup_error}",
                        ArtifactCleanupWarning,
                        stacklevel=2,
                    )
                else:
                    pending_error.add_note(f"Staging cleanup lỗi; giữ tại {workspace}: {cleanup_error}")
