"""Chứng minh reader chấp nhận whitespace và phát hiện raw sai cấu trúc."""

from pathlib import Path

import pandas as pd
import pytest

from cmapss.data.contracts import ContractError, OBSERVATION_COLUMNS
from cmapss.data.readers import read_observations, read_test_labels


def test_whitespace_reader_preserves_order_and_does_not_read_labels(tmp_path: Path) -> None:
    path = tmp_path / "observations.txt"
    rows = ["\t".join(["1", str(cycle)] + ["0.5"] * 24) for cycle in (2, 1)]
    path.write_text("  " + "  \n ".join(rows) + "  \n", encoding="utf-8")
    frame = read_observations(path, split="train")
    assert tuple(frame.columns) == OBSERVATION_COLUMNS
    assert frame["cycle"].tolist() == [2, 1]
    assert frame["engine_id"].dtype == "int64"
    assert frame["sensor_21"].dtype == "float64"
    assert "rul" not in frame
    assert not (tmp_path / "RUL_FD001.txt").exists()


@pytest.mark.parametrize("tokens,rule", [
    (["1", "1"] + ["0"] * 23, "RAW_WIDTH"),
    (["1", "1"] + ["0"] * 25, "RAW_WIDTH"),
    (["engine_id", "cycle"] + ["sensor"] * 24, "NUMERIC"),
    (["1.5", "1"] + ["0"] * 24, "NUMERIC"),
    ([str(2**63), "1"] + ["0"] * 24, "NUMERIC"),
    (["1", "1", "bad"] + ["0"] * 23, "NUMERIC"),
    (["1", "1", "nan"] + ["0"] * 23, "NUMERIC"),
    (["1", "1", "inf"] + ["0"] * 23, "NUMERIC"),
    (["0", "1"] + ["0"] * 24, "NUMERIC"),
])
def test_malformed_raw(tmp_path: Path, tokens: list[str], rule: str) -> None:
    path = tmp_path / "bad.txt"
    path.write_text(" ".join(tokens), encoding="utf-8")
    with pytest.raises(ContractError, match=rf"\[{rule}\].*bad.txt") as caught:
        read_observations(path, split="train")
    assert "data/CONTRACTS.md#" in str(caught.value)
    assert "Hành động:" in str(caught.value)


@pytest.mark.parametrize("content,rule", [("", "NONEMPTY"), ("\n", "RAW_WIDTH")])
def test_empty_files(tmp_path: Path, content: str, rule: str) -> None:
    path = tmp_path / "empty.txt"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ContractError, match=rule):
        read_observations(path, split="test")


def test_missing_field_in_later_row_is_not_padded(tmp_path: Path) -> None:
    path = tmp_path / "ragged.txt"
    path.write_text(" ".join(["1", "1"] + ["0"] * 24) + "\n1 2 0\n", encoding="utf-8")
    with pytest.raises(ContractError, match=r"\[RAW_WIDTH\].*dòng 2"):
        read_observations(path, split="test")


def test_invalid_split_rejected_before_file_read(tmp_path: Path) -> None:
    with pytest.raises(ContractError, match="SPLIT"):
        read_observations(tmp_path / "absent.txt", split="validation")


def test_label_line_mapping(tmp_path: Path) -> None:
    path = tmp_path / "labels.txt"
    path.write_text(" 30 \n\t10\n0\n", encoding="utf-8")
    frame = read_test_labels(path)
    assert frame.to_dict("list") == {
        "split": ["test"] * 3, "engine_id": [1, 2, 3], "rul": [30, 10, 0],
    }
    assert pd.api.types.is_integer_dtype(frame["rul"])


@pytest.mark.parametrize("content,rule", [
    ("30\n\n10\n", "RAW_WIDTH"), ("30 10\n", "RAW_WIDTH"),
    ("30\n-1\n", "NUMERIC"), ("1.5\n", "NUMERIC"), ("", "NONEMPTY"),
])
def test_bad_labels(tmp_path: Path, content: str, rule: str) -> None:
    path = tmp_path / "labels.txt"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ContractError, match=rule):
        read_test_labels(path)
