"""Đọc C-MAPSS observations; giữ observations và test labels riêng biệt."""

from __future__ import annotations


from pathlib import Path

import pandas as pd

from cmapss.data.contracts import RAW_DTYPES, SPLITS, ContractError, Split
from cmapss.data.validation import validate_observations, validate_test_labels


def _read_table(path: Path, dtypes: dict[str, str]) -> pd.DataFrame:
    """Kiểm tra số trường trước khi chuyển kiểu để tránh tự bù cột thiếu."""
    rows = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            fields = line.split()
            if len(fields) != len(dtypes):
                raise ContractError(
                    "RAW_WIDTH",
                    f"{path.name}: dòng {line_number}",
                    f"cần {len(dtypes)} cột, nhận {len(fields)}",
                )
            rows.append(fields)

    if not rows:
        raise ContractError("NONEMPTY", path.name, "file không có dữ liệu")

    data = pd.DataFrame(rows, columns=list(dtypes))
    for column, dtype in dtypes.items():
        try:
            data[column] = data[column].astype(dtype)
        except (ValueError, OverflowError) as error:
            raise ContractError(
                "NUMERIC", f"{path.name}.{column}", f"cần giá trị chuyển được sang {dtype}"
            ) from error
    return data


def read_observations(path: str | Path, *, split: Split) -> pd.DataFrame:
    """Đọc và kiểm tra file 26 cột không header; thêm cột split, giữ thứ tự dòng."""
    path = Path(path)
    if split not in SPLITS:
        raise ContractError("SPLIT", path.name, "split phải là train hoặc test")

    data = _read_table(path, RAW_DTYPES)
    data.insert(0, "split", pd.Series(split, index=data.index, dtype="string"))
    validate_observations(data, source=path.name)
    return data


def read_test_labels(path: str | Path) -> pd.DataFrame:
    """Đọc file nhãn test (một cột RUL); dòng thứ i thuộc test engine i.

    Trả về cột split, engine_id, rul. Chỉ dùng cho đánh giá cuối, không dùng khi huấn
    luyện.
    """
    path = Path(path)
    labels = _read_table(path, {"rul": "int64"})
    labels.insert(0, "engine_id", pd.Series(range(1, len(labels) + 1), dtype="int64"))
    labels.insert(0, "split", pd.Series("test", index=labels.index, dtype="string"))
    validate_test_labels(labels, source=path.name)
    return labels
