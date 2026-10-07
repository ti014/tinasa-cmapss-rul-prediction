"""Đọc C-MAPSS observations; giữ observations và test labels riêng biệt."""

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
    """Đọc và kiểm tra file 26 cột; thêm split, giữ nguyên thứ tự dòng.

    Args:
        path: File observations không header.
        split: Nguồn của file, phải là train hoặc test.

    Returns:
        DataFrame gồm split, engine_id, cycle, 3 settings và 21 sensors.

    Raises:
        ContractError: Schema, kiểu dữ liệu, khóa hoặc cycle không hợp lệ.
        OSError: Không đọc được file.
    """
    path = Path(path)
    if split not in SPLITS:
        raise ContractError("SPLIT", path.name, "split phải là train hoặc test")

    data = _read_table(path, RAW_DTYPES)
    data.insert(0, "split", pd.Series(split, index=data.index, dtype="string"))
    validate_observations(data, source=path.name)
    return data


def read_test_labels(path: str | Path) -> pd.DataFrame:
    """Đọc riêng test labels; dòng thứ i thuộc test engine i.

    Chỉ gọi khi kiểm tra tính toàn vẹn hoặc đánh giá cuối. Hàm không đọc
    observations, không join nhãn và không thuộc luồng huấn luyện.

    Args:
        path: File một cột RUL nguyên, không âm, không header.

    Returns:
        DataFrame gồm split, engine_id và rul.

    Raises:
        ContractError: Schema hoặc giá trị nhãn không hợp lệ.
        OSError: Không đọc được file.
    """
    path = Path(path)
    labels = _read_table(path, {"rul": "int64"})
    labels.insert(0, "engine_id", pd.Series(range(1, len(labels) + 1), dtype="int64"))
    labels.insert(0, "split", pd.Series("test", index=labels.index, dtype="string"))
    validate_test_labels(labels, source=path.name)
    return labels
