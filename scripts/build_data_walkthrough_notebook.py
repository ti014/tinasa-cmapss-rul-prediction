"""Build notebooks/00_data_pipeline_walkthrough.ipynb.

The notebook walks through how raw C-MAPSS files are combined, stored, queried,
transformed into features and joined, using the package functions that the
pipeline itself uses. It only reads raw files and existing artifacts; it never
writes to outputs/.

Usage (from the repository root):
    .venv/Scripts/python.exe scripts/build_data_walkthrough_notebook.py [--execute]
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import nbformat as nbf

from cmapss.project import add_notebook_identity

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "notebooks" / "00_data_pipeline_walkthrough.ipynb"
cells: list = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(text).strip()))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(textwrap.dedent(text).strip()))


md("""
# Từ file raw đến dữ liệu cho mô hình: gom, lưu trữ, truy vấn, biến đổi và quan hệ

Notebook này chạy lại từng bước xử lý dữ liệu của pipeline trên FD001 và in
kết quả trung gian để trả lời năm câu hỏi:

1. Dữ liệu có nhiều file thì gom lại như thế nào?
2. Dữ liệu được lưu ở đâu và truy vấn ra sao (SQL trên SQLite, Spark SQL trên Parquet)?
3. Các bước biến đổi (transform) diễn ra theo thứ tự nào?
4. Những cột nào được dùng làm đặc trưng cho mô hình?
5. Quan hệ giữa các bảng là một-một, một-nhiều hay nhiều-nhiều?

Notebook gọi đúng các hàm mà pipeline dùng (`cmapss.data`, `cmapss.etl`,
`cmapss.features`, `cmapss.modeling`), chỉ đọc file raw và artifact đã có, không
ghi gì vào `outputs/`. Chạy bằng **Restart Kernel and Run All Cells** từ thư mục
gốc repository.

```text
data/raw/*.txt  --đọc + thêm split-->  union theo dòng  -->  Silver: observations.parquet
                                                         -->  Silver: train_targets.parquet (RUL)
                                                         -->  Gold:   engine_summary (Parquet + SQLite)
                                                         -->  Gold:   rolling_features.parquet (126 đặc trưng)
                                     join theo (split, engine_id, cycle)  -->  X (126 cột), y (rul)
```
""")

code("""
from pathlib import Path
import os
import sqlite3
import sys

import numpy as np
import pandas as pd

ROOT = Path.cwd()
if not (ROOT / "src" / "cmapss").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 12)

from cmapss.data.contracts import KEY
from cmapss.data.readers import read_observations
from cmapss.data.targets import make_train_targets
from cmapss.etl.pandas_pipeline import build_engine_summary, load_dataset
from cmapss.features.rolling import build_rolling_features
from cmapss.features.contracts import ROLLING_FEATURE_STATS
from cmapss.modeling.contracts import MODELING_FEATURE_COLUMNS
from cmapss.modeling.pipeline import prepare_modeling_dataset

RAW = ROOT / "data" / "raw"
print("Khóa của một quan sát:", KEY)
""")

md("""
## 1. Dữ liệu gốc gồm những file nào?

Mỗi subset có ba file text không header. Hai file quan sát (`train`, `test`) có
cùng 26 cột; file `RUL_*.txt` chỉ có một cột là RUL thật tại cycle cuối của mỗi
động cơ test.
""")

code("""
rows = []
for name in ["train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt"]:
    lines = (RAW / name).read_text(encoding="utf-8").splitlines()
    rows.append({"file": name, "số dòng": len(lines), "số cột": len(lines[0].split()),
                 "dòng đầu tiên (rút gọn)": " ".join(lines[0].split()[:7]) + " ..."})
pd.DataFrame(rows)
""")

md("""
## 2. Gom nhiều file thành một bảng

**Cách gom:** đọc từng file, gán tên cột và kiểu dữ liệu, thêm cột `split`, rồi
**nối theo chiều dòng** (union). Không join train với test, vì hai file chứa hai
nhóm động cơ khác nhau với cùng cấu trúc cột.

`read_observations` kiểm tra mỗi dòng có đúng 26 trường, giá trị là số hữu hạn,
khóa không trùng và cycle liên tục; nếu sai sẽ báo `ContractError`.
""")

code("""
train = read_observations(RAW / "train_FD001.txt", split="train")
test = read_observations(RAW / "test_FD001.txt", split="test")
print("train:", train.shape, "| test:", test.shape)
train.iloc[:3, :8]
""")

code("""
observations = pd.concat([train, test], ignore_index=True)   # PySpark: train.unionByName(test)
print("Sau khi nối:", observations.shape)
print("Khóa (split, engine_id, cycle) duy nhất:", not observations.duplicated(list(KEY)).any())
print("Khóa (engine_id, cycle) duy nhất:", not observations.duplicated(["engine_id", "cycle"]).any())
observations.query("engine_id == 1 and cycle == 1").iloc[:, :6]
""")

md("""
Dòng thứ hai trong kết quả trên cho thấy vì sao phải có cột `split`: train engine 1
và test engine 1 là hai động cơ khác nhau nhưng có cùng `(engine_id, cycle)`.
Vì vậy khóa của một quan sát là **`(split, engine_id, cycle)`**.

File nhãn `RUL_FD001.txt` **không được gom** vào bảng này. Nó chỉ được đọc ở
bước đánh giá cuối, sau khi dự đoán test đã được ghi, để tránh data leakage.
Bốn subset FD001--FD004 được xử lý ở các thư mục riêng; muốn gộp cả bốn thì phải
thêm cột mã subset vào khóa.
""")

md("""
## 3. Lưu trữ theo lớp và đối chiếu với artifact

Hàm `load_dataset` của ETL thực hiện đúng các bước trên (đọc, nối, sắp theo khóa)
và tạo nhãn train. Ô dưới đây so sánh kết quả tính lại với file Parquet đã ghi.
""")

code("""
silver_obs, silver_targets, _ = load_dataset(RAW, "FD001")
stored_obs = pd.read_parquet("outputs/etl/silver/observations.parquet")
stored_targets = pd.read_parquet("outputs/etl/silver/train_targets.parquet")
pd.testing.assert_frame_equal(silver_obs, stored_obs, check_dtype=False)
pd.testing.assert_frame_equal(silver_targets, stored_targets, check_dtype=False)

layers = [
    ("Bronze", "data/raw/train_FD001.txt + test_FD001.txt", "1 cycle của 1 động cơ", len(train) + len(test), 26),
    ("Silver", "observations.parquet", "1 cycle của 1 động cơ", len(stored_obs), stored_obs.shape[1]),
    ("Silver", "train_targets.parquet", "1 cycle train", len(stored_targets), stored_targets.shape[1]),
    ("Gold", "engine_summary.parquet / etl.sqlite", "1 động cơ", len(pd.read_parquet("outputs/etl/gold/engine_summary.parquet")), 5),
    ("Gold", "rolling_features.parquet", "1 window 30 cycles", *pd.read_parquet("outputs/features/gold/rolling_features.parquet").shape),
]
pd.DataFrame(layers, columns=["lớp", "file", "một dòng là", "số dòng", "số cột"])
""")

md("""
## 4. Truy vấn dữ liệu

### 4.1 SQL trên SQLite

SQLite chứa hai bảng Gold/Silver nhỏ để truy vấn nhanh bằng SQL: `engine_summary`
và `train_targets`, mỗi bảng có index theo khóa.
""")

code("""
con = sqlite3.connect("outputs/etl/gold/etl.sqlite")
print(pd.read_sql("SELECT type, name FROM sqlite_master ORDER BY type DESC, name", con))
pd.read_sql(\"\"\"
    SELECT s.split, s.engine_id, s.n_cycles, s.last_cycle, t.cycle, t.rul
    FROM engine_summary AS s
    JOIN train_targets AS t
      ON t.split = s.split AND t.engine_id = s.engine_id
    WHERE s.split = 'train' AND s.engine_id = 1 AND t.cycle IN (1, 100, 192)
    ORDER BY t.cycle
\"\"\", con)
""")

md("""
### 4.2 Lọc và chọn cột trên Parquet bằng Pandas

Dữ liệu sensor chi tiết nằm trong Parquet. Lọc dòng (selection) và chọn cột
(projection): lấy `sensor_11` của train engine 1, cycles 71--100.
""")

code("""
window_rows = stored_obs.loc[
    stored_obs["split"].eq("train") & stored_obs["engine_id"].eq(1) & stored_obs["cycle"].between(71, 100),
    ["split", "engine_id", "cycle", "sensor_11"],
]
print(len(window_rows), "dòng; trung bình sensor_11 =", round(window_rows["sensor_11"].mean(), 3))
window_rows.tail(3)
""")

md("""
### 4.3 Spark SQL trực tiếp trên Parquet

Cùng file Parquet có thể được truy vấn bằng Spark SQL. Spark chạy ở chế độ
local (`local[1]`) trên máy này; đây là minh họa cách truy vấn, không phải phép
đo hiệu năng trên cụm máy. Ô này cần Java; nếu không có Java, notebook bỏ qua và
in thông báo.

Truy vấn thứ ba dùng window function để tính trung bình 30 cycles gần nhất, tức
cùng phép tính với đặc trưng `sensor_11__mean`. Phép lọc cycle phải đặt **sau**
window (trong truy vấn ngoài); nếu đặt `WHERE` cùng cấp, window chỉ nhìn thấy các
dòng đã lọc và cho kết quả sai.
""")

code("""
java_home = ROOT / "tmp" / "java-runtime" / "jdk-17.0.20.1+1-jre"
if "JAVA_HOME" not in os.environ and java_home.exists():
    os.environ["JAVA_HOME"] = str(java_home)
    os.environ["PATH"] = str(java_home / "bin") + os.pathsep + os.environ["PATH"]

try:
    from cmapss.etl.spark_pipeline import create_spark_session
    spark = create_spark_session("data-walkthrough")
except Exception as error:  # Java hoặc PySpark không có sẵn
    spark = None
    print("Bỏ qua Spark SQL:", error)

if spark is not None:
    spark.read.parquet("outputs/etl/silver/observations.parquet").createOrReplaceTempView("observations")
    spark.read.parquet("outputs/etl/silver/train_targets.parquet").createOrReplaceTempView("train_targets")

    print("(a) GROUP BY: số dòng và số động cơ theo split")
    spark.sql(\"\"\"
        SELECT split, COUNT(*) AS n_rows, COUNT(DISTINCT engine_id) AS n_engines
        FROM observations GROUP BY split ORDER BY split
    \"\"\").show()

    print("(b) JOIN theo đủ ba khóa: sensor và nhãn RUL")
    spark.sql(\"\"\"
        SELECT o.cycle, o.sensor_11, t.rul
        FROM observations o
        JOIN train_targets t
          ON o.split = t.split AND o.engine_id = t.engine_id AND o.cycle = t.cycle
        WHERE o.split = 'train' AND o.engine_id = 1 AND o.cycle BETWEEN 98 AND 100
        ORDER BY o.cycle
    \"\"\").show()

    print("(c) Window function: trung bình 30 cycles gần nhất của sensor_11")
    spark.sql(\"\"\"
        SELECT engine_id, cycle, n_rows_in_window, ROUND(sensor_11_mean30, 3) AS sensor_11_mean30
        FROM (
            SELECT engine_id, cycle,
                   COUNT(*) OVER w AS n_rows_in_window,
                   AVG(sensor_11) OVER w AS sensor_11_mean30
            FROM observations
            WHERE split = 'train' AND engine_id = 1
            WINDOW w AS (PARTITION BY split, engine_id ORDER BY cycle
                         ROWS BETWEEN 29 PRECEDING AND CURRENT ROW)
        )
        WHERE cycle IN (29, 30, 100)
        ORDER BY cycle
    \"\"\").show()
    spark.stop()
""")

md("""
Ở truy vấn (c), cycle 29 mới có 29 dòng trong window nên pipeline **không** tạo
đặc trưng cho cycle này; từ cycle 30 window mới đủ 30 dòng. Giá trị tại cycle 100
(47,395) trùng với kết quả Pandas ở mục 4.2 và với cột `sensor_11__mean` trong
artifact đặc trưng ở mục 5.
""")

md("""
## 5. Các bước biến đổi (transform)

"Transform" ở đây là bước T trong ETL: đổi kiểu, gom dòng, tổng hợp, tạo nhãn và
tạo đặc trưng. Nó không liên quan đến mô hình Transformer.

### 5.1 Tạo nhãn RUL: nhóm theo động cơ, lấy cycle cuối
""")

code("""
targets = make_train_targets(train)          # RUL = max(cycle của động cơ) - cycle
targets.query("engine_id == 1 and cycle in [1, 100, 192]")
""")

md("### 5.2 Tổng hợp theo động cơ (Gold `engine_summary`)")

code("""
summary = build_engine_summary(observations)   # GROUP BY split, engine_id
print(summary.shape)
summary.query("engine_id == 1")
""")

md("""
### 5.3 Tạo đặc trưng bằng trailing window 30 cycles

Với mỗi động cơ, trượt một window 30 cycles và tính sáu thống kê cho từng sensor.
Ô dưới đây tính tay sáu thống kê của `sensor_11` tại cycle 100 của train engine 1,
rồi so với kết quả của hàm `build_rolling_features`.
""")

code("""
z = window_rows["sensor_11"].to_numpy()        # 30 giá trị, cycles 71..100
k = np.arange(1, len(z) + 1)
by_hand = {
    "last": z[-1], "mean": z.mean(), "std": z.std(ddof=1), "min": z.min(), "max": z.max(),
    "slope": ((k - k.mean()) * (z - z.mean())).sum() / ((k - k.mean()) ** 2).sum(),
}

features = build_rolling_features(silver_obs, window_cycles=30)
row = features.query("split == 'train' and engine_id == 1 and cycle == 100").iloc[0]
pd.DataFrame({
    "tính tay": pd.Series(by_hand),
    "build_rolling_features": pd.Series({s: row[f"sensor_11__{s}"] for s in ROLLING_FEATURE_STATS}),
}).round(6)
""")

code("""
stored_features = pd.read_parquet("outputs/features/gold/rolling_features.parquet")
pd.testing.assert_frame_equal(features, stored_features, check_dtype=False)
n_cycles = observations.groupby(["split", "engine_id"]).size()
print("Số dòng đặc trưng:", len(features), "= tổng (số cycle - 29) =", int((n_cycles - 29).sum()))
print(features["split"].value_counts().to_dict())
""")

md("""
## 6. Những cột nào được dùng làm đặc trưng?

Mỗi sensor sinh ra sáu cột theo mẫu `<sensor>__<thống kê>`, tổng cộng
21 × 6 = 126 cột. Bảng dưới liệt kê vai trò của từng nhóm cột.
""")

code("""
roles = pd.DataFrame([
    ("split, engine_id, cycle", 3, "khóa: join, chia fold theo động cơ", "không"),
    ("setting_1..setting_3", 3, "điều kiện vận hành; FD001 gần như không đổi", "không"),
    ("sensor_1..sensor_21", 21, "tín hiệu gốc; chỉ dùng qua đặc trưng window", "không (trực tiếp)"),
    ("sensor_k__{last, mean, std, min, max, slope}", 126, "đặc trưng window 30 cycles", "có"),
    ("rul", 1, "nhãn (biến mục tiêu y)", "là y"),
], columns=["nhóm cột", "số cột", "vai trò", "trong X?"])
display(roles)

dataset = prepare_modeling_dataset(stored_features, stored_targets)   # join 1-1 theo 3 khóa
X = dataset.loc[:, list(MODELING_FEATURE_COLUMNS)]
y = dataset["rul"]
print("X:", X.shape, "| y:", y.shape)
print("Vài tên cột của X:", list(X.columns[:6]), "...")
""")

md("""
Không có bước chọn lọc đặc trưng (feature selection): mô hình nhận đủ 126 cột.
Các sensor hằng vẫn nằm trong X nhưng không mang thông tin; mô hình cây sẽ không
dùng chúng để chia nhánh.
""")

md("""
## 7. Quan hệ giữa các bảng

Kiểm tra từng quan hệ trực tiếp trên dữ liệu.
""")

code("""
per_engine = observations.groupby(["split", "engine_id"]).size()
checks = []
checks.append(("engine_summary -- observations", "1 : N",
               f"mỗi động cơ có {per_engine.min()}--{per_engine.max()} cycles"))

obs_train = observations.query("split == 'train'")[list(KEY)]
m = obs_train.merge(stored_targets, on=list(KEY), how="outer", validate="one_to_one", indicator=True)
checks.append(("observation train -- train_target", "1 : 1",
               f"{(m['_merge'] == 'both').sum()} / {len(obs_train)} dòng khớp"))

m = observations[list(KEY)].merge(stored_features[list(KEY)], on=list(KEY), how="left",
                                  validate="one_to_one", indicator=True)
checks.append(("observation -- dòng đặc trưng", "1 : 0..1",
               f"{(m['_merge'] == 'both').sum()} có đặc trưng, {(m['_merge'] == 'left_only').sum()} chưa đủ window"))

checks.append(("đặc trưng train -- train_target", "1 : 1",
               f"{len(dataset)} dòng sau join = {int(stored_features['split'].eq('train').sum())} dòng đặc trưng train"))
final_eval = pd.read_parquet("outputs/modeling/predictions/final_evaluation.parquet")
checks.append(("dự đoán test -- nhãn test", "1 : 1",
               f"{len(final_eval)} dòng, {final_eval['engine_id'].nunique()} động cơ test khác nhau"))
pd.DataFrame(checks, columns=["hai bảng", "quan hệ", "kiểm tra trên dữ liệu"])
""")

md("""
**Không có quan hệ nhiều-nhiều.** Một window dùng 30 quan sát, và hai window liên
tiếp dùng chung 29 quan sát, nhưng đó là phụ thuộc theo thời gian khi tính toán,
không phải một phép join nhiều-nhiều giữa hai bảng.

Ô cuối minh họa điều gì xảy ra nếu join sai khóa: chỉ join theo `engine_id` sẽ
ghép mọi dòng đặc trưng với mọi dòng nhãn của cùng động cơ.
""")

code("""
f1 = stored_features.query("split == 'train' and engine_id == 1")
t1 = stored_targets.query("engine_id == 1")
wrong = f1.merge(t1, on=["engine_id"])
right = f1.merge(t1, on=list(KEY), validate="one_to_one")
print(f"đặc trưng: {len(f1)} dòng, nhãn: {len(t1)} dòng")
print(f"join chỉ theo engine_id : {len(wrong):>6} dòng  (= {len(f1)} x {len(t1)}, sai)")
print(f"join theo đủ 3 khóa     : {len(right):>6} dòng  (đúng)")
""")

md("""
## Tóm tắt

| Câu hỏi | Trả lời ngắn |
|---|---|
| Gom nhiều file? | Đọc train và test, thêm `split`, nối theo dòng; file nhãn test để riêng |
| Lưu ở đâu? | Silver/Gold trong Parquet; bảng tổng hợp và nhãn thêm trong SQLite |
| Truy vấn thế nào? | SQL trên SQLite; Spark SQL hoặc Pandas trên Parquet |
| Biến đổi gì? | Đổi kiểu → nối → tạo nhãn RUL → tổng hợp theo động cơ → window 30 cycles → join |
| Cột nào vào mô hình? | Đúng 126 cột đặc trưng; khóa, settings và sensor gốc không vào X |
| Quan hệ? | Động cơ–quan sát 1:N; các join còn lại 1:1 hoặc 1:0..1; không có N:N |
""")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="execute the notebook after writing it")
    args = parser.parse_args()
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata["kernelspec"] = {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"}
    notebook.metadata["language_info"] = {"name": "python"}
    add_notebook_identity(notebook)
    if args.execute:
        from nbclient import NotebookClient

        NotebookClient(notebook, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
    nbf.write(notebook, TARGET)
    print(f"wrote {TARGET.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
