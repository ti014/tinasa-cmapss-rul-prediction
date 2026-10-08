"""Job Hadoop Streaming: số cycle, cycle đầu và cycle cuối của từng động cơ.

Chạy trên YARN dưới dạng ``python3 lifetime_job.py map`` và ``python3 lifetime_job.py
reduce``. Chỉ dùng thư viện chuẩn để chạy được trong container Hadoop. Kết quả có
cùng nội dung với bảng Gold ``engine_summary`` của ETL Pandas.

Ví dụ: dòng ``1 192 ...`` của ``train_FD001.txt`` được map thành ``train:0001<TAB>192``;
reducer gom mọi cycle của khóa ``train:0001`` và in ``train 1 192 1 192``.
"""

import os
import sys

N_FIELDS = 26


def split_from_path(path):
    name = os.path.basename(path)
    for split in ("train", "test"):
        if name.startswith(split + "_"):
            return split
    return None


def map_lines(lines, source):
    """Sinh cặp ``split:engine<TAB>cycle``; engine ID đệm 0 để Hadoop sắp đúng thứ tự."""
    split = split_from_path(source)
    for line in lines:
        fields = line.split()
        if split is None or len(fields) != N_FIELDS:
            # Dòng sai định dạng được đếm qua counter của Hadoop thay vì làm job dừng.
            sys.stderr.write("reporter:counter:cmapss,malformed_lines,1\n")
            continue
        yield "%s:%04d\t%d" % (split, int(fields[0]), int(fields[1]))


def reduce_lines(lines):
    """Gom các cycle liên tiếp cùng khóa; input đã được Hadoop sắp theo khóa."""
    current, cycles = None, []
    for line in lines:
        key, value = line.rstrip("\n").split("\t")
        if key != current and current is not None:
            yield _summary(current, cycles)
            cycles = []
        current = key
        cycles.append(int(value))
    if current is not None:
        yield _summary(current, cycles)


def _summary(key, cycles):
    split, engine = key.split(":")
    return "%s\t%d\t%d\t%d\t%d" % (split, int(engine), len(cycles), min(cycles), max(cycles))


def main(argv):
    mode = argv[1] if len(argv) > 1 else ""
    if mode == "map":
        source = os.environ.get("mapreduce_map_input_file") or os.environ.get("map_input_file", "")
        output = map_lines(sys.stdin, source)
    elif mode == "reduce":
        output = reduce_lines(sys.stdin)
    else:
        sys.stderr.write("cách dùng: lifetime_job.py map|reduce\n")
        return 2
    for line in output:
        sys.stdout.write(line + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
