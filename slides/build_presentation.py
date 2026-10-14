"""Dựng deck thuyết trình Đề tài 8 trên template PowerPoint của IUH.

Học viên: Phan Nguyễn Mai Phương · MSHV 25002711.

Deck gồm 20 slide theo thứ tự rubric: bài toán và dữ liệu → pipeline dữ liệu →
khử nhiễu và cửa sổ thời gian → xu hướng và bất thường → dự đoán RUL và cảnh
báo → mở rộng và kết luận. Mọi số liệu đọc từ artifact đã nghiệm thu; script
không huấn luyện lại mô hình.

Chạy từ thư mục gốc repository:

    uv run --frozen python slides/extract_slide_data.py
    uv run --frozen python -m cmapss.degradation
    uv run --frozen python slides/slide_figures.py
    uv run --no-project --with python-pptx==1.0.2 --with pillow==12.3.0 python slides/build_presentation.py
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "slides/template/Template Powerpoint v2 (ty le 16-9).pptx"
OUT = ROOT / "slides/cmapss_fd001_presentation.pptx"
TOTAL = 20

D = json.loads((ROOT / "slides/slide_data.json").read_text(encoding="utf-8"))
DEG = json.loads((ROOT / "outputs/degradation/summary.json").read_text(encoding="utf-8"))
with (ROOT / "outputs/degradation/tables/noise_summary.csv").open(encoding="utf-8") as file:
    NOISE = {row["sensor"]: row for row in csv.DictReader(file)}

# Màu lấy từ template IUH: xanh thương hiệu và vàng của đường viền footer.
BLUE, GOLD = "21409A", "FCB525"
INK, SLATE, LINE = "1B2733", "5A6B7B", "C9D2E3"
TINT, GOLD_TINT, WHITE, ICE = "EAF0FA", "FFF4DA", "FFFFFF", "C9D6F0"
RED, TEAL = "B23A3A", "157A7F"
FONT = "Arial"
PARTS = [
    "Bài toán và dữ liệu",
    "Pipeline dữ liệu",
    "Khử nhiễu và cửa sổ thời gian",
    "Xu hướng và bất thường",
    "Dự đoán RUL và cảnh báo",
    "Mở rộng và kết luận",
]

M, P, A, DG, I = D["modeling"], D["pipeline"], D["alignment"], D["diagnostics"], D["improvement"]
FD = {row["name"]: row for row in D["subsets"]}
WARN = {f"{r['model']}_{r['threshold']}": r for r in A["warning_policies"]}
AN, AN_RAW, ACFG = DEG["anomaly_smoothed"], DEG["anomaly_raw_for_comparison"], DEG["anomaly_config"]
MNT = json.loads((ROOT / "outputs/maintenance/summary.json").read_text(encoding="utf-8"))
HDFS = json.loads((ROOT / "outputs/hdfs/summary.json").read_text(encoding="utf-8"))
assert HDFS["mapreduce_vs_pandas"]["exact_match"] and HDFS["spark_vs_pandas"]["exact_match"]
# Đối chiếu rolling features Pandas–PySpark ở cả ba lượt benchmark.
BENCH_PARITY = json.loads((ROOT / "outputs/benchmarks/20260929/run_info.json").read_text(encoding="utf-8"))["parity"]
ROLLING_ROWS = {run["features"]["rows"] for run in BENCH_PARITY.values()}
assert len(ROLLING_ROWS) == 1 and all(run["features"]["all_cells_match"] for run in BENCH_PARITY.values())
with (ROOT / "outputs/maintenance/tables/status_vs_actual.csv").open(encoding="utf-8") as file:
    MNT_CHECK = list(csv.reader(file))
with (ROOT / "outputs/degradation/tables/anomaly_sensitivity.csv").open(encoding="utf-8") as file:
    SENS = [row for row in csv.DictReader(file) if row["signal"] == "ewma"]
SENS_LEAD_MIN = min(float(row["lead_min"]) for row in SENS)
# Sensor thay đổi cùng chiều ở mọi động cơ train (kiểm định đầu/cuối vòng đời).
with (ROOT / "outputs/statistics/tables/sensor_inference.csv").open(encoding="utf-8") as file:
    INFERENCE = list(csv.DictReader(file))
SAME_DIRECTION = [row["sensor"] for row in INFERENCE
                  if "100" in (row["n_positive"], row["n_negative"])]
assert {"sensor_4", "sensor_7", "sensor_11", "sensor_12", "sensor_20"} <= set(SAME_DIRECTION)
SENS_ALL_DETECTED = all(row["n_detected"] == row["n_engines"] for row in SENS)


def vn(x: float, d: int = 2) -> str:
    """Định dạng số kiểu Việt Nam: phẩy thập phân, chấm hàng nghìn."""
    text = f"{float(x):,.{d}f}"
    return text.replace(",", "_").replace(".", ",").replace("_", ".")


def rgb(hex_color: str) -> RGBColor:
    return RGBColor.from_string(hex_color)


# ----------------------------------------------------------------------------- primitives
def text(slide, x, y, w, h, content, *, size=14, color=INK, bold=False, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, space_after=0):
    """Thêm textbox. ``content`` là chuỗi, hoặc danh sách đoạn; mỗi đoạn là chuỗi
    hoặc danh sách run ``(chữ, {"bold":..., "color":..., "size":...})``."""
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.word_wrap = True
    frame.auto_size = None
    frame.margin_left = frame.margin_right = frame.margin_top = frame.margin_bottom = 0
    frame.vertical_anchor = anchor
    paragraphs = content if isinstance(content, list) else [content]
    for index, para in enumerate(paragraphs):
        p = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        p.alignment = align
        p.space_after = Pt(space_after)
        runs = para if isinstance(para, list) else [(para, {})]
        for chunk, style in runs:
            run = p.add_run()
            run.text = chunk
            run.font.name = FONT
            run.font.size = Pt(style.get("size", size))
            run.font.bold = style.get("bold", bold)
            run.font.color.rgb = rgb(style.get("color", color))
    return box


def rect(slide, x, y, w, h, fill, *, line=None, rounded=True):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE,
        Inches(x), Inches(y), Inches(w), Inches(h),
    )
    if rounded:
        shape.adjustments[0] = 0.08
    shape.fill.solid()
    shape.fill.fore_color.rgb = rgb(fill)
    if line:
        shape.line.color.rgb = rgb(line)
        shape.line.width = Pt(0.75)
    else:
        shape.line.fill.background()
    shape.shadow.inherit = False
    return shape


def circle(slide, x, y, d, label, fill=BLUE, color=WHITE, size=13):
    shape = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(y), Inches(d), Inches(d))
    shape.fill.solid()
    shape.fill.fore_color.rgb = rgb(fill)
    shape.line.fill.background()
    shape.shadow.inherit = False
    text(slide, x, y, d, d, label, size=size, color=color, bold=True, align=PP_ALIGN.CENTER,
         anchor=MSO_ANCHOR.MIDDLE)


def arrow(slide, x1, y1, x2, y2, color=SLATE, *, head=True, dash=False):
    connector = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    connector.line.color.rgb = rgb(color)
    connector.line.width = Pt(1.5)
    if dash:
        connector.line.dash_style = MSO_LINE_DASH_STYLE.DASH
    if head:
        line = connector.line._get_or_add_ln()
        tail = line.makeelement(qn("a:tailEnd"), {"type": "triangle"})
        line.append(tail)


def path(slide, points, color=SLATE, *, dash=False):
    """Đường gấp khúc qua các điểm; chỉ đoạn cuối có đầu mũi tên."""
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        arrow(slide, x1, y1, x2, y2, color, head=(x2, y2) == points[-1], dash=dash)


def picture(slide, rel, x, y, w, h, *, crop=None, align="center"):
    """Đặt hình vừa khung (giữ tỉ lệ); trả về (x, y, w, h) thực tế."""
    image = Image.open(ROOT / rel)
    if crop:
        image = image.crop(crop)
    iw, ih = image.size
    scale = min(w / iw, h / ih)
    pw, ph = iw * scale, ih * scale
    px = x + (w - pw) / 2 if align == "center" else x
    py = y + (h - ph) / 2 if align == "center" else y
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    slide.shapes.add_picture(buffer, Inches(px), Inches(py), Inches(pw), Inches(ph))
    return px, py, pw, ph


def table(slide, rows, x, y, col_widths, *, row_h=0.42, size=12, highlight=None, align_right_from=None):
    n_rows, n_cols = len(rows), len(rows[0])
    shape = slide.shapes.add_table(n_rows, n_cols, Inches(x), Inches(y), Inches(sum(col_widths)), Inches(row_h * n_rows))
    tbl = shape.table
    tbl.first_row = True
    for j, width in enumerate(col_widths):
        tbl.columns[j].width = Inches(width)
    for i in range(n_rows):
        tbl.rows[i].height = Inches(row_h)
        for j in range(n_cols):
            cell = tbl.cell(i, j)
            cell.margin_left = cell.margin_right = Inches(0.08)
            cell.margin_top = cell.margin_bottom = Inches(0.03)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            if i == 0:
                cell.fill.fore_color.rgb = rgb(BLUE)
            elif highlight is not None and i == highlight:
                cell.fill.fore_color.rgb = rgb(GOLD_TINT)
            else:
                cell.fill.fore_color.rgb = rgb(TINT if i % 2 == 0 else WHITE)
            frame = cell.text_frame
            frame.word_wrap = True
            p = frame.paragraphs[0]
            p.text = ""
            if align_right_from is not None and j >= align_right_from and i > 0:
                p.alignment = PP_ALIGN.RIGHT
            run = p.add_run()
            run.text = str(rows[i][j])
            run.font.name = FONT
            run.font.size = Pt(size)
            run.font.bold = i == 0 or (highlight is not None and i == highlight)
            run.font.color.rgb = rgb(WHITE if i == 0 else INK)
    return shape


def bar_chart(slide, x, y, w, h, categories, series, colors, *, title, number_format="0", max_value=None,
              horizontal=False, legend=True, decimals=0):
    """``decimals`` > 0: hiện số thập phân với dấu phẩy kiểu Việt Nam ở mọi locale.

    PowerPoint lấy dấu thập phân theo máy chiếu, nên giá trị được nhân 10^decimals và
    nhãn chèn dấu phẩy cố định trước các chữ số cuối; trục giá trị đang ẩn nên không lộ thang đo.
    """
    scale = 10 ** decimals
    if decimals:
        number_format = "0\\," + "0" * decimals
        series = [(name, [round(v * scale) for v in values]) for name, values in series]
        max_value = None if max_value is None else max_value * scale
    data = CategoryChartData()
    data.categories = categories
    for name, values in series:
        data.add_series(name, values)
    kind = XL_CHART_TYPE.BAR_CLUSTERED if horizontal else XL_CHART_TYPE.COLUMN_CLUSTERED
    chart = slide.shapes.add_chart(kind, Inches(x), Inches(y), Inches(w), Inches(h), data).chart
    chart.font.name = FONT
    chart.font.size = Pt(12)
    chart.font.color.rgb = rgb(INK)
    chart.has_title = True
    chart.chart_title.text_frame.text = title
    title_run = chart.chart_title.text_frame.paragraphs[0].runs[0]
    title_run.font.size, title_run.font.bold, title_run.font.name = Pt(13), True, FONT
    title_run.font.color.rgb = rgb(INK)
    chart.has_legend = legend
    if legend:
        chart.legend.position = XL_LEGEND_POSITION.BOTTOM
        chart.legend.include_in_layout = False
        chart.legend.font.size = Pt(11)
    plot = chart.plots[0]
    plot.gap_width = 70
    plot.has_data_labels = True
    labels = plot.data_labels
    labels.number_format = number_format
    labels.number_format_is_linked = False
    labels.position = XL_LABEL_POSITION.OUTSIDE_END
    labels.font.size = Pt(12)
    for item, color in zip(plot.series, colors):
        item.format.fill.solid()
        item.format.fill.fore_color.rgb = rgb(color)
    value_axis = chart.value_axis
    value_axis.visible = False
    value_axis.has_major_gridlines = False
    value_axis.minimum_scale = 0
    if max_value is not None:
        value_axis.maximum_scale = max_value
    chart.category_axis.tick_labels.font.size = Pt(12)
    chart.category_axis.format.line.color.rgb = rgb(LINE)
    if horizontal:
        chart.category_axis.reverse_order = True
    return chart


def stat(slide, x, y, w, value, label, *, color=BLUE, size=28):
    text(slide, x, y, w, 0.55, value, size=size, bold=True, color=color, anchor=MSO_ANCHOR.BOTTOM)
    text(slide, x, y + 0.6, w, 0.5, label, size=12, color=SLATE)


# ----------------------------------------------------------------------------- slide frames
PRS = Presentation(str(TEMPLATE))
# Xóa 10 slide mẫu, giữ nguyên master và layout của template.
for slide_id in list(PRS.slides._sldIdLst):
    PRS.part.drop_rel(slide_id.rId)
    PRS.slides._sldIdLst.remove(slide_id)
LAYOUT = {layout.name: layout for layout in PRS.slide_layouts}


def _remove(placeholder) -> None:
    element = placeholder._element
    element.getparent().remove(element)


def content(part: int, n: int, title: str, takeaway: str, source: str, notes: str):
    """Slide nội dung trên layout Title and Content của template."""
    slide = PRS.slides.add_slide(LAYOUT["Title and Content"])
    for placeholder in list(slide.placeholders):
        if placeholder.placeholder_format.idx != 0:
            _remove(placeholder)
    heading = slide.shapes.title
    heading.left, heading.top, heading.width, heading.height = Inches(0.92), Inches(1.3), Inches(11.5), Inches(0.62)
    heading.text_frame.word_wrap = True
    heading.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    heading.text = title
    run = heading.text_frame.paragraphs[0].runs[0]
    run.font.size, run.font.bold, run.font.name = Pt(26), True, FONT
    run.font.color.rgb = rgb(BLUE)
    text(slide, 6.2, 0.42, 6.2, 0.38, f"PHẦN {part} · {PARTS[part - 1].upper()}", size=12, bold=True,
         color=WHITE, align=PP_ALIGN.RIGHT, anchor=MSO_ANCHOR.MIDDLE)
    text(slide, 0.92, 1.93, 11.5, 0.42, takeaway, size=15, color=SLATE, anchor=MSO_ANCHOR.MIDDLE)
    text(slide, 0.45, 6.98, 6.6, 0.36, f"Nguồn: {source}", size=8.5, color=WHITE, anchor=MSO_ANCHOR.MIDDLE)
    text(slide, 11.2, 6.98, 1.22, 0.32, f"{n} / {TOTAL}", size=11, color=SLATE, align=PP_ALIGN.RIGHT,
         anchor=MSO_ANCHOR.MIDDLE)
    slide.notes_slide.notes_text_frame.text = notes
    return slide


# ============================================================================ 1. Bìa
def slide_cover():
    slide = PRS.slides.add_slide(LAYOUT["Title Slide"])
    for placeholder in list(slide.placeholders):
        if placeholder.placeholder_format.idx not in (0, 1):
            _remove(placeholder)
    title, subtitle = slide.placeholders[0], slide.placeholders[1]
    title.left, title.top, title.width, title.height = Inches(0.92), Inches(1.75), Inches(11.5), Inches(1.5)
    title.text = "Phân tích dữ liệu IoT và bảo trì dự đoán"
    p = title.text_frame.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    p.runs[0].font.size, p.runs[0].font.bold, p.runs[0].font.name = Pt(40), True, FONT
    p.runs[0].font.color.rgb = rgb(BLUE)
    subtitle.left, subtitle.top, subtitle.width, subtitle.height = Inches(1.4), Inches(3.25), Inches(10.5), Inches(1.0)
    # "\v" là ngắt dòng trong cùng đoạn, tránh để PowerPoint ngắt giữa chữ "C-MAPSS".
    subtitle.text = "Phát hiện bất thường, dự đoán tuổi thọ còn lại (RUL) và cảnh báo sớm\vtrên dữ liệu cảm biến NASA C-MAPSS"
    p = subtitle.text_frame.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    for run in p.runs:
        run.font.size, run.font.name = Pt(18), FONT
        run.font.color.rgb = rgb(INK)
    text(slide, 0.92, 1.3, 11.5, 0.4, "BÁO CÁO CUỐI KỲ · PHÂN TÍCH DỮ LIỆU LỚN · ĐỀ TÀI 8", size=14, bold=True,
         color=GOLD_DARK, align=PP_ALIGN.CENTER)
    items = [
        (str(FD["FD001"]["train_engines"]), "động cơ train FD001"),
        (f"{vn(DEG['noise_reduction_pct_median'], 0)}%", "dao động giảm nhờ EWMA"),
        (vn(AN["lead_median"], 0), "cycle báo trước khi hỏng (trung vị)"),
        (vn(M["final"]["mean_engine_mae"]), "sai số MAE trên tập test (cycle)"),
    ]
    for i, (value, label) in enumerate(items):
        x = 1.2 + i * 2.8
        rect(slide, x, 4.45, 2.55, 1.15, WHITE, line=LINE)
        text(slide, x, 4.52, 2.55, 0.6, value, size=28, bold=True, color=BLUE, align=PP_ALIGN.CENTER)
        text(slide, x, 5.1, 2.55, 0.4, label, size=12, color=SLATE, align=PP_ALIGN.CENTER)
    text(slide, 0.92, 5.95, 11.5, 0.4, "Phan Nguyễn Mai Phương · MSHV 25002711", size=18, bold=True, color=BLUE,
         align=PP_ALIGN.CENTER)
    text(slide, 0.92, 6.35, 11.5, 0.35, "Học viên cao học ngành Khoa học máy tính · Tháng 10/2026", size=13,
         color=INK, align=PP_ALIGN.CENTER)
    slide.notes_slide.notes_text_frame.text = (
        "Thời lượng: 30 giây.\n"
        "Kính chào quý thầy cô. Em là Phan Nguyễn Mai Phương, MSHV 25002711. Đề tài 8: phân tích dữ liệu cảm biến để "
        "phát hiện dấu hiệu bất thường và dự đoán thời điểm cần bảo trì, kèm biểu đồ cảnh báo sớm.\n"
        f"Bốn con số là khung của bài: 100 động cơ train; làm mượt EWMA giảm khoảng {vn(DEG['noise_reduction_pct_median'], 0)}% "
        f"dao động ngắn hạn; bất thường được báo trước khi hỏng trung vị {vn(AN['lead_median'], 0)} cycle; sai số tuyệt "
        f"đối trung bình (MAE) của RUL trên 100 động cơ test là {vn(M['final']['mean_engine_mae'])} cycle."
    )


GOLD_DARK = "B37800"


# ============================================================================ 2. Lộ trình
def slide_agenda():
    s = content(1, 2, "Cấu trúc bài trình bày",
                "Đi theo thứ tự pipeline: chuẩn bị dữ liệu → phân tích tín hiệu → dự đoán → cảnh báo",
                "report/main.pdf; handoff/SPEECH_SCRIPT_VI.pdf",
                "Thời lượng: 25 giây.\nĐọc nhanh sáu phần nội dung và phần demo. Thứ tự bám quy trình của môn học: thu "
                "thập, làm sạch, lưu trữ trước; rồi làm mượt tín hiệu, xu hướng và bất thường; sau cùng là dự đoán RUL và "
                "cảnh báo. Năm câu hỏi bên phải sẽ được trả lời lại ở slide 18. Demo dashboard khoảng 2,5 phút, ngay sau "
                "slide 19.")
    rows = [
        ("Bài toán và dữ liệu", "Slide 3–5"), ("Pipeline dữ liệu", "Slide 6–8"),
        ("Khử nhiễu và cửa sổ thời gian", "Slide 9–10"), ("Xu hướng và bất thường", "Slide 11–12"),
        ("Dự đoán RUL, cảnh báo, lịch bảo trì", "Slide 13–16"), ("Mở rộng và kết luận", "Slide 17–19"),
        ("Demo dashboard", "Khoảng 2,5 phút"),
    ]
    for i, (name, where) in enumerate(rows):
        y = 2.5 + i * 0.58
        circle(s, 0.92, y, 0.44, str(i + 1), fill=GOLD if i == 6 else BLUE, color=INK if i == 6 else WHITE, size=12)
        text(s, 1.55, y - 0.02, 4.8, 0.28, name, size=15, bold=True)
        text(s, 1.55, y + 0.26, 4.8, 0.24, where, size=11, color=SLATE)
    rect(s, 6.75, 2.5, 5.67, 4.1, TINT)
    text(s, 7.05, 2.68, 5.1, 0.4, "Câu hỏi nghiên cứu", size=18, bold=True, color=BLUE)
    questions = [
        "Dữ liệu có đạt các kiểm tra chất lượng không; Pandas và PySpark có cho cùng kết quả không?",
        "Làm mượt giảm được bao nhiêu nhiễu; dữ liệu được tổng hợp theo cửa sổ thời gian ra sao?",
        "Tín hiệu thay đổi thế nào theo tuổi máy; có phát hiện được bất thường trước khi hỏng không?",
        "Dự đoán RUL chính xác đến đâu trên tập test độc lập?",
        "Cảnh báo sớm và lịch bảo trì hiệu quả đến đâu?",
    ]
    for i, question in enumerate(questions):
        y = 3.22 + i * 0.66
        circle(s, 7.05, y, 0.42, f"Q{i + 1}", fill=GOLD, color=INK, size=10.5)
        text(s, 7.65, y - 0.04, 4.6, 0.55, question, size=12.5, anchor=MSO_ANCHOR.MIDDLE)


# ============================================================================ 3. Bài toán
def slide_problem():
    s = content(1, 3, "Phát biểu bài toán",
                "Phát hiện dấu hiệu suy giảm và dự đoán thời điểm bảo trì từ dữ liệu cảm biến; đối chiếu yêu cầu Đề tài 8",
                "references/context/assignment-brief.png; report/sections/problem-data.tex",
                "Thời lượng: 45 giây.\nRemaining Useful Life (RUL, tuổi thọ còn lại) là số cycle vận hành còn lại "
                "trước khi hỏng. Ví dụ động cơ hỏng ở cycle 200 thì tại cycle 170 có RUL bằng 30. Đơn vị là cycle; dữ liệu "
                "không quy đổi ra giờ.\nBảng bên phải đối chiếu từng yêu cầu của đề tài với slide trình bày: chuỗi thời gian "
                "ở slide 6, 9, 10; làm mượt ở slide 9; cửa sổ thời gian ở slide 10; xu hướng ở slide 11; bất thường ở slide "
                "12; cảnh báo ở slide 15 và lịch bảo trì ở slide 16.")
    text(s, 0.92, 2.5, 5.3, 0.32, "Định nghĩa RUL", size=15, bold=True, color=BLUE)
    x0, x1, y = 1.0, 5.6, 3.4
    tx = x0 + (x1 - x0) * 170 / 200
    rect(s, x0, y, x1 - x0, 0.2, ICE, rounded=False)
    rect(s, tx, y, x1 - tx, 0.2, GOLD, rounded=False)
    dot = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x1 - 0.12), Inches(y - 0.06), Inches(0.32), Inches(0.32))
    dot.fill.solid(); dot.fill.fore_color.rgb = rgb(RED); dot.line.fill.background()
    text(s, x0 - 0.1, y + 0.32, 1.0, 0.28, "cycle 1", size=11, color=SLATE)
    text(s, tx - 0.5, y - 0.42, 1.0, 0.3, "t = 170", size=12, bold=True, color=GOLD_DARK, align=PP_ALIGN.CENTER)
    text(s, x1 - 0.6, y + 0.32, 1.2, 0.5, "hỏng\ncycle 200", size=11, color=RED, align=PP_ALIGN.CENTER)
    text(s, 0.92, 4.35, 5.4, 0.45, "RUL(t) = T − t", size=22, bold=True, color=BLUE)
    text(s, 0.92, 4.85, 5.4, 0.4, "RUL(170) = 200 − 170 = 30 cycle", size=16, bold=True, color=GOLD_DARK)
    text(s, 0.92, 5.4, 5.3, 1.1, "T: cycle động cơ hỏng. Đầu vào: chuỗi 21 tín hiệu cảm biến theo cycle. Đầu ra: dấu "
         "hiệu bất thường, RUL dự đoán và mức cảnh báo.", size=14)
    table(s, [
        ["Yêu cầu của đề tài", "Cách thực hiện", "Slide"],
        ["Xử lý chuỗi thời gian cảm biến", "Kiểm tra khóa, cycle liên tục; sắp theo động cơ và cycle", "6, 9, 10"],
        ["Xử lý nhiễu tín hiệu", "Làm mượt EWMA span 10 theo từng động cơ, chỉ dùng quá khứ", "9"],
        ["Chuẩn hóa, tổng hợp theo cửa sổ", "StandardScaler; cửa sổ 30 cycle → 126 đặc trưng", "10"],
        ["Phát hiện dấu hiệu bất thường", "Health Index (chỉ số sức khỏe) giảm quá 3σ so với 30 cycle đầu", "12"],
        ["Trực quan hóa xu hướng suy giảm", "Quỹ đạo sensor, kiểm định đầu/cuối, Health Index", "11, 12"],
        ["Dự đoán thời điểm bảo trì, cảnh báo sớm", "Lịch bảo trì cho 100 động cơ test; hai mức cảnh báo", "15, 16"],
    ], 6.55, 2.5, [2.15, 2.95, 0.77], row_h=0.58, size=11.5)


# ============================================================================ 4. Dữ liệu
def slide_data():
    s = content(1, 4, "Bộ dữ liệu NASA C-MAPSS",
                "Chọn NASA Turbofan (C-MAPSS) trong gợi ý của đề vì có chuỗi cảm biến chạy đến khi hỏng của từng động cơ",
                "data/raw/readme.txt; outputs/cross_dataset/FD00x/etl/run_info.json",
                "Thời lượng: 50 giây.\nDữ liệu là bộ NASA C-MAPSS, tải trực tiếp từ cổng dữ liệu mở của NASA. Mỗi động cơ "
                "có chuỗi cảm biến theo cycle cho tới khi hỏng, và tập test có nhãn RUL thật để chấm điểm.\n"
                "C-MAPSS có bốn tập con (subset). Mỗi subset có train (chạy đến khi hỏng), test (bị cắt trước khi hỏng) và "
                "file nhãn RUL test. FD001: 1 điều kiện vận hành, 1 dạng lỗi ở máy nén áp suất cao (HPC), 100 động cơ "
                "train và 100 động cơ test, 33.727 quan sát. Mỗi dòng có 26 trường: engine_id, cycle, 3 setting và 21 sensor.")
    meta = {"FD001": ("1 điều kiện", "1 dạng lỗi (HPC)"), "FD002": ("6 điều kiện", "1 dạng lỗi (HPC)"),
            "FD003": ("1 điều kiện", "2 dạng lỗi (HPC, Fan)"), "FD004": ("6 điều kiện", "2 dạng lỗi (HPC, Fan)")}
    for i, row in enumerate(D["subsets"]):
        x, main = 0.92 + i * 2.9, i == 0
        rect(s, x, 2.5, 2.72, 2.45, BLUE if main else TINT)
        fg, sub = (WHITE, ICE) if main else (INK, SLATE)
        text(s, x + 0.22, 2.62, 2.3, 0.5, row["name"], size=24, bold=True, color=WHITE if main else BLUE)
        text(s, x + 0.22, 3.12, 2.3, 0.3, "Phân tích chính" if main else "Thử nghiệm mở rộng", size=11.5, bold=True,
             color=GOLD if main else TEAL)
        text(s, x + 0.22, 3.5, 2.4, 0.62, f"{meta[row['name']][0]} vận hành\n{meta[row['name']][1]}", size=13, color=fg)
        text(s, x + 0.22, 4.2, 2.4, 0.62,
             f"{row['train_engines']} train / {row['test_engines']} test\n{vn(row['observations'], 0)} quan sát", size=13,
             color=sub)
    text(s, 0.92, 5.2, 8, 0.34, "Cấu trúc một bản ghi quan sát (26 trường)", size=15, bold=True, color=BLUE)
    cx = 0.92
    for label, width, color in (("engine_id", 1.35, BLUE), ("cycle", 1.0, BLUE), ("setting_1 … setting_3", 2.55, TEAL),
                                ("sensor_1 … sensor_21", 6.36, GOLD)):
        rect(s, cx, 5.6, width, 0.48, color)
        text(s, cx, 5.6, width, 0.48, label, size=13, bold=True, color=INK if color == GOLD else WHITE,
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        cx += width + 0.08
    text(s, 0.92, 6.2, 11.5, 0.35, "Khóa chính gồm split (train hoặc test, thêm khi ETL), engine_id và cycle. "
         "Không có giá trị thiếu. HPC: máy nén áp suất cao.", size=13, color=SLATE)


# ============================================================================ 5. Nhãn
def slide_labels():
    s = content(1, 5, "Xây dựng nhãn RUL và ranh giới tập test",
                "Nhãn tập test chỉ được dùng sau khi đã lưu kết quả dự đoán, để tránh rò rỉ dữ liệu (data leakage)",
                "data/CONTRACTS.md; outputs/modeling/run_info.json; outputs/modeling/tables/final_metrics.json",
                "Thời lượng: 50 giây.\nTrain: mỗi động cơ chạy đến khi hỏng nên RUL = cycle cuối − cycle hiện tại; FD001 có "
                "20.631 nhãn train.\nTest: chuỗi bị cắt trước khi hỏng. Mô hình tạo đúng một dự đoán tại cycle cuối quan sát "
                "của mỗi động cơ test; sau khi ghi dự đoán ra file mới ghép RUL_FD001.txt để chấm điểm.")

    def lane(y, label, sub, full):
        text(s, 0.92, y, 1.6, 0.34, label, size=17, bold=True, color=BLUE)
        text(s, 0.92, y + 0.36, 1.75, 0.6, sub, size=11.5, color=SLATE)
        x0, x1 = 2.75, 8.0 if full else 6.2
        rect(s, x0, y + 0.12, x1 - x0, 0.26, ICE, rounded=False)
        if full:
            dot = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x1 - 0.15), Inches(y + 0.07), Inches(0.36), Inches(0.36))
            dot.fill.solid(); dot.fill.fore_color.rgb = rgb(RED); dot.line.fill.background()
            text(s, x1 - 0.5, y + 0.5, 1.0, 0.28, "hỏng", size=11.5, color=RED, align=PP_ALIGN.CENTER)
        else:
            dash = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y + 0.25), Inches(x1 + 1.8),
                                          Inches(y + 0.25))
            dash.line.color.rgb = rgb(SLATE); dash.line.dash_style = 4
            dot = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x1 - 0.13), Inches(y + 0.1), Inches(0.3), Inches(0.3))
            dot.fill.solid(); dot.fill.fore_color.rgb = rgb(GOLD); dot.line.fill.background()
            text(s, x1 - 1.6, y + 0.5, 1.8, 0.28, "cycle cuối quan sát", size=11.5, color=GOLD_DARK, align=PP_ALIGN.RIGHT)
            text(s, x1 + 0.3, y + 0.5, 1.6, 0.28, "chưa biết lúc hỏng", size=11.5, color=SLATE, align=PP_ALIGN.RIGHT)

    lane(2.55, "Train", f"{FD['FD001']['train_engines']} động cơ chạy đến khi hỏng", True)
    lane(3.85, "Test", f"Chuỗi của {FD['FD001']['test_engines']} động cơ bị cắt trước khi hỏng", False)
    rect(s, 0.92, 5.1, 7.4, 1.45, TINT)
    text(s, 1.15, 5.25, 7.0, 1.2, [
        [("Train: ", {"bold": True, "color": BLUE}), (f"RUL(t) = cycle cuối − t; tổng cộng {vn(P['parity'][1]['rows'], 0)} nhãn", {})],
        [("Test: ", {"bold": True, "color": BLUE}),
         ("một dự đoán tại cycle cuối của mỗi động cơ; ghép RUL_FD001.txt sau khi đã lưu dự đoán", {})],
    ], size=14.5, space_after=8)
    rect(s, 8.7, 2.5, 3.72, 4.05, BLUE)
    text(s, 8.95, 2.68, 3.3, 0.35, "Kiểm soát nhãn tập test", size=16, bold=True, color=WHITE)
    for i, (head, body) in enumerate((("Không dùng sớm", "nhãn test không dùng trong EDA, tạo đặc trưng hay chọn mô hình"),
                                      ("Ghép sau dự đoán", "nhãn chỉ được ghép khi dự đoán đã được lưu"),
                                      ("Truy vết được", "metadata ghi trạng thái ghép nhãn và mã băm SHA-256"))):
        yy = 3.2 + i * 1.1
        text(s, 8.95, yy, 3.3, 0.4, head, size=19, bold=True, color=GOLD)
        text(s, 8.95, yy + 0.42, 3.3, 0.6, body, size=12.5, color=ICE)


# ============================================================================ 6. Pipeline
def slide_pipeline():
    s = content(2, 6, "Kiến trúc pipeline dữ liệu Bronze–Silver–Gold",
                "Ba bước chuẩn bị chung, rồi tách hai nhánh: dự đoán RUL và theo dõi tín hiệu",
                "report/figures/pipeline.pdf; handoff/DATA_WALKTHROUGH.md; outputs/*/run_info.json",
                "Thời lượng: 55 giây.\nBước 1–3 là phần chuẩn bị chung. Bước 1 tải dữ liệu từ cổng NASA và kiểm tra "
                "MD5, SHA-256. Bước 2 giữ nguyên file text gốc ở lớp Bronze. Bước 3 kiểm tra đủ 26 cột, đúng kiểu, "
                "không trùng khóa, cycle liên tục, rồi tính RUL cho train. Ô nét đứt là bước chạy lại trên Hadoop: "
                f"MapReduce tính lại bảng tổng hợp động cơ, khớp {HDFS['mapreduce_vs_pandas']['rows_candidate']}/"
                f"{HDFS['mapreduce_vs_pandas']['rows_reference']} động cơ, không đi vào mô hình.\n"
                "Nhánh A dự đoán RUL: lưu bảng Silver và Gold dạng Parquet, SQLite; tạo "
                f"{P['feature_columns']} đặc trưng từ cửa sổ 30 cycle; huấn luyện HistGradientBoosting (HGB), chia 5 phần theo động cơ; "
                "lưu 100 dự đoán test rồi mới mở đáp án để chấm điểm. Pandas và PySpark cho cùng kết quả ở A1–A2.\n"
                "Nhánh B chỉ học trên train: khám phá dữ liệu, làm mượt bằng EWMA, gộp sensor thành chỉ số sức khỏe "
                "bằng PCA và phát hiện bất thường. Lịch bảo trì ghép RUL dự đoán với trạng thái bất thường của "
                "động cơ test.\nHàng dưới cùng gắn năm bước của quy trình dữ liệu với sơ đồ: thu thập là bước 1; làm sạch "
                "là bước 3 và làm mượt B2; chuẩn hóa là thống nhất kiểu dữ liệu và StandardScaler ở slide 10; tích hợp là "
                "gộp train với test thành 33.727 dòng và ghép nhãn theo khóa; lưu trữ là Parquet, SQLite và HDFS.")

    def node(x, y, w, h, title, detail, kind="step"):
        fill, line = {"step": (TINT, None), "result": (BLUE, None), "side": (WHITE, SLATE)}[kind]
        shape = rect(s, x, y, w, h, fill, line=line)
        if kind == "side":
            shape.line.dash_style = MSO_LINE_DASH_STYLE.DASH
            shape.line.width = Pt(1.25)
        head_color = WHITE if kind == "result" else SLATE if kind == "side" else BLUE
        body_color = WHITE if kind == "result" else INK
        text(s, x + 0.1, y + 0.08, w - 0.2, 0.3, title, size=12.5, bold=True, color=head_color)
        text(s, x + 0.1, y + 0.4, w - 0.2, h - 0.45, detail, size=10.5, color=body_color)

    # Cột trái: bước 1–3 và ô Hadoop chỉ để đối chiếu.
    lx, lw, lh = 0.92, 2.35, 0.8
    ly = (2.45, 3.49, 4.525)
    node(lx, ly[0], lw, lh, "1. Tải dữ liệu từ NASA", "Kiểm tra MD5, SHA-256")
    node(lx, ly[1], lw, lh, "2. Dữ liệu gốc (Bronze)", "File text, giữ nguyên")
    node(lx, ly[2], lw, lh, "3. Đọc và kiểm tra", "26 cột, khóa, cycle liên tục;\ntính RUL cho train")
    node(lx, 5.6, lw, lh, "Chạy lại trên Hadoop",
         f"MapReduce khớp\n{HDFS['mapreduce_vs_pandas']['rows_candidate']}/"
         f"{HDFS['mapreduce_vs_pandas']['rows_reference']} động cơ", "side")
    cx = lx + lw / 2
    arrow(s, cx, ly[0] + lh, cx, ly[1])
    arrow(s, cx, ly[1] + lh, cx, ly[2])
    path(s, [(lx, ly[1] + lh / 2), (lx - 0.17, ly[1] + lh / 2), (lx - 0.17, 6.0), (lx, 6.0)], dash=True)

    # Hai nhánh: năm cột cùng bề rộng.
    x0, bw, step, bh = 3.6, 1.62, 1.79, 1.05
    col = [x0 + i * step for i in range(5)]
    ya, yb = 2.8, 4.4
    text(s, x0, 2.45, 6.0, 0.3, "A. Dự đoán RUL", size=12.5, bold=True, color=BLUE)
    text(s, x0, 4.05, 6.0, 0.3, "B. Theo dõi tín hiệu (chỉ học trên train)", size=12.5, bold=True, color=BLUE)
    lane_a = [("A1. Lưu bảng", "Gộp train + test;\nSilver, Gold"),
              ("A2. Đặc trưng", f"Cửa sổ 30 cycle\n{P['feature_columns']} đặc trưng"),
              ("A3. Huấn luyện", "HGB, chia 5 phần\ntheo động cơ"),
              ("A4. Dự đoán test", "100 động cơ, lưu\ntrước khi mở đáp án"),
              ("A5. Chấm điểm", "MAE, RMSE,\nNASA score")]
    lane_b = [("B1. Khám phá", "EDA: sensor đổi\ntheo tuổi máy"),
              ("B2. Làm mượt", "EWMA, 14 sensor,\nspan 10"),
              ("B3. Health Index", "Gộp 21 sensor\nbằng PCA"),
              ("B4. Bất thường", "Giảm quá 3σ, 5 cycle\n→ cảnh báo mức 1")]
    for i, (title, detail) in enumerate(lane_a):
        node(col[i], ya, bw, bh, title, detail, "result" if i == 3 else "step")
        if i:
            arrow(s, col[i - 1] + bw + 0.02, ya + bh / 2, col[i] - 0.02, ya + bh / 2)
    for i, (title, detail) in enumerate(lane_b):
        node(col[i], yb, bw, bh, title, detail)
        if i:
            arrow(s, col[i - 1] + bw + 0.02, yb + bh / 2, col[i] - 0.02, yb + bh / 2)
    node(col[4], yb, bw, bh, "Lịch bảo trì", "RUL ≤ 30: bảo trì\nBất thường: theo dõi", "result")
    arrow(s, col[3] + bw + 0.02, yb + bh / 2, col[4] - 0.02, yb + bh / 2)

    # Bước 3 tỏa sang hai nhánh; A4 đi xuống lịch bảo trì.
    mid3 = ly[2] + lh / 2
    path(s, [(lx + lw, mid3), (x0 - 0.17, mid3), (x0 - 0.17, ya + bh / 2), (x0 - 0.02, ya + bh / 2)])
    arrow(s, x0 - 0.17, mid3, x0 - 0.02, yb + bh / 2)
    a4x = col[3] + bw / 2
    path(s, [(a4x, ya + bh), (a4x, 4.2), (col[4] + 0.4, 4.2), (col[4] + 0.4, yb - 0.02)])

    # Năm bước của quy trình quản lý dữ liệu, gắn với các ô trên sơ đồ.
    stages = [("Thu thập", "bước 1"), ("Làm sạch", "bước 3; B2 làm mượt"), ("Chuẩn hóa", "kiểu dữ liệu; slide 10"),
              ("Tích hợp", "gộp train + test"), ("Lưu trữ", "Parquet, SQLite, HDFS")]
    cw, cg = 1.676, 0.1
    for i, (head, body) in enumerate(stages):
        x = x0 + i * (cw + cg)
        rect(s, x, 5.62, cw, 0.6, GOLD_TINT)
        text(s, x + 0.08, 5.65, cw - 0.16, 0.26, head, size=11.5, bold=True, color=GOLD_DARK)
        text(s, x + 0.08, 5.9, cw - 0.16, 0.28, body, size=10, color=INK)
    text(s, x0, 6.28, 8.8, 0.28, "Ô nét đứt là kiểm tra phụ, không đi vào mô hình. Pandas và PySpark cho cùng kết quả ở A1–A2.",
         size=11, color=SLATE)


# ============================================================================ 7. Parity và benchmark
def slide_parity():
    bench = {(r["engine"], int(r["scale"])): r for r in D["stage_benchmark"]}
    pandas10, spark10 = bench[("pandas", 10)]["pipeline_seconds"], bench[("spark", 10)]["pipeline_seconds"]
    s = content(2, 7, "Đối chiếu Pandas, PySpark và Hadoop",
                f"Pandas, PySpark và Hadoop cho cùng kết quả; với dữ liệu gấp 10 lần, PySpark nhanh hơn Pandas khoảng {vn(pandas10 / spark10, 1)} lần",
                "outputs/etl/parity.json; outputs/hdfs/summary.json; outputs/benchmarks/20260929/measurements.csv",
                "Thời lượng: 55 giây.\nBảng trái: cùng đầu vào, Pandas và PySpark cho cùng schema, khóa, số dòng và giá trị ở "
                "ba bảng ETL và bảng đặc trưng cửa sổ; sai khác bằng 0. Hai dòng cuối là bảng engine_summary tính lại trên cụm Hadoop: một job "
                "MapReduce chạy trên YARN và một job Spark đọc dữ liệu từ HDFS; cả hai khớp 200/200 động cơ.\nÔ dưới: cụm "
                f"{HDFS['hadoop_version']} một node trong Docker; HDFS ở trạng thái HEALTHY.\nBiểu đồ phải: trung vị ba lượt. "
                f"Pandas tăng từ {vn(bench[('pandas', 1)]['pipeline_seconds'], 1)} lên {vn(pandas10, 1)} giây; Spark từ "
                f"{vn(bench[('spark', 1)]['pipeline_seconds'], 1)} lên {vn(spark10, 1)} giây.\nGiới hạn: một máy, chưa "
                "phải cluster nhiều node.")
    names = {"silver_observations": "Silver observations", "silver_train_targets": "Silver train_targets",
             "gold_engine_summary": "Gold engine_summary"}
    hdfs_rows = [
        ["Gold · MapReduce trên HDFS", vn(HDFS["mapreduce_vs_pandas"]["rows_candidate"], 0),
         vn(HDFS["mapreduce_vs_pandas"]["mismatched_cells"], 0), "Khớp"],
        ["Gold · Spark trên HDFS", vn(HDFS["spark_vs_pandas"]["rows_candidate"], 0),
         vn(HDFS["spark_vs_pandas"]["mismatched_cells"], 0), "Khớp"],
    ]
    rolling_row = [["Gold rolling_features", vn(ROLLING_ROWS.pop(), 0), "0", "Khớp"]]
    table(s, [["Bảng (so với Pandas)", "Số dòng", "Sai khác", "Kết quả"]] + [
        [names.get(p["name"], p["name"]), vn(p["rows"], 0), vn(p["max_abs_diff"], 0), "Khớp" if p["status"] == "passed" else p["status"]]
        for p in P["parity"]
    ] + rolling_row + hdfs_rows, 0.92, 2.45, [2.75, 1.05, 1.3, 1.0], row_h=0.37, size=11.5)
    rect(s, 0.92, 5.1, 6.1, 1.45, TINT)
    text(s, 1.15, 5.2, 5.7, 0.34, f"Cụm {HDFS['hadoop_version']} một node trong Docker", size=14, bold=True, color=BLUE)
    text(s, 1.15, 5.58, 5.7, 0.95, f"Raw lên HDFS; MapReduce trên YARN: 3 map, 1 reduce; Spark ghi Silver Parquet "
         f"{vn(HDFS['spark_silver_rows'], 0)} dòng lên HDFS; fsck: HEALTHY.", size=12.5)
    bar_chart(s, 7.3, 2.45, 5.12, 3.6, ["1× · 33.727 dòng", "10× · 337.270 dòng"],
              [("Pandas", [bench[("pandas", k)]["pipeline_seconds"] for k in (1, 10)]),
               ("PySpark local[4]", [bench[("spark", k)]["pipeline_seconds"] for k in (1, 10)])],
              [BLUE, GOLD], title="Thời gian pipeline gồm đọc, ghi (giây)", decimals=1)
    text(s, 7.3, 6.12, 5.12, 0.45, "Trung vị ba lượt; không gồm thời gian khởi động Spark; đo trên một máy, chưa phải cluster.",
         size=11, color=SLATE)


# ============================================================================ 8. EDA
def slide_eda():
    E = D["eda"]
    life = E["lifetime"]
    s = content(2, 8, "Phân tích khám phá dữ liệu",
                "Tuổi thọ động cơ từ 128 đến 362 cycle; bảy cột không đổi trên toàn bộ tập train",
                "data/raw/train_FD001.txt; outputs/eda/tables/constant_engine_counts.csv; slides/slide_figures.py",
                "Thời lượng: 25 giây.\nTuổi thọ train từ 128 đến 362 cycle, trung vị 199, lệch phải vì có vài động cơ chạy "
                "rất lâu.\nBảy cột không đổi trong cả 100 động cơ: sensor 1, 5, 10, 16, 18, 19 và setting_3. Sensor_6 không "
                "đổi ở 38/100 động cơ nên cũng bị bỏ. Vì vậy bước làm mượt chỉ áp dụng cho 14 sensor có xu hướng. Không "
                "dùng EDA để chọn đặc trưng (feature selection), để tránh rò rỉ vào kiểm định chéo.")
    picture(s, "slides/figures/engine_lifetimes.png", 0.92, 2.5, 7.4, 2.55, align="left")
    for i, (label, value) in enumerate((("Ngắn nhất", life["min"]), ("Trung vị", life["50%"]), ("Trung bình", life["mean"]),
                                        ("Dài nhất", life["max"]))):
        stat(s, 0.92 + i * 1.85, 5.25, 1.8, vn(value, 2 if value % 1 else 0), f"{label} (cycle)", size=26)
    rect(s, 8.7, 2.5, 3.72, 4.05, TINT)
    text(s, 8.92, 2.66, 3.3, 0.35, "Cột không đổi ở 100/100 động cơ", size=14.5, bold=True, color=BLUE)
    for i, name in enumerate(E["always_constant"]):
        x, y = 8.92 + (i % 2) * 1.7, 3.15 + (i // 2) * 0.52
        rect(s, x, y, 1.6, 0.42, TEAL if name.startswith("setting") else SLATE)
        text(s, x, y, 1.6, 0.42, name, size=12, bold=True, color=WHITE, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    partial = E["partially_constant"][0]
    text(s, 8.92, 5.3, 3.3, 0.4, f"{partial['measurement']} không đổi ở {partial['constant_engines']}/100 động cơ.",
         size=12.5)
    text(s, 8.92, 5.75, 3.3, 0.7, "Làm mượt áp dụng cho 14 sensor có xu hướng: bỏ 6 sensor trên và sensor_6.",
         size=12, color=SLATE)


# ============================================================================ 9. Khử nhiễu
def slide_denoise():
    reductions = [float(r["reduction_pct"]) for r in NOISE.values()]
    s4 = NOISE["sensor_4"]
    s = content(3, 9, "Khử nhiễu tín hiệu bằng EWMA",
                f"Độ dao động ngắn hạn giảm khoảng {vn(DEG['noise_reduction_pct_median'], 0)}% (trung vị trên 14 sensor); xu hướng suy giảm được bảo toàn",
                "outputs/degradation/tables/signals.parquet; outputs/degradation/tables/noise_summary.csv",
                "Thời lượng: 1 phút 5 giây.\nĐây là bước xử lý nhiễu tín hiệu mà đề tài yêu cầu. Đường xám là tín hiệu gốc "
                "của động cơ train 1, nhảy mạnh giữa hai cycle liền kề. Đường xanh là sau EWMA span 10, tức mỗi giá trị nhớ "
                "khoảng 10 cycle gần nhất.\nCông thức: s_t = α·x_t + "
                "(1 − α)·s_{t−1}, α = 2/(span + 1) ≈ 0,18: giá trị mới chiếm 18%, phần còn lại là lịch sử đã làm mượt. Chỉ "
                "dùng quá khứ nên không rò rỉ tương lai.\nĐo bằng độ lệch chuẩn của bước nhảy giữa hai cycle liền kề. Ví dụ "
                f"sensor_4: {vn(float(s4['diff_std_raw']))} xuống {vn(float(s4['diff_std_smoothed']))}. Trên 14 sensor, mức "
                f"giảm từ {vn(min(reductions), 1)}% đến {vn(max(reductions), 1)}%.\nGiới hạn: không có tín hiệu sạch để đối "
                "chiếu, nên đây là mức giảm dao động, chưa chứng minh đã loại đúng nhiễu đo. Span 10 cố định trước, không "
                "tinh chỉnh.")
    picture(s, "slides/figures/denoise_before_after.png", 0.92, 2.45, 7.85, 2.85, align="left")
    rows = [["Sensor", "Gốc", "EWMA", "Giảm"]]
    for key in ("sensor_4", "sensor_9", "sensor_11", "sensor_12", "sensor_14"):
        r = NOISE[key]
        raw, smooth = float(r["diff_std_raw"]), float(r["diff_std_smoothed"])
        rows.append([key, vn(raw, 3 if raw < 1 else 2), vn(smooth, 3 if smooth < 1 else 2), f"{vn(float(r['reduction_pct']), 1)}%"])
    table(s, rows, 8.95, 2.5, [1.1, 0.72, 0.98, 0.7], row_h=0.4, size=11, align_right_from=1)
    text(s, 8.95, 4.98, 3.42, 0.45, "Cách đo: độ lệch chuẩn của chênh lệch giữa hai cycle liền kề, tính riêng từng động cơ.",
         size=10.5, color=SLATE)
    rect(s, 0.92, 5.5, 11.5, 1.05, TINT)
    text(s, 1.15, 5.6, 11.0, 0.9, [
        [("Công thức: ", {"bold": True, "color": BLUE}),
         ("sₜ = α·xₜ + (1 − α)·sₜ₋₁, với α = 2 / (span + 1) ≈ 0,18 khi span = 10", {})],
        [("Vai trò: ", {"bold": True, "color": BLUE}),
         ("tín hiệu đã làm mượt là đầu vào của Health Index và phát hiện bất thường (slide 12); mô hình RUL dùng đặc trưng cửa sổ trên tín hiệu gốc.", {})],
    ], size=13.5, space_after=4)


# ============================================================================ 10. Window
def slide_window():
    fr = P["feature_rows"]
    s = content(3, 10, "Tổng hợp đặc trưng theo cửa sổ thời gian",
                f"Cửa sổ trượt {P['window']} cycle, chỉ dùng quá khứ; 21 sensor × 6 thống kê = {P['feature_columns']} đặc trưng",
                "outputs/features/run_info.json; src/cmapss/features/rolling.py",
                "Thời lượng: 45 giây.\nVí dụ tại cycle 170: cửa sổ gồm cycle 141–170; cycle 171 trở đi là tương lai nên "
                "không được dùng.\nVới mỗi sensor tính 6 thống kê: giá trị cuối (tại cycle t), trung bình, độ lệch chuẩn, "
                "nhỏ nhất, lớn nhất và độ dốc (hệ số góc hồi quy). 21 × 6 = 126 cột. 29 cycle đầu mỗi động cơ chưa đủ cửa sổ "
                "nên bị bỏ, không bù giá trị; FD001 có 17.731 cửa sổ train và 10.196 cửa sổ test.\nVề chuẩn hóa: "
                "StandardScaler đưa mỗi cột về trung bình 0, độ lệch chuẩn 1; nó chỉ học trên phần train của từng fold "
                "cho Ridge, và dùng trước PCA khi tạo Health Index. HGB giữ thang đo gốc vì cây quyết định không phụ thuộc "
                "thang đo.")
    n, start, cw, x0, y0 = 40, 136, 0.19, 0.92, 2.85
    for k in range(n):
        cycle = start + k
        fill = GOLD if cycle == 170 else BLUE if 141 <= cycle <= 170 else "E3E7EE" if cycle > 170 else ICE
        rect(s, x0 + k * cw, y0, cw - 0.03, 0.42, fill, rounded=False)
    text(s, x0 + 34.5 * cw - 0.45, y0 - 0.34, 0.9, 0.3, "t = 170", size=12, bold=True, color=GOLD_DARK, align=PP_ALIGN.CENTER)
    text(s, x0 + 5 * cw, y0 + 0.5, 30 * cw, 0.3, "cửa sổ: cycle 141–170", size=12.5, bold=True, color=BLUE,
         align=PP_ALIGN.CENTER)
    text(s, x0 + 35 * cw - 0.9, y0 + 0.5, 5 * cw + 0.9, 0.3, "từ 171: không dùng", size=11.5, color=SLATE, align=PP_ALIGN.RIGHT)
    text(s, 0.92, 3.8, 7.5, 0.4, "21 sensor × 6 thống kê = 126 đặc trưng", size=18, bold=True, color=BLUE)
    for i, (head, body) in enumerate((("last", "giá trị cuối, tại cycle t"), ("mean", "giá trị trung bình"),
                                      ("std", "độ lệch chuẩn mẫu"), ("min", "giá trị nhỏ nhất"), ("max", "giá trị lớn nhất"),
                                      ("slope", "độ dốc (hệ số góc hồi quy)"))):
        x, y = 0.92 + (i % 3) * 2.5, 4.3 + (i // 3) * 0.9
        rect(s, x, y, 2.36, 0.78, TINT)
        text(s, x + 0.15, y + 0.07, 2.1, 0.34, head, size=15, bold=True, color=TEAL)
        text(s, x + 0.15, y + 0.42, 2.1, 0.3, body, size=12)
    text(s, 0.92, 6.15, 7.5, 0.4, f"{vn(fr['rolling_features_train'], 0)} cửa sổ train · {vn(fr['rolling_features_test'], 0)} "
         "cửa sổ test · 29 cycle đầu bị bỏ, không bù giá trị", size=12.5, color=SLATE)
    rect(s, 8.7, 2.5, 3.72, 4.05, BLUE)
    text(s, 8.95, 2.68, 3.3, 0.36, "Chuẩn hóa", size=16, bold=True, color=WHITE)
    text(s, 8.95, 3.15, 3.25, 3.3, [
        [("ETL: ", {"bold": True, "color": GOLD}), ("thống nhất schema và kiểu dữ liệu.", {})],
        [("StandardScaler: ", {"bold": True, "color": GOLD}),
         ("đưa về trung bình 0, độ lệch chuẩn 1; chỉ học trên phần train của từng fold.", {})],
        [("Dùng cho: ", {"bold": True, "color": GOLD}), ("Ridge và PCA khi tạo Health Index.", {})],
        [("HGB: ", {"bold": True, "color": GOLD}), ("giữ thang đo gốc vì cây quyết định không phụ thuộc thang đo.", {})],
    ], size=13, color=WHITE, space_after=8)


# ============================================================================ 11. Xu hướng
def slide_trend():
    st = D["statistics"]
    s = content(4, 11, "Xu hướng suy giảm của tín hiệu cảm biến",
                f"{st['n_reject']}/{st['n_sensors']} sensor thay đổi có ý nghĩa thống kê giữa 30 cycle đầu và 30 cycle cuối của mỗi động cơ",
                "data/raw/train_FD001.txt; outputs/statistics/tables/sensor_inference.csv",
                "Thời lượng: 40 giây.\nHai hình là quỹ đạo gốc của ba động cơ train dài 178, 199 và 229 cycle: sensor_4 tăng "
                "dần, sensor_7 giảm dần khi tiến gần lúc hỏng. Đây là biểu đồ xu hướng suy giảm theo thời gian.\nKiểm định: "
                "mỗi động cơ lấy trung bình 30 cycle cuối trừ 30 cycle đầu; bootstrap 10.000 lần cho khoảng tin cậy; "
                "Wilcoxon cho p-value; hiệu chỉnh Benjamini–Hochberg (FDR, kiểm soát tỷ lệ phát hiện sai) với ngưỡng "
                f"q = 0,05 trên 21 sensor.\nKết quả: {st['n_reject']} sensor có thay đổi; {len(SAME_DIRECTION)} sensor thay "
                "đổi cùng chiều ở cả 100 động cơ, ví dụ sensor 4, 7, 11, 12, 20; 6 sensor không đổi nên không kiểm định được.")
    picture(s, "slides/figures/sensor_trends.png", 0.92, 2.45, 4.25, 4.0, align="left")
    for i, (head, body) in enumerate((("Ghép cặp trong động cơ", "Δ = trung bình 30 cycle cuối − 30 cycle đầu"),
                                      ("Bootstrap và Wilcoxon", f"Lấy mẫu lại động cơ {vn(st['resamples'], 0)} lần"),
                                      ("Hiệu chỉnh kiểm định bội", "Benjamini–Hochberg (FDR) trên 21 sensor, q = 0,05"))):
        y = 2.55 + i * 0.85
        circle(s, 5.4, y, 0.48, str(i + 1), fill=TEAL, size=14)
        text(s, 6.05, y - 0.03, 6.3, 0.32, head, size=15, bold=True, color=BLUE)
        text(s, 6.05, y + 0.3, 6.3, 0.36, body, size=13)
    rect(s, 5.4, 5.15, 3.2, 1.4, BLUE)
    text(s, 5.62, 5.2, 2.9, 0.7, f"{st['n_reject']}/{st['n_sensors']}", size=38, bold=True, color=WHITE)
    text(s, 5.62, 5.9, 2.9, 0.6, "sensor thay đổi có ý nghĩa thống kê", size=12, color=ICE)
    rect(s, 8.8, 5.15, 3.62, 1.4, TINT)
    text(s, 9.0, 5.25, 3.25, 1.25, f"{len(SAME_DIRECTION)} sensor thay đổi cùng chiều ở 100/100 động cơ, ví dụ "
         f"sensor 4, 7, 11, 12, 20; {st['n_degenerate']} sensor không đổi nên không kiểm định được.", size=12,
         anchor=MSO_ANCHOR.MIDDLE)


# ============================================================================ 12. Bất thường
def slide_anomaly():
    s = content(4, 12, "Phát hiện dấu hiệu bất thường dựa trên Health Index",
                f"Ngưỡng {vn(ACFG['z_threshold'], 0)}σ so với {ACFG['baseline_cycles']} cycle đầu, duy trì {ACFG['persistence']} cycle: "
                f"phát hiện {AN['n_detected']}/{AN['n_engines']} động cơ train, trung vị {vn(AN['lead_median'], 0)} cycle trước khi hỏng",
                "outputs/degradation/tables/anomaly_events.csv; outputs/degradation/summary.json",
                "Thời lượng: 1 phút 20 giây.\nĐây là phần phát hiện dấu hiệu bất thường của đề tài. Health Index (chỉ số sức "
                "khỏe) gộp 21 sensor bằng StandardScaler rồi PCA một thành phần: 14 sensor có xu hướng đã được làm mượt "
                "EWMA, 7 sensor gần như không đổi giữ nguyên. Chỉ số giảm khi động cơ suy giảm. Không dùng nhãn "
                f"RUL.\nQuy tắc cố định trước: lấy trung bình và độ lệch chuẩn của {ACFG['baseline_cycles']} cycle đầu làm "
                f"mức tham chiếu. Khi chỉ số giảm quá {vn(ACFG['z_threshold'], 0)} độ lệch chuẩn và giữ {ACFG['persistence']} cycle "
                "liên tiếp thì báo. Yêu cầu liên tiếp để một gai đơn lẻ không gây báo động.\nHình trái, động cơ 1: báo ở "
                f"cycle 84, hỏng ở cycle 192, tức trước 108 cycle. Hình phải: trung vị {vn(AN['lead_median'], 0)}, nhỏ nhất "
                f"{vn(AN['lead_min'], 0)}, lớn nhất {vn(AN['lead_max'], 0)} cycle.\nCùng quy tắc trên tín hiệu gốc thì trung "
                f"vị chỉ còn {vn(AN_RAW['lead_median'], 1)} cycle. Báo sớm hơn không tự động là tốt hơn: không có nhãn thời "
                "điểm bắt đầu hỏng nên chưa đo được báo động giả.\nDải vàng phía dưới: em đã chạy lại quy tắc với 9 cấu "
                "hình, 2/3/4 sigma nhân 3/5/10 cycle. Cả 9 đều bắt đủ 100 động cơ, động cơ báo muộn nhất vẫn còn ít nhất "
                f"{vn(SENS_LEAD_MIN, 0)} cycle. Chọn tham số chỉ là đánh đổi giữa báo sớm và báo quá sớm.")
    picture(s, "slides/figures/anomaly_health_index.png", 0.92, 2.45, 8.15, 3.1, align="left")
    cards = [
        (f"{AN['n_detected']}/{AN['n_engines']}", "động cơ train được phát hiện trước khi hỏng", BLUE, WHITE, ICE),
        (vn(AN["lead_median"], 0), f"cycle báo trước khi hỏng (trung vị); tứ phân vị {vn(AN['lead_q1'], 0)}–{vn(AN['lead_q3'], 0)}",
         TINT, BLUE, SLATE),
        (f"{vn(AN['detection_life_fraction_median'] * 100, 0)}%", "vòng đời đã qua khi phát hiện (trung vị)", TINT, BLUE, SLATE),
    ]
    for i, (value, label, fill, fg, sub) in enumerate(cards):
        y = 2.5 + i * 1.03
        rect(s, 9.3, y, 3.12, 0.93, fill)
        text(s, 9.5, y + 0.06, 2.8, 0.48, value, size=24, bold=True, color=fg)
        text(s, 9.5, y + 0.53, 2.85, 0.38, label, size=10.5, color=sub)
    rect(s, 0.92, 5.7, 11.5, 0.85, GOLD_TINT)
    assert SENS_ALL_DETECTED
    text(s, 1.15, 5.74, 11.05, 0.78, [
        f"Thử 9 cấu hình (2/3/4σ × 3/5/10 cycle): đều phát hiện {AN['n_engines']}/{AN['n_engines']} động cơ, báo trước ít "
        f"nhất {vn(SENS_LEAD_MIN, 0)} cycle. Nếu không làm mượt: trung vị chỉ còn {vn(AN_RAW['lead_median'], 1)} cycle.",
        "Giới hạn: chưa đo được báo động giả vì dữ liệu không ghi thời điểm bắt đầu suy giảm.",
    ], size=12, anchor=MSO_ANCHOR.MIDDLE)


# ============================================================================ 13. Kết quả RUL
def slide_rul():
    f = M["final"]
    cv = {r["model_name"]: r for r in M["cv"]}
    s = content(5, 13, "Mô hình dự đoán RUL",
                f"HistGradientBoosting (HGB), chọn bằng kiểm định chéo 5 fold chia theo động cơ (GroupKFold); MAE tập test {vn(f['mean_engine_mae'])} cycle",
                "outputs/modeling/tables/model_metrics.csv; outputs/modeling/tables/final_metrics.json; "
                "outputs/modeling/predictions/final_evaluation.parquet",
                "Thời lượng: 1 phút 10 giây.\nCách đánh giá: GroupKFold 5 fold theo động cơ, mọi cửa sổ của một động cơ nằm "
                "cùng một phía. Nếu chia ngẫu nhiên theo dòng, kết quả sẽ tốt hơn thực tế vì hai cửa sổ liền kề trùng 29/30 "
                "cycle. Mỗi động cơ có tổng trọng số bằng 1. Dự đoán trên phần động cơ mà mô hình chưa thấy khi huấn luyện "
                "gọi là dự đoán OOF (out-of-fold).\nBảng trên: so sánh ba mô hình bằng GroupKFold. HGB có MAE 20,21 cycle, "
                "thấp hơn Ridge 3,78 cycle và chưa bằng một nửa MAE của baseline trung vị (46,95).\nTrên test: MAE "
                f"{vn(f['mean_engine_mae'])} cycle, khoảng tin cậy 95% [{vn(M['ci_mae']['ci_lower'], 2)}; "
                f"{vn(M['ci_mae']['ci_upper'], 2)}], RMSE {vn(f['rmse'], 2)}, NASA score {vn(f['nasa_score'], 0)}.\nMAE test "
                "thấp hơn MAE kiểm định chéo vì test chỉ có một điểm cuối mỗi chuỗi, RUL thực tế từ 7 đến 145; "
                "kiểm định chéo tính cả đầu đời có RUL rất lớn.\nĐọc hình: điểm càng gần đường chéo càng đúng; vùng RUL "
                "thấp bám đường chéo tốt hơn.")
    table(s, [
        ["Mô hình (GroupKFold 5 fold)", "MAE", "Độ lệch chuẩn (5 fold)"],
        ["Baseline trung vị", vn(cv["median_baseline"]["mean_engine_mae"]), vn(cv["median_baseline"]["std_engine_mae"])],
        ["Ridge (α = 1)", vn(cv["ridge_alpha_1"]["mean_engine_mae"]), vn(cv["ridge_alpha_1"]["std_engine_mae"])],
        ["HistGradientBoosting", vn(cv["hist_gradient_boosting"]["mean_engine_mae"]),
         vn(cv["hist_gradient_boosting"]["std_engine_mae"])],
    ], 0.92, 2.5, [2.7, 1.0, 2.0], row_h=0.42, size=12.5, highlight=3, align_right_from=1)
    cards = [
        (vn(f["mean_engine_mae"]), "MAE test (cycle)", f"khoảng tin cậy 95%: {vn(M['ci_mae']['ci_lower'])}–{vn(M['ci_mae']['ci_upper'])}",
         BLUE, WHITE, ICE),
        (vn(f["rmse"]), "RMSE (cycle)", "nhạy với sai số lớn", TINT, BLUE, SLATE),
        (vn(f["nasa_score"], 0), "NASA score", "phạt nặng hơn khi dự đoán cao hơn thực tế", TINT, BLUE, SLATE),
    ]
    for i, (value, label, desc, fill, fg, sub) in enumerate(cards):
        x = 0.92 + i * 1.95
        rect(s, x, 4.45, 1.83, 2.1, fill)
        text(s, x + 0.15, 4.55, 1.6, 0.6, value, size=24, bold=True, color=fg)
        text(s, x + 0.15, 5.2, 1.6, 0.35, label, size=12.5, bold=True, color=fg)
        text(s, x + 0.15, 5.6, 1.6, 0.85, desc, size=11, color=sub)
    picture(s, "slides/figures/final_true_vs_predicted.png", 6.95, 2.45, 5.47, 4.1)


# ============================================================================ 14. Sai số
def slide_errors():
    w = M["worst"]
    total = M["n_over"] + M["n_under"]
    s = content(5, 14, "Phân tích sai số dự đoán",
                "Sai số nhỏ hơn khi RUL thực tế nhỏ; mô hình hay dự đoán cao hơn thực tế, nên dễ cảnh báo muộn",
                "outputs/modeling/predictions/final_evaluation.parquet; outputs/diagnostics/error_diagnostics.json",
                "Thời lượng: 40 giây.\nHình: sai số bằng RUL dự đoán trừ RUL thực tế, xếp theo RUL thực tế. MAE chỉ 4,38 cycle "
                "với 25 động cơ có RUL thực tế ≤ 30, tức đúng vùng cần cảnh báo; khoảng 21 cycle khi RUL trên 60.\n"
                f"{M['n_over']} dự đoán cao hơn thực tế, {M['n_under']} thấp hơn. Ví dụ động cơ {w['engine_id']}: RUL thực tế "
                f"{vn(w['y_true'], 0)}, dự đoán {vn(w['y_pred'], 1)}, sai {vn(w['abs_error'], 1)} cycle. 10 động cơ sai nhiều "
                f"nhất đóng góp {vn(DG['top10_nasa_share'] * 100, 1)}% tổng NASA score.\nNguyên nhân: khi động cơ còn khỏe, "
                "tín hiệu cảm biến gần như đi ngang nên khó phân biệt còn 160 hay 300 cycle; trong dự đoán OOF, giá trị dự "
                "đoán hiếm khi vượt quá khoảng 145 cycle.")
    picture(s, "slides/figures/final_residuals_by_rul.png", 0.92, 2.45, 5.9, 4.1, align="left")
    text(s, 7.3, 2.5, 5.1, 0.34, "Chiều của sai số (100 động cơ test)", size=15, bold=True, color=BLUE)
    width = 5.1
    rect(s, 7.3, 2.95, width * M["n_over"] / total, 0.55, GOLD, rounded=False)
    rect(s, 7.3 + width * M["n_over"] / total, 2.95, width * M["n_under"] / total, 0.55, TEAL, rounded=False)
    text(s, 7.42, 2.95, 3.4, 0.55, f"{M['n_over']} dự đoán cao hơn thực tế", size=13, bold=True, color=INK,
         anchor=MSO_ANCHOR.MIDDLE)
    text(s, 7.3 + width * M["n_over"] / total, 2.95, width * M["n_under"] / total, 0.55, f"{M['n_under']} thấp hơn",
         size=12, bold=True, color=WHITE, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    rect(s, 7.3, 3.75, 5.12, 1.85, TINT)
    text(s, 7.52, 3.88, 4.7, 0.34, f"Sai số lớn nhất: động cơ {w['engine_id']}", size=14, bold=True, color=BLUE)
    for i, (label, value) in enumerate((("RUL thực tế", vn(w["y_true"], 0)), ("Dự đoán", vn(w["y_pred"], 1)),
                                        ("Sai số", vn(w["abs_error"], 1)))):
        text(s, 7.52 + i * 1.6, 4.3, 1.55, 0.6, value, size=24, bold=True, color=GOLD_DARK if i == 2 else BLUE)
        text(s, 7.52 + i * 1.6, 4.95, 1.55, 0.4, f"{label} (cycle)", size=11.5, color=SLATE)
    text(s, 7.3, 5.75, 5.12, 0.8, f"MAE chỉ 4,38 cycle khi RUL thực tế ≤ 30. 10 động cơ sai nhiều nhất đóng góp "
         f"{vn(DG['top10_nasa_share'] * 100, 1)}% tổng NASA score.", size=12.5, color=SLATE)


# ============================================================================ 15. Cảnh báo
def slide_warning():
    wg = M["warning"]
    tm, ld = wg["timing"], wg["lead"]
    w30, w40 = WARN["hgb126_30"], WARN["hgb126_40"]
    s = content(5, 15, "Cơ chế cảnh báo sớm hai mức",
                "Ngưỡng RUL dự đoán ≤ 30: không bỏ sót động cơ, báo trước trung vị 30 cycle, nhưng 49% báo muộn",
                "outputs/modeling/tables/warning_events.csv; outputs/course_alignment/20261007/warning_fixed_policies.csv",
                "Thời lượng: 1 phút 20 giây.\nĐây là biểu đồ xu hướng cảnh báo sớm, sản phẩm đặc thù của đề tài, minh họa "
                "trên động cơ train 1. Đường xanh là RUL thực tế, đường đỏ là RUL dự đoán OOF, đường cam là ngưỡng 30 cycle; "
                "động cơ 1 được cảnh báo ở cycle 162, hỏng ở 192. Cùng quy tắc này được áp dụng cho động cơ test ở slide 16.\n"
                f"Bảng bên phải dùng dự đoán OOF trên 100 động cơ train. Ở ngưỡng 30: {tm['before_threshold']} động cơ được "
                f"báo trước khi RUL thực tế chạm 30, {tm['at_threshold']} đúng lúc, {tm['after_threshold']} muộn nhưng vẫn "
                f"trước khi hỏng; không bỏ sót động cơ nào. Thời gian báo trước (lead time) trung vị {vn(ld['median'], 0)} "
                f"cycle, ngắn nhất {vn(ld['min'], 0)}. Vì OOF cũng dùng để chọn mô hình, kết quả này hơi lạc quan.\n"
                "Cảnh báo muộn tập trung ở động cơ suy giảm nhanh. Ngưỡng 40 nâng tỷ lệ kịp thời từ 51% lên 94%, đổi lại "
                "tỷ lệ báo quá sớm tăng khoảng bốn lần, từ 1,3% lên 5,3%. Chọn ngưỡng thật cần chi phí bảo trì, dữ liệu "
                "không có.\nGhép hai mức trên động cơ 1: cycle 84 bất thường, cycle 162 RUL dự đoán ≤ 30, cycle 192 hỏng.")
    _, py, _, ph = picture(s, "slides/figures/warning_trajectory.png", 0.92, 2.45, 6.35, 3.6, align="left")
    text(s, 0.92, py + ph + 0.06, 6.35, 0.3, "Minh họa trên động cơ train 1: cảnh báo mức 2 tại cycle 162; hỏng tại cycle 192.",
         size=11, color=SLATE)
    text(s, 7.6, 2.45, 4.82, 0.3, "Dự đoán OOF trên 100 động cơ train", size=13, bold=True, color=BLUE)
    table(s, [
        ["Ngưỡng RUL", "Kịp thời", "Muộn", "Báo quá sớm"],
        ["≤ 30 cycle", f"{vn(w30['on_time_rate'] * 100, 0)}%", f"{vn(w30['late_or_missed_rate'] * 100, 0)}%",
         f"{vn(w30['early_alarm_burden'] * 100, 1)}%"],
        ["≤ 40 cycle", f"{vn(w40['on_time_rate'] * 100, 0)}%", f"{vn(w40['late_or_missed_rate'] * 100, 0)}%",
         f"{vn(w40['early_alarm_burden'] * 100, 1)}%"],
    ], 7.6, 2.8, [1.3, 1.0, 0.9, 1.62], row_h=0.4, size=12.5, highlight=1, align_right_from=1)
    text(s, 7.6, 4.04, 4.82, 0.5, "Kịp thời: báo trước ít nhất 30 cycle. Báo quá sớm: tỷ lệ cycle có cảnh báo khi RUL "
         "thực tế còn trên 30. OOF: dự đoán trên động cơ mô hình chưa thấy khi huấn luyện.", size=10, color=SLATE)
    rect(s, 7.6, 4.62, 4.82, 1.93, TINT)
    text(s, 7.82, 4.7, 4.4, 0.32, "Hai mức cảnh báo trên động cơ 1", size=14, bold=True, color=BLUE)
    events = [("84", "Mức 1 · Health Index bất thường", GOLD), ("162", "Mức 2 · RUL dự đoán ≤ 30", BLUE),
              ("192", "Hỏng", RED)]
    for i, (cycle, label, color) in enumerate(events):
        y = 5.12 + i * 0.47
        rect(s, 7.82, y, 0.95, 0.38, color)
        text(s, 7.82, y, 0.95, 0.38, f"cycle {cycle}", size=11, bold=True, color=INK if color == GOLD else WHITE,
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        text(s, 8.9, y, 3.4, 0.38, label, size=12.5, anchor=MSO_ANCHOR.MIDDLE)


# ============================================================================ 16. Lịch bảo trì
def slide_maintenance():
    ev, counts, plan = MNT["evaluation"], MNT["status_counts"], MNT["planning"]
    s = content(5, 16, "Báo cáo dự đoán thời điểm bảo trì",
                f"{ev['n_correct_needs_maintenance']}/{ev['n_flagged_needs_maintenance']} động cơ được xếp cần bảo trì đều có "
                f"RUL thực tế ≤ 30, không báo nhầm; bắt được {ev['n_correct_needs_maintenance']}/"
                f"{ev['n_actual_needs_maintenance']} động cơ cần bảo trì",
                "outputs/maintenance/tables/maintenance_schedule.csv; outputs/maintenance/summary.json",
                "Thời lượng: 1 phút 15 giây.\nĐây là sản phẩm đặc thù của đề tài: mỗi động cơ test có một dòng gồm cycle "
                "hiện tại, RUL dự đoán, cycle cần bảo trì (lúc RUL dự đoán chạm 30) và trạng thái. Bảng dưới trích 5 dòng. "
                "Ví dụ động cơ 90 ở cycle 146, RUL dự đoán 31,2, nên cần bảo trì ở cycle 147.\nTrạng thái: cần bảo trì nếu "
                "RUL dự đoán ≤ 30; theo dõi tăng cường nếu Health Index đã báo bất thường; còn lại bình thường.\nHình trái: "
                "100 động cơ test xếp theo RUL dự đoán. Hình phải: "
                f"{counts['Cần bảo trì']} cần bảo trì ngay, {counts['Theo dõi tăng cường']} theo dõi, {counts['Bình thường']} "
                f"bình thường. Kế hoạch: thêm {plan['1–25']} động cơ đến hạn trong 25 cycle tới và {plan['26–50']} trong "
                "26–50 cycle tới.\nKiểm tra bằng nhãn test sau khi đã ghi lịch: "
                f"{ev['n_actual_needs_maintenance']} động cơ thật sự cần bảo trì; {ev['n_correct_needs_maintenance']} được "
                "báo đúng; 4 động cơ còn lại có RUL dự đoán 31–38 nên đến hạn sau 1–8 cycle và vẫn ở mức theo dõi. Cả 20 "
                "động cơ bình thường có RUL thực tế trên 60.\nGiới hạn: mức theo dõi còn rộng, 41/59 động cơ có RUL thực tế "
                "trên 60. 12 động cơ còn xa mới hỏng (RUL thực tế từ 77) có lịch muộn hơn lúc hỏng 4–27 cycle, vì mô hình "
                "dự đoán cao hơn thực tế ở vùng RUL lớn.")
    picture(s, "slides/figures/fleet_status.png", 0.92, 2.45, 7.45, 2.15, align="left")

    # Trích lịch bảo trì theo từng động cơ: đủ ba trạng thái.
    with (ROOT / "outputs/maintenance/tables/maintenance_schedule.csv").open(encoding="utf-8") as file:
        schedule = {row["engine_id"]: row for row in csv.DictReader(file)}
    rows = [["Động cơ", "Trạng thái", "Cycle hiện tại", "RUL dự đoán", "Cycle cần bảo trì"]]
    for engine in ("34", "41", "90", "91", "11"):
        r = schedule[engine]
        rows.append([engine, r["status"], r["current_cycle"], vn(float(r["rul_pred"]), 1), r["maintenance_cycle"]])
    table(s, rows, 0.92, 4.72, [0.95, 2.15, 1.35, 1.3, 1.7], row_h=0.3, size=10.5, align_right_from=2)

    text(s, 8.65, 2.5, 3.77, 0.34, "Đối chiếu với RUL thực tế", size=14, bold=True, color=BLUE)
    head, *body = MNT_CHECK
    table(s, [["Trạng thái", *head[1:]]] + [[row[0], *row[1:]] for row in body],
          8.65, 2.92, [1.62, 0.72, 0.72, 0.71], row_h=0.42, size=11, highlight=1, align_right_from=1)
    rect(s, 8.65, 4.72, 3.77, 1.83, BLUE)
    text(s, 8.85, 4.8, 3.4, 1.7, [
        [(f"{ev['n_correct_needs_maintenance']}/{ev['n_actual_needs_maintenance']}", {"bold": True, "color": GOLD, "size": 20}),
         (" động cơ có RUL thực tế ≤ 30 được xếp cần bảo trì; 4 động cơ còn lại đến hạn sau 1–8 cycle.", {"color": WHITE})],
        [("Giới hạn: ", {"bold": True, "color": GOLD}),
         ("12 động cơ còn xa mới hỏng có lịch muộn hơn lúc hỏng 4–27 cycle.", {"color": ICE})],
    ], size=11, space_after=6)


# ============================================================================ 17. Mở rộng
def slide_extended():
    targets = ["FD002", "FD003", "FD004"]
    row = lambda source, target: next(r for r in D["transfer"]["rows"] if r["source"] == source and r["target"] == target)
    within = [row(t, t)["mae"] for t in targets]
    cross = [row("FD001", t)["mae"] for t in targets]
    o = I["outer"]
    t126, t169 = A["trajectory"]["hgb126"]["mean_engine_mae"], A["trajectory"]["hgb169"]["mean_engine_mae"]
    ts = D["timeseries"]
    s = content(6, 17, "Các thử nghiệm mở rộng",
                f"Mô hình FD001 không dùng tốt được cho FD002–FD004 (cửa sổ {D['transfer']['window']} cycle); ba hướng cải tiến không thay mô hình chính",
                "outputs/cross_dataset/matrix.json; outputs/experiments/; outputs/course_alignment/20261007/summary.json",
                "Thời lượng: 50 giây.\nBiểu đồ: cùng pipeline với cửa sổ 10 cycle vì FD004 có động cơ test ngắn nhất 19 cycle. "
                "Cột xanh là huấn luyện và test trên chính subset đó; cột vàng là dùng lại mô hình FD001 cho subset khác. MAE "
                f"tăng khoảng {vn(min(c / w for c, w in zip(cross, within)), 1)}–{vn(max(c / w for c, w in zip(cross, within)), 1)} "
                "lần. Với FD002 và FD004 chỉ khoảng 15% giá trị sensor nằm trong miền FD001 nên dự đoán gần như hằng số; "
                "FD003 có thêm dạng lỗi ở quạt (Fan).\nBên phải là ba thử nghiệm cải thiện, không cái nào thay mô hình chính: "
                f"kiểm định chéo lồng nhau (nested CV) chọn lại đúng cấu hình gốc, cảnh báo muộn tăng từ {o['baseline']['late']} "
                f"lên {o['weights']['late']}; thêm cột cycle và 42 đặc trưng lịch sử giảm MAE OOF từ {vn(t126)} xuống "
                f"{vn(t169)} nhưng tỷ lệ cảnh báo kịp thời ở ngưỡng 30 giảm từ 51% xuống 45%; ARIMA dự báo Health Index với "
                f"MAE {vn(ts['mae']['ARIMA'], 2)}, nhưng chỉ tốt hơn cách kéo thẳng xu hướng gần nhất (drift) ở 7/10 động cơ.")
    bar_chart(s, 0.92, 2.45, 6.5, 4.1, targets,
              [("Huấn luyện trên chính subset đó", within),
               ("Dùng lại mô hình FD001", cross)],
              [BLUE, GOLD], title="MAE trên tập test (cycle)", max_value=72, decimals=1)
    items = [
        ("Nested CV: cửa sổ, siêu tham số, trọng số",
         f"Chọn lại đúng cấu hình gốc; số cảnh báo muộn tăng từ {o['baseline']['late']} lên {o['weights']['late']}/100."),
        ("Thêm cột cycle và 42 đặc trưng lịch sử",
         f"169 đặc trưng: MAE OOF {vn(t126)} → {vn(t169)}; tỷ lệ cảnh báo kịp thời (ngưỡng 30) giảm từ 51% xuống 45%."),
        ("ARIMA dự báo Health Index",
         f"MAE {vn(ts['mae']['ARIMA'])} so với {vn(ts['mae']['drift'])} của cách kéo thẳng xu hướng (drift); tốt hơn ở 7/10 động cơ."),
    ]
    for i, (head, body) in enumerate(items):
        y = 2.5 + i * 1.37
        rect(s, 7.75, y, 4.67, 1.22, TINT)
        text(s, 7.95, y + 0.1, 4.3, 0.36, head, size=13.5, bold=True, color=BLUE)
        text(s, 7.95, y + 0.5, 4.3, 0.68, body, size=12.5)


# ============================================================================ 18. Kết luận
def slide_conclusion():
    f = M["final"]
    s = content(6, 18, "Kết luận",
                "Trả lời năm câu hỏi nghiên cứu; kết luận chỉ áp dụng cho dữ liệu mô phỏng C-MAPSS FD001",
                "report/sections/conclusion.tex; các artifact đã dẫn ở slide 6–16",
                "Thời lượng: 50 giây.\nQ1: dữ liệu đạt mọi kiểm tra schema, khóa, cycle; Pandas và PySpark khớp tuyệt đối.\n"
                "Q2: EWMA giảm khoảng 86% dao động; cửa sổ 30 cycle tạo 126 đặc trưng.\nQ3: 15/21 sensor thay đổi; Health "
                f"Index phát hiện bất thường ở 100/100 động cơ train, trung vị {vn(AN['lead_median'], 0)} cycle trước khi "
                f"hỏng.\nQ4: MAE test {vn(f['mean_engine_mae'])} cycle, nhỏ nhất ở vùng gần hỏng.\nQ5: hai kết quả khác "
                "nhau. Trên lịch bảo trì của tập test: báo cần bảo trì 21 động cơ, đúng cả 21, bắt được 21/25 động cơ cần "
                "bảo trì. Trên dự đoán OOF của tập train: cảnh báo không bỏ sót động cơ nào, báo trước trung vị 30 cycle, "
                "nhưng 49% muộn; ngưỡng 40 giảm tỷ lệ muộn xuống 6%.\nSản phẩm: lịch dự đoán thời điểm cần bảo trì kèm biểu "
                "đồ xu hướng cảnh báo sớm.")
    ev = MNT["evaluation"]
    cards = [
        ("Q1", "Pipeline dữ liệu", "0 sai khác",
         "đạt mọi kiểm tra; Pandas và PySpark khớp; PySpark nhanh hơn khoảng 3,4 lần khi dữ liệu gấp 10"),
        ("Q2", "Làm mượt, cửa sổ", f"−{vn(DEG['noise_reduction_pct_median'], 0)}%",
         "dao động sau EWMA; 126 đặc trưng từ cửa sổ 30 cycle"),
        ("Q3", "Xu hướng, bất thường", f"{AN['n_detected']}/{AN['n_engines']}",
         f"động cơ train được phát hiện, trung vị {vn(AN['lead_median'], 0)} cycle trước khi hỏng"),
        ("Q4", "Dự đoán RUL", f"{vn(f['mean_engine_mae'])}", "MAE tập test (cycle); chỉ 4,38 khi RUL ≤ 30"),
        ("Q5", "Cảnh báo, lịch bảo trì", f"{ev['n_correct_needs_maintenance']}/{ev['n_flagged_needs_maintenance']}",
         f"báo cần bảo trì trên test đều đúng, bắt {ev['n_correct_needs_maintenance']}/{ev['n_actual_needs_maintenance']}; "
         "OOF train: không bỏ sót, 49% muộn"),
    ]
    for i, (q, head, value, desc) in enumerate(cards):
        x, y = 0.92 + (i % 3) * 3.9, 2.5 + (i // 3) * 2.1
        rect(s, x, y, 3.7, 1.95, TINT)
        circle(s, x + 0.2, y + 0.18, 0.46, q, fill=GOLD, color=INK, size=11)
        text(s, x + 0.8, y + 0.2, 2.8, 0.42, head, size=14, bold=True, color=BLUE, anchor=MSO_ANCHOR.MIDDLE)
        text(s, x + 0.2, y + 0.76, 3.3, 0.5, value, size=24, bold=True, color=BLUE)
        text(s, x + 0.2, y + 1.3, 3.35, 0.6, desc, size=11)
    rect(s, 8.72, 4.6, 3.7, 1.95, BLUE)
    text(s, 8.92, 4.75, 3.3, 0.36, "Sản phẩm đặc thù", size=14, bold=True, color=GOLD)
    text(s, 8.92, 5.15, 3.3, 1.3, "Lịch dự đoán thời điểm cần bảo trì cho 100 động cơ test (slide 16), kèm biểu đồ "
         "xu hướng cảnh báo sớm (slide 15).", size=13, color=WHITE)


# ============================================================================ 19. Khó khăn
def slide_lessons():
    s = content(6, 19, "Khó khăn, bài học và giới hạn",
                "Các vấn đề phát sinh trong quá trình thực hiện, cách xử lý và bài học rút ra",
                "report/sections/discussion.tex (Mục Khó khăn và bài học rút ra)",
                "Thời lượng: 50 giây.\nMột: dữ liệu không ghi lúc bắt đầu hỏng, nên so mỗi động cơ với chính nó lúc khỏe và "
                "nói rõ chưa đo được báo động giả.\nHai: dữ liệu chuỗi thời gian dễ rò rỉ khi chia tập; phải chia theo từng "
                "động cơ, cửa sổ và EWMA chỉ dùng quá khứ.\nBa: PySpark trên Windows thiếu winutils nên ghi qua Arrow, và "
                "luôn đối chiếu từng giá trị với Pandas. Hadoop trong Docker cần image có sẵn Python và hệ số nhân bản "
                "(replication) bằng 1 trên một node; kiểm tra lại bằng fsck.\nBốn: dễ sa vào tối ưu mô hình; với bảo trì dự "
                "đoán, pipeline đúng và cảnh báo dễ hiểu quan trọng hơn vài cycle MAE.\nGiới hạn: dữ liệu mô phỏng; chưa có "
                "chi phí bảo trì; Hadoop và Spark đều chạy trên một máy; kết quả cảnh báo trên OOF train hơi lạc quan vì OOF "
                "cũng dùng để chọn mô hình.")
    table(s, [
        ["Khó khăn", "Cách xử lý", "Bài học"],
        ["Thiếu nhãn thời điểm bất thường", "Lấy 30 cycle đầu của từng động cơ làm mức tham chiếu",
         "Nói rõ mức tham chiếu và phần chưa kiểm chứng được"],
        ["Rò rỉ dữ liệu khi chia tập", "GroupKFold theo động cơ; cửa sổ và EWMA chỉ dùng quá khứ",
         "Chia tập theo từng động cơ, không chia theo từng dòng"],
        ["PySpark thiếu winutils trên Windows; Hadoop một node", "Ghi Parquet qua Arrow; image Hadoop có sẵn Python; replication = 1",
         "Đối chiếu từng giá trị giữa các cách làm; kiểm tra HDFS bằng fsck sau khi ghi"],
        ["Dễ sa vào tối ưu mô hình", "Cố định tiêu chí đánh giá trước; tách thử nghiệm sang phần mở rộng",
         "Ưu tiên pipeline đúng và cảnh báo dễ hiểu hơn việc giảm thêm vài cycle MAE"],
    ], 0.92, 2.5, [2.7, 4.3, 4.5], row_h=0.62, size=12)
    rect(s, 0.92, 5.8, 11.5, 0.75, GOLD_TINT)
    text(s, 1.15, 5.8, 11.05, 0.75, [[("Giới hạn: ", {"bold": True, "color": BLUE}),
                                     ("dữ liệu mô phỏng; chưa có chi phí bảo trì để chọn ngưỡng; Hadoop và Spark chạy trên một máy, "
                                      "chưa có cluster nhiều node hay streaming; kết quả cảnh báo trên OOF train hơi lạc quan.", {})]],
         size=12, anchor=MSO_ANCHOR.MIDDLE)


# ============================================================================ 20. Kết thúc
def slide_end():
    slide = PRS.slides.add_slide(LAYOUT["Slide End"])
    title = slide.placeholders[0]
    title.left, title.top, title.width, title.height = Inches(0.69), Inches(2.0), Inches(6.2), Inches(2.2)
    title.text = "Trân trọng cảm ơn quý thầy cô đã lắng nghe"
    p = title.text_frame.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    p.runs[0].font.size, p.runs[0].font.bold, p.runs[0].font.name = Pt(36), True, FONT
    p.runs[0].font.color.rgb = rgb(BLUE)
    text(slide, 0.69, 4.3, 6.2, 0.5, "Kính mời quý thầy cô đặt câu hỏi", size=20, color=INK)
    text(slide, 0.69, 5.2, 6.2, 0.7, [
        [("Demo dashboard: ", {"bold": True, "color": BLUE}), ("uv run --extra dashboard streamlit run app.py", {})],
        [("Phan Nguyễn Mai Phương · MSHV 25002711", {"color": SLATE})],
    ], size=12.5, space_after=4)
    slide.notes_slide.notes_text_frame.text = (
        "Thời lượng: 10 giây.\nDemo dashboard (khoảng 2,5 phút) đã làm ngay sau slide 19, theo handoff/SPEECH_SCRIPT_VI.pdf. "
        "Cảm ơn quý thầy cô và mời đặt câu hỏi. Không chạy lại huấn luyện khi trình bày."
    )


def main() -> None:
    for build in (slide_cover, slide_agenda, slide_problem, slide_data, slide_labels, slide_pipeline, slide_parity,
                  slide_eda, slide_denoise, slide_window, slide_trend, slide_anomaly, slide_rul,
                  slide_errors, slide_warning, slide_maintenance, slide_extended, slide_conclusion, slide_lessons, slide_end):
        build()
    assert len(PRS.slides) == TOTAL, len(PRS.slides)
    PRS.core_properties.author = "Phan Nguyễn Mai Phương"
    PRS.core_properties.title = "Đề tài 8 · Phân tích dữ liệu IoT và bảo trì dự đoán"
    PRS.save(str(OUT))
    print(f"wrote {OUT.relative_to(ROOT)} ({len(PRS.slides)} slides)")


if __name__ == "__main__":
    main()
