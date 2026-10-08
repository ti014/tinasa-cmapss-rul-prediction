"""Dựng deck thuyết trình Đề tài 8 trên template PowerPoint của IUH.

Học viên: Phan Nguyễn Mai Phương · MSHV 25002711.

Deck gồm 20 slide theo thứ tự rubric: bài toán và dữ liệu → pipeline dữ liệu →
khử nhiễu và cửa sổ thời gian → xu hướng và bất thường → dự đoán RUL và cảnh
báo → mở rộng và kết luận. Mọi số liệu đọc từ artifact đã nghiệm thu; script
không huấn luyện lại mô hình.

Chạy từ thư mục gốc repository:

    uv run --frozen python slides/extract_slide_data.py
    uv run --frozen python -m cmapss.degradation
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
with (ROOT / "outputs/maintenance/tables/status_vs_actual.csv").open(encoding="utf-8") as file:
    MNT_CHECK = list(csv.reader(file))
with (ROOT / "outputs/degradation/tables/anomaly_sensitivity.csv").open(encoding="utf-8") as file:
    SENS = [row for row in csv.DictReader(file) if row["signal"] == "ewma"]
SENS_LEAD_MIN = min(float(row["lead_min"]) for row in SENS)
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


def arrow(slide, x1, y1, x2, y2, color=SLATE):
    connector = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    connector.line.color.rgb = rgb(color)
    connector.line.width = Pt(1.5)
    line = connector.line._get_or_add_ln()
    tail = line.makeelement(qn("a:tailEnd"), {"type": "triangle"})
    line.append(tail)


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
              horizontal=False, legend=True):
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
    subtitle.text = "Phát hiện bất thường, dự đoán RUL và cảnh báo sớm trên dữ liệu cảm biến NASA C-MAPSS"
    p = subtitle.text_frame.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    p.runs[0].font.size, p.runs[0].font.name = Pt(20), FONT
    p.runs[0].font.color.rgb = rgb(INK)
    text(slide, 0.92, 1.3, 11.5, 0.4, "BÁO CÁO CUỐI KỲ · PHÂN TÍCH DỮ LIỆU LỚN · ĐỀ TÀI 8", size=14, bold=True,
         color=GOLD_DARK, align=PP_ALIGN.CENTER)
    items = [
        (str(FD["FD001"]["train_engines"]), "động cơ train FD001"),
        (f"{vn(DEG['noise_reduction_pct_median'], 0)}%", "mức giảm độ dao động (EWMA)"),
        (vn(AN["lead_median"], 0), "cycle phát hiện trước hỏng hóc"),
        (vn(M["final"]["mean_engine_mae"]), "MAE trên tập kiểm tra (cycle)"),
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
        "Thời lượng: 20 giây.\n"
        "Chào thầy/cô. Em là Phan Nguyễn Mai Phương, MSHV 25002711. Đề tài 8: phân tích dữ liệu cảm biến để phát hiện "
        "dấu hiệu bất thường và dự đoán thời điểm cần bảo trì, kèm biểu đồ cảnh báo sớm.\n"
        f"Bốn con số là khung của bài: 100 động cơ train; EWMA giảm khoảng {vn(DEG['noise_reduction_pct_median'], 1)}% "
        f"dao động ngắn hạn; bất thường được phát hiện trung vị {vn(AN['lead_median'], 0)} cycle trước khi hỏng; MAE RUL "
        f"trên 100 động cơ test là {vn(M['final']['mean_engine_mae'], 4)} cycle."
    )


GOLD_DARK = "B37800"


# ============================================================================ 2. Lộ trình
def slide_agenda():
    s = content(1, 2, "Cấu trúc bài trình bày",
                "Trình bày theo trình tự pipeline quản lý dữ liệu: xử lý, phân tích tín hiệu, dự đoán và cảnh báo",
                "report/main.pdf; handoff/SPEECH_SCRIPT_VI.pdf",
                "Thời lượng: 25 giây.\nĐọc nhanh sáu phần. Thứ tự bám quy trình của môn học: thu thập, làm sạch, lưu trữ "
                "trước; rồi khử nhiễu, xu hướng và bất thường; sau cùng là dự đoán RUL và cảnh báo. Năm câu hỏi bên phải "
                "sẽ được trả lời lại ở slide 18. Demo dashboard khoảng 3 phút ở cuối.")
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
        "Dữ liệu có đạt ràng buộc chất lượng; hai cách triển khai ETL có tương đương?",
        "Mức giảm nhiễu đạt được; dữ liệu được tổng hợp theo cửa sổ thời gian thế nào?",
        "Xu hướng biến đổi của tín hiệu; khả năng phát hiện bất thường trước hỏng hóc?",
        "Độ chính xác dự đoán RUL trên tập kiểm tra độc lập?",
        "Hiệu quả của cảnh báo sớm và lịch bảo trì dựa trên dự đoán?",
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
                "Thời lượng: 40 giây.\nRemaining Useful Life (RUL, tuổi thọ hữu ích còn lại) là số cycle vận hành còn lại "
                "trước khi hỏng. Ví dụ động cơ hỏng ở cycle 200 thì tại cycle 170 có RUL bằng 30. Đơn vị là cycle; dữ liệu "
                "không quy đổi ra giờ.\nBảng bên phải đối chiếu từng yêu cầu của đề tài với slide trình bày: khử nhiễu ở "
                "slide 9, window ở slide 10, xu hướng ở slide 11, bất thường ở slide 12, cảnh báo ở slide 15 và lịch bảo trì ở slide 16.")
    text(s, 0.92, 2.5, 5.3, 0.32, "Định nghĩa RUL", size=15, bold=True, color=BLUE)
    x0, x1, y = 1.0, 5.6, 3.4
    tx = x0 + (x1 - x0) * 170 / 200
    rect(s, x0, y, x1 - x0, 0.2, ICE, rounded=False)
    rect(s, tx, y, x1 - tx, 0.2, GOLD, rounded=False)
    dot = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x1 - 0.12), Inches(y - 0.06), Inches(0.32), Inches(0.32))
    dot.fill.solid(); dot.fill.fore_color.rgb = rgb(RED); dot.line.fill.background()
    text(s, x0 - 0.1, y + 0.32, 1.0, 0.28, "cycle 1", size=11, color=SLATE)
    text(s, tx - 0.5, y - 0.42, 1.0, 0.3, "t = 170", size=12, bold=True, color=GOLD_DARK, align=PP_ALIGN.CENTER)
    text(s, x1 - 0.6, y + 0.32, 1.2, 0.5, "hỏng hóc\ncycle 200", size=11, color=RED, align=PP_ALIGN.CENTER)
    text(s, 0.92, 4.35, 5.4, 0.45, "RUL(t) = T − t", size=22, bold=True, color=BLUE)
    text(s, 0.92, 4.85, 5.4, 0.4, "RUL(170) = 200 − 170 = 30 cycle", size=16, bold=True, color=GOLD_DARK)
    text(s, 0.92, 5.4, 5.3, 1.1, "T: cycle hỏng hóc của động cơ. Đầu vào: chuỗi 21 tín hiệu cảm biến theo cycle. Đầu ra: dấu "
         "hiệu bất thường, ước lượng RUL và mức cảnh báo.", size=14)
    table(s, [
        ["Yêu cầu của đề tài", "Phương pháp thực hiện", "Slide"],
        ["Xử lý chuỗi thời gian cảm biến", "Kiểm tra schema, khóa, cycle liên tục; ETL Pandas–PySpark", "6, 7"],
        ["Xử lý nhiễu tín hiệu", "EWMA span 10 theo từng động cơ, chỉ dùng quá khứ", "9"],
        ["Chuẩn hóa, tổng hợp theo window", "StandardScaler cho Health Index; window 30 → 126 đặc trưng", "10"],
        ["Phát hiện dấu hiệu bất thường", "Health Index giảm quá 3σ so với 30 cycle đầu", "12"],
        ["Trực quan hóa xu hướng suy giảm", "Quỹ đạo sensor, kiểm định đầu/cuối, Health Index", "11, 12"],
        ["Dự đoán thời điểm bảo trì, cảnh báo sớm", "Lịch bảo trì cho 100 động cơ test; hai mức cảnh báo", "15, 16"],
    ], 6.55, 2.5, [2.25, 3.0, 0.62], row_h=0.58, size=11.5)


# ============================================================================ 4. Dữ liệu
def slide_data():
    s = content(1, 4, "Bộ dữ liệu NASA C-MAPSS",
                "Bốn subset khác nhau về số điều kiện vận hành và số dạng lỗi; FD001 là đối tượng phân tích chính",
                "data/raw/readme.txt; outputs/cross_dataset/FD00x/etl/run_info.json",
                "Thời lượng: 45 giây.\nC-MAPSS là dữ liệu mô phỏng động cơ turbofan của NASA, có tên trong gợi ý dữ liệu "
                "của đề tài. Mỗi subset có train (chạy đến khi hỏng), test (bị cắt trước khi hỏng) và file nhãn RUL test.\n"
                "FD001: 1 điều kiện vận hành, 1 dạng lỗi, 100 train và 100 test engines, 33.727 quan sát. Mỗi dòng có 26 "
                "trường: engine_id, cycle, 3 setting và 21 sensor.")
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
    text(s, 0.92, 6.2, 11.5, 0.35, "Khóa chính: (split, engine_id, cycle). Dữ liệu không có giá trị thiếu.",
         size=13, color=SLATE)


# ============================================================================ 5. Nhãn
def slide_labels():
    s = content(1, 5, "Xây dựng nhãn RUL và ranh giới tập kiểm tra",
                "Nhãn tập kiểm tra chỉ được sử dụng sau khi đã lưu kết quả dự đoán, nhằm tránh rò rỉ dữ liệu (data leakage)",
                "data/CONTRACTS.md; outputs/modeling/run_info.json; outputs/modeling/tables/final_metrics.json",
                "Thời lượng: 45 giây.\nTrain: mỗi động cơ chạy đến khi hỏng nên RUL = cycle cuối − cycle hiện tại; FD001 có "
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
            text(s, x1 - 0.5, y + 0.5, 1.0, 0.28, "hỏng hóc", size=11.5, color=RED, align=PP_ALIGN.CENTER)
        else:
            dash = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y + 0.25), Inches(x1 + 1.8),
                                          Inches(y + 0.25))
            dash.line.color.rgb = rgb(SLATE); dash.line.dash_style = 4
            dot = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x1 - 0.13), Inches(y + 0.1), Inches(0.3), Inches(0.3))
            dot.fill.solid(); dot.fill.fore_color.rgb = rgb(GOLD); dot.line.fill.background()
            text(s, x1 - 1.6, y + 0.5, 1.8, 0.28, "cycle cuối quan sát", size=11.5, color=GOLD_DARK, align=PP_ALIGN.RIGHT)
            text(s, x1 + 0.3, y + 0.5, 1.6, 0.28, "chưa biết thời điểm hỏng", size=11.5, color=SLATE, align=PP_ALIGN.RIGHT)

    lane(2.55, "Train", f"{FD['FD001']['train_engines']} động cơ vận hành đến hỏng hóc", True)
    lane(3.85, "Test", f"{FD['FD001']['test_engines']} động cơ bị cắt trước hỏng hóc", False)
    rect(s, 0.92, 5.1, 7.4, 1.45, TINT)
    text(s, 1.15, 5.25, 7.0, 1.2, [
        [("Train: ", {"bold": True, "color": BLUE}), (f"RUL(t) = cycle cuối − t → {vn(P['parity'][1]['rows'], 0)} nhãn", {})],
        [("Test: ", {"bold": True, "color": BLUE}),
         ("một dự đoán tại cycle cuối của mỗi động cơ; ghép RUL_FD001.txt sau khi đã lưu dự đoán", {})],
    ], size=14.5, space_after=8)
    rect(s, 8.7, 2.5, 3.72, 4.05, BLUE)
    text(s, 8.95, 2.68, 3.3, 0.35, "Kiểm soát nhãn tập kiểm tra", size=16, bold=True, color=WHITE)
    for i, (head, body) in enumerate((("Không sử dụng", "trong EDA, xây dựng đặc trưng và lựa chọn mô hình"),
                                      ("Ghép sau dự đoán", "nhãn chỉ được ghép khi dự đoán đã được lưu"),
                                      ("Truy vết được", "metadata ghi trạng thái ghép nhãn và mã băm SHA-256"))):
        yy = 3.2 + i * 1.1
        text(s, 8.95, yy, 3.3, 0.4, head, size=19, bold=True, color=GOLD)
        text(s, 8.95, yy + 0.42, 3.3, 0.6, body, size=12.5, color=ICE)


# ============================================================================ 6. Pipeline
def slide_pipeline():
    s = content(2, 6, "Kiến trúc pipeline dữ liệu Bronze–Silver–Gold",
                "Mỗi tầng lưu kết quả kèm mã băm và metadata, bảo đảm khả năng truy vết và tái lập",
                "report/figures/pipeline.pdf; handoff/DATA_WALKTHROUGH.md; outputs/*/run_info.json",
                "Thời lượng: 55 giây.\nBronze là file text gốc. Reader kiểm tra đúng 26 trường, kiểu dữ liệu, khóa duy nhất "
                "và cycle liên tục; train và test được union theo dòng.\nSilver lưu observations và train_targets dạng "
                "Parquet. Gold lưu engine_summary dạng Parquet và SQLite. Từ Silver, bước đặc trưng tạo window 30 cycle; "
                "mô hình dùng GroupKFold, refit và tạo 100 dự đoán test.\nNhánh dưới bên trái là phân tích tín hiệu: khử "
                "nhiễu, Health Index, bất thường, chỉ dùng train. Đây là phần thu thập, làm sạch, chuẩn hóa, tích hợp, "
                "lưu trữ mà rubric chấm 40%.")
    steps = [
        ("Bronze", "Text gốc", "train, test,\nRUL_FD00x"),
        ("Kiểm tra", "Reader", "26 trường, kiểu,\nkhóa, cycle"),
        ("Silver", "Parquet", f"observations\n{vn(FD['FD001']['observations'], 0)} dòng"),
        ("Gold", "Parquet + SQLite", f"engine_summary\n{P['parity'][2]['rows']} dòng"),
        ("Đặc trưng", "Window 30", f"{vn(P['feature_rows']['rolling_features'], 0)} dòng\n{P['feature_columns']} đặc trưng"),
        ("Mô hình", "HGB", "GroupKFold →\nrefit → test"),
    ]
    bw, gap, y, bh = 1.72, 0.236, 2.95, 1.65
    rect(s, 0.92 + 2 * (bw + gap) - 0.08, 2.5, 2 * bw + gap + 0.16, 0.34, GOLD_TINT)
    text(s, 0.92 + 2 * (bw + gap) - 0.08, 2.5, 2 * bw + gap + 0.16, 0.34, "Đối chiếu Pandas–PySpark: sai khác 0", size=12,
         bold=True, color=GOLD_DARK, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    for i, (head, tool, body) in enumerate(steps):
        x = 0.92 + i * (bw + gap)
        fill = BLUE if i == 5 else TEAL if i == 4 else TINT
        dark = i >= 4
        rect(s, x, y, bw, bh, fill)
        text(s, x + 0.12, y + 0.12, bw - 0.24, 0.34, head, size=15, bold=True, color=WHITE if dark else BLUE)
        text(s, x + 0.12, y + 0.47, bw - 0.24, 0.3, tool, size=12, bold=True, color=ICE if dark else TEAL)
        text(s, x + 0.12, y + 0.85, bw - 0.2, 0.75, body, size=11.5, color=WHITE if dark else INK)
        if i < 5:
            arrow(s, x + bw + 0.02, y + bh / 2, x + bw + gap - 0.02, y + bh / 2)
    for col, head, sub, width in ((1, "Khử nhiễu → Health Index → bất thường", "chỉ dùng tập train · slide 9–12", 4.3),
                                  (5, "Cảnh báo, lịch bảo trì", "slide 13–16", 2.9)):
        cx = 0.92 + col * (bw + gap) + bw / 2
        x = min(max(cx - width / 2, 0.92), 12.42 - width)
        arrow(s, cx, y + bh + 0.03, cx, 5.2)
        rect(s, x, 5.22, width, 0.85, WHITE, line=LINE)
        text(s, x + 0.15, 5.3, width - 0.3, 0.38, head, size=13, bold=True, color=BLUE)
        text(s, x + 0.15, 5.68, width - 0.3, 0.3, sub, size=11.5, color=SLATE)
    text(s, 0.92, 6.22, 11.5, 0.35, "Đặc trưng và nhãn được ghép bằng đủ ba cột khóa, kiểm tra quan hệ 1:1.", size=13,
         color=SLATE)


# ============================================================================ 7. Parity và benchmark
def slide_parity():
    bench = {(r["engine"], int(r["scale"])): r for r in D["stage_benchmark"]}
    pandas10, spark10 = bench[("pandas", 10)]["pipeline_seconds"], bench[("spark", 10)]["pipeline_seconds"]
    s = content(2, 7, "Đối chiếu hai cách triển khai Pandas và PySpark",
                f"Kết quả trùng khớp ở mọi tầng dữ liệu; ở quy mô 10×, PySpark cục bộ nhanh hơn Pandas khoảng {vn(pandas10 / spark10, 1)} lần",
                "outputs/etl/parity.json; outputs/benchmarks/20260929/measurements.csv",
                "Thời lượng: 1 phút.\nBảng trái: cùng input, Pandas và PySpark cho cùng schema, khóa, số dòng và giá trị ở "
                "ba boundary; sai khác tuyệt đối lớn nhất bằng 0. Rolling features cũng khớp toàn bộ 27.927 dòng với "
                "tolerance 10^-7.\nBiểu đồ phải: trung vị ba lượt, gồm đọc Parquet, tổng hợp, rolling features và ghi file. "
                f"Pandas tăng gần tuyến tính từ {vn(bench[('pandas', 1)]['pipeline_seconds'], 1)} lên {vn(pandas10, 1)} giây; "
                f"Spark từ {vn(bench[('spark', 1)]['pipeline_seconds'], 1)} lên {vn(spark10, 1)} giây. Ở 1× Pandas nhanh hơn "
                "vì Spark tốn chi phí khởi động.\nGiới hạn: Spark local[4] trên một máy Windows, ghi qua driver; chưa đo "
                "cluster.")
    names = {"silver_observations": "Silver observations", "silver_train_targets": "Silver train_targets",
             "gold_engine_summary": "Gold engine_summary"}
    table(s, [["Boundary", "Số dòng", "Sai khác lớn nhất", "Kết quả"]] + [
        [names.get(p["name"], p["name"]), vn(p["rows"], 0), vn(p["max_abs_diff"], 0), "Khớp" if p["status"] == "passed" else p["status"]]
        for p in P["parity"]
    ], 0.92, 2.5, [2.4, 1.15, 1.55, 1.0], row_h=0.5, size=12.5)
    rect(s, 0.92, 4.7, 6.1, 1.85, TINT)
    text(s, 1.15, 4.85, 5.7, 0.34, "Đối chiếu ở mức từng giá trị", size=14, bold=True, color=BLUE)
    text(s, 1.15, 5.25, 5.7, 1.2, "Rolling features: so toàn bộ 27.927 dòng × 126 đặc trưng, tolerance 10⁻⁷. Cùng "
         "window 30, độ lệch chuẩn mẫu và độ dốc tính theo từng động cơ.", size=13)
    bar_chart(s, 7.3, 2.45, 5.12, 3.6, ["1× · 33.727 dòng", "10× · 337.270 dòng"],
              [("Pandas", [round(bench[("pandas", k)]["pipeline_seconds"]) for k in (1, 10)]),
               ("PySpark local[4]", [round(bench[("spark", k)]["pipeline_seconds"]) for k in (1, 10)])],
              [BLUE, GOLD], title="Thời gian pipeline gồm I/O (giây)")
    text(s, 7.3, 6.12, 5.12, 0.45, "Không gồm thời gian khởi động Spark; đo trên một máy, chưa phải cluster.", size=12,
         bold=True, color=BLUE)


# ============================================================================ 8. EDA
def slide_eda():
    E = D["eda"]
    life = E["lifetime"]
    s = content(2, 8, "Phân tích khám phá dữ liệu",
                "Tuổi thọ động cơ dao động từ 128 đến 362 cycle; bảy cột không biến thiên trên toàn bộ tập train",
                "outputs/eda/figures/engine_lifetimes.png; outputs/eda/tables/constant_engine_counts.csv",
                "Thời lượng: 30 giây.\nTuổi thọ train từ 128 đến 362 cycle, trung vị 199, lệch phải vì có vài động cơ sống "
                "rất lâu.\nBảy cột không đổi trong cả 100 động cơ: sensor 1, 5, 10, 16, 18, 19 và setting_3. Vì vậy bước khử "
                "nhiễu chỉ áp dụng cho 14 sensor có xu hướng. Không làm feature selection từ EDA để tránh rò rỉ vào "
                "cross-validation.")
    picture(s, "outputs/eda/figures/engine_lifetimes.png", 0.92, 2.5, 7.4, 2.55, align="left")
    for i, (label, value) in enumerate((("Ngắn nhất", life["min"]), ("Trung vị", life["50%"]), ("Trung bình", life["mean"]),
                                        ("Dài nhất", life["max"]))):
        stat(s, 0.92 + i * 1.85, 5.25, 1.8, vn(value, 2 if value % 1 else 0), f"{label} (cycle)", size=26)
    rect(s, 8.7, 2.5, 3.72, 4.05, TINT)
    text(s, 8.92, 2.66, 3.3, 0.35, "Cột hằng trên 100/100 động cơ", size=14.5, bold=True, color=BLUE)
    for i, name in enumerate(E["always_constant"]):
        x, y = 8.92 + (i % 2) * 1.7, 3.15 + (i // 2) * 0.52
        rect(s, x, y, 1.6, 0.42, TEAL if name.startswith("setting") else SLATE)
        text(s, x, y, 1.6, 0.42, name, size=12, bold=True, color=WHITE, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    partial = E["partially_constant"][0]
    text(s, 8.92, 5.3, 3.3, 0.4, f"{partial['measurement']} không đổi ở {partial['constant_engines']}/100 động cơ.",
         size=12.5)
    text(s, 8.92, 5.75, 3.3, 0.7, "Khử nhiễu được áp dụng cho 14 sensor còn lại.", size=12.5, color=SLATE)


# ============================================================================ 9. Khử nhiễu
def slide_denoise():
    reductions = [float(r["reduction_pct"]) for r in NOISE.values()]
    s4 = NOISE["sensor_4"]
    s = content(3, 9, "Khử nhiễu tín hiệu bằng EWMA",
                f"Độ dao động ngắn hạn giảm khoảng {vn(DEG['noise_reduction_pct_median'], 0)}% (trung vị trên 14 sensor); xu hướng suy giảm được bảo toàn",
                "outputs/degradation/figures/denoise_before_after.png; outputs/degradation/tables/noise_summary.csv",
                "Thời lượng: 1 phút.\nĐây là bước xử lý nhiễu tín hiệu mà đề tài yêu cầu. Đường xám là tín hiệu gốc của động "
                "cơ train 1, nhảy mạnh giữa hai cycle liền kề. Đường xanh là sau EWMA span 10.\nCông thức: s_t = α·x_t + "
                "(1 − α)·s_{t−1}, α = 2/(span + 1) ≈ 0,18: giá trị mới chiếm 18%, phần còn lại là lịch sử đã làm mượt. Chỉ "
                "dùng quá khứ nên không rò rỉ tương lai.\nĐo bằng độ lệch chuẩn của bước nhảy giữa hai cycle liền kề. Ví dụ "
                f"sensor_4: {vn(float(s4['diff_std_raw']))} xuống {vn(float(s4['diff_std_smoothed']))}. Trên 14 sensor, mức "
                f"giảm từ {vn(min(reductions), 1)}% đến {vn(max(reductions), 1)}%.\nGiới hạn: không có tín hiệu sạch để đối "
                "chiếu, nên đây là mức giảm dao động, chưa chứng minh đã loại đúng nhiễu đo. Span 10 cố định trước, không "
                "tinh chỉnh.")
    picture(s, "outputs/degradation/figures/denoise_before_after.png", 0.92, 2.45, 7.85, 2.85, align="left")
    rows = [["Sensor", "Gốc", "EWMA", "Giảm"]]
    for key in ("sensor_4", "sensor_9", "sensor_11", "sensor_12", "sensor_14"):
        r = NOISE[key]
        raw, smooth = float(r["diff_std_raw"]), float(r["diff_std_smoothed"])
        rows.append([key, vn(raw, 3 if raw < 1 else 2), vn(smooth, 3 if smooth < 1 else 2), f"{vn(float(r['reduction_pct']), 1)}%"])
    table(s, rows, 8.95, 2.5, [1.1, 0.72, 0.98, 0.7], row_h=0.4, size=11, align_right_from=1)
    text(s, 8.95, 4.98, 3.42, 0.45, "Độ lệch chuẩn bước nhảy giữa hai cycle liền kề, trong từng động cơ.", size=10.5,
         color=SLATE)
    rect(s, 0.92, 5.5, 11.5, 1.05, TINT)
    text(s, 1.15, 5.6, 11.0, 0.9, [
        [("Công thức: ", {"bold": True, "color": BLUE}), ("sₜ = α·xₜ + (1 − α)·sₜ₋₁,  α = 2 / (span + 1) ≈ 0,18", {})],
        [("Vai trò: ", {"bold": True, "color": BLUE}),
         ("tín hiệu EWMA là đầu vào của Health Index và phát hiện bất thường (slide 12); mô hình RUL dùng đặc trưng cửa sổ trên tín hiệu gốc.", {})],
    ], size=13.5, space_after=4)


# ============================================================================ 10. Window
def slide_window():
    fr = P["feature_rows"]
    s = content(3, 10, "Tổng hợp đặc trưng theo cửa sổ thời gian",
                f"Cửa sổ trượt nhân quả {P['window']} cycle; 21 sensor × 6 thống kê = {P['feature_columns']} đặc trưng",
                "outputs/features/run_info.json; src/cmapss/features/rolling.py",
                "Thời lượng: 45 giây.\nVí dụ tại cycle 170: window gồm cycle 141–170; cycle 171 trở đi không được dùng.\nVới "
                "mỗi sensor tính 6 thống kê: giá trị cuối, trung bình, độ lệch chuẩn, min, max và độ dốc. 21 × 6 = 126 cột. "
                "29 cycle đầu mỗi động cơ chưa đủ window nên không tạo dòng; FD001 có 17.731 window train và 10.196 window "
                "test.\nVề chuẩn hóa: StandardScaler fit trong training fold cho Ridge và trước PCA khi tạo Health Index; "
                "HGB dùng thang đo gốc vì cây quyết định không phụ thuộc thang đo.")
    n, start, cw, x0, y0 = 40, 136, 0.19, 0.92, 2.85
    for k in range(n):
        cycle = start + k
        fill = GOLD if cycle == 170 else BLUE if 141 <= cycle <= 170 else "E3E7EE" if cycle > 170 else ICE
        rect(s, x0 + k * cw, y0, cw - 0.03, 0.42, fill, rounded=False)
    text(s, x0 + 34.5 * cw - 0.45, y0 - 0.34, 0.9, 0.3, "t = 170", size=12, bold=True, color=GOLD_DARK, align=PP_ALIGN.CENTER)
    text(s, x0 + 5 * cw, y0 + 0.5, 30 * cw, 0.3, "window: cycle 141–170", size=12.5, bold=True, color=BLUE,
         align=PP_ALIGN.CENTER)
    text(s, x0 + 35 * cw - 0.7, y0 + 0.5, 5 * cw + 0.7, 0.3, "171+: không sử dụng", size=11.5, color=SLATE, align=PP_ALIGN.RIGHT)
    text(s, 0.92, 3.8, 7.5, 0.4, "21 sensor × 6 thống kê = 126 đặc trưng", size=18, bold=True, color=BLUE)
    for i, (head, body) in enumerate((("last", "giá trị tại cycle t"), ("mean", "giá trị trung bình"), ("std", "độ lệch chuẩn mẫu"),
                                      ("min", "giá trị nhỏ nhất"), ("max", "giá trị lớn nhất"), ("slope", "hệ số góc hồi quy"))):
        x, y = 0.92 + (i % 3) * 2.5, 4.3 + (i // 3) * 0.9
        rect(s, x, y, 2.36, 0.78, TINT)
        text(s, x + 0.15, y + 0.07, 2.1, 0.34, head, size=15, bold=True, color=TEAL)
        text(s, x + 0.15, y + 0.42, 2.1, 0.3, body, size=12)
    text(s, 0.92, 6.15, 7.5, 0.4, f"{vn(fr['rolling_features_train'], 0)} window train · {vn(fr['rolling_features_test'], 0)} "
         "window test · không padding 29 cycle đầu", size=12.5, color=SLATE)
    rect(s, 8.7, 2.5, 3.72, 4.05, BLUE)
    text(s, 8.95, 2.68, 3.3, 0.36, "Chuẩn hóa", size=16, bold=True, color=WHITE)
    text(s, 8.95, 3.15, 3.25, 3.3, [
        "• ETL chuẩn hóa schema và kiểu dữ liệu.",
        "• StandardScaler fit trong training fold (Ridge) và trước PCA (Health Index).",
        "• HGB dùng thang đo gốc do mô hình cây bất biến với phép biến đổi đơn điệu.",
        "• Các thống kê trên cửa sổ có tác dụng làm trơn bổ sung.",
    ], size=13, color=WHITE, space_after=8)


# ============================================================================ 11. Xu hướng
def slide_trend():
    st = D["statistics"]
    s = content(4, 11, "Xu hướng suy giảm của tín hiệu cảm biến",
                f"{st['n_reject']}/{st['n_sensors']} sensor thay đổi có ý nghĩa thống kê giữa 30 cycle đầu và cuối vòng đời (Wilcoxon, hiệu chỉnh FDR)",
                "outputs/eda/figures/sensor_trajectories_1.png; outputs/statistics/tables/sensor_inference.csv",
                "Thời lượng: 45 giây.\nHai hình là quỹ đạo gốc của ba động cơ train dài 178, 199 và 229 cycle: sensor_4 tăng "
                "dần, sensor_7 giảm dần khi tiến gần lúc hỏng. Đây là biểu đồ xu hướng suy giảm theo thời gian.\nKiểm định: "
                "mỗi động cơ lấy trung bình 30 cycle cuối trừ 30 cycle đầu; bootstrap 10.000 lần cho khoảng tin cậy; "
                "Wilcoxon cho p-value; Benjamini–Hochberg q = 0,05 trên 21 sensor.\nKết quả: 15 sensor có thay đổi; năm "
                "sensor 4, 7, 11, 12, 20 thay đổi cùng hướng ở cả 100 động cơ; 6 sensor hằng không kiểm định được.")
    src = "outputs/eda/figures/sensor_trajectories_1.png"
    _, _, _, h1 = picture(s, src, 0.92, 2.45, 4.05, 2.0, crop=(905, 492, 1817, 930), align="left")
    picture(s, src, 0.92, 2.45 + h1 + 0.06, 4.05, 2.0, crop=(0, 1372, 905, 1817), align="left")
    text(s, 0.92, 6.38, 4.4, 0.25, "Động cơ 87 (178 cycle), 26 (199), 75 (229); giá trị gốc.", size=10, color=SLATE)
    for i, (head, body) in enumerate((("Ghép cặp trong động cơ", "Δ = trung bình 30 cycle cuối − 30 cycle đầu"),
                                      ("Bootstrap và Wilcoxon", f"Lấy mẫu lại động cơ {vn(st['resamples'], 0)} lần"),
                                      ("Hiệu chỉnh đa kiểm định", "Benjamini–Hochberg trên 21 sensor, q = 0,05"))):
        y = 2.55 + i * 0.85
        circle(s, 5.4, y, 0.48, str(i + 1), fill=TEAL, size=14)
        text(s, 6.05, y - 0.03, 6.3, 0.32, head, size=15, bold=True, color=BLUE)
        text(s, 6.05, y + 0.3, 6.3, 0.36, body, size=13)
    rect(s, 5.4, 5.15, 3.2, 1.4, BLUE)
    text(s, 5.62, 5.2, 2.9, 0.7, f"{st['n_reject']}/{st['n_sensors']}", size=38, bold=True, color=WHITE)
    text(s, 5.62, 5.9, 2.9, 0.6, "sensor thay đổi có ý nghĩa thống kê (q = 0,05)", size=12, color=ICE)
    rect(s, 8.8, 5.15, 3.62, 1.4, TINT)
    text(s, 9.0, 5.25, 3.25, 1.25, "Sensor 4, 7, 11, 12, 20 thay đổi cùng chiều ở 100/100 động cơ; "
         f"{st['n_degenerate']} sensor hằng không kiểm định được.", size=12.5, anchor=MSO_ANCHOR.MIDDLE)


# ============================================================================ 12. Bất thường
def slide_anomaly():
    s = content(4, 12, "Phát hiện dấu hiệu bất thường dựa trên Health Index",
                f"Ngưỡng {vn(ACFG['z_threshold'], 0)}σ so với {ACFG['baseline_cycles']} cycle đầu, duy trì {ACFG['persistence']} cycle: "
                f"phát hiện {AN['n_detected']}/{AN['n_engines']} động cơ, trung vị {vn(AN['lead_median'], 0)} cycle trước hỏng hóc",
                "outputs/degradation/figures/anomaly_health_index.png; outputs/degradation/summary.json",
                "Thời lượng: 1 phút 15 giây.\nĐây là phần phát hiện dấu hiệu bất thường của đề tài. Health Index gộp 21 sensor "
                "đã khử nhiễu bằng StandardScaler rồi PCA một thành phần; chỉ số giảm khi động cơ suy giảm. Không dùng nhãn "
                f"RUL.\nQuy tắc cố định trước: lấy trung bình và độ lệch chuẩn của {ACFG['baseline_cycles']} cycle đầu làm "
                f"mức tham chiếu. Khi chỉ số giảm quá {vn(ACFG['z_threshold'], 0)} độ lệch chuẩn và giữ {ACFG['persistence']} cycle "
                "liên tiếp thì báo. Yêu cầu liên tiếp để một gai đơn lẻ không gây báo động.\nHình trái, động cơ 1: báo ở "
                f"cycle 84, hỏng ở cycle 192, tức trước 108 cycle. Hình phải: trung vị {vn(AN['lead_median'], 0)}, nhỏ nhất "
                f"{vn(AN['lead_min'], 0)}, lớn nhất {vn(AN['lead_max'], 0)} cycle.\nCùng quy tắc trên tín hiệu gốc thì trung "
                f"vị chỉ còn {vn(AN_RAW['lead_median'], 1)} cycle. Báo sớm hơn không tự động là tốt hơn: không có nhãn thời "
                "điểm bắt đầu hỏng nên chưa đo được báo động giả.\nDải vàng phía dưới: em đã chạy lại quy tắc với 9 cấu "
                "hình, 2/3/4 sigma nhân 3/5/10 cycle. Cả 9 đều bắt đủ 100 động cơ, động cơ báo muộn nhất vẫn còn ít nhất "
                f"{vn(SENS_LEAD_MIN, 0)} cycle. Chọn tham số chỉ là đánh đổi giữa báo sớm và báo quá sớm.")
    picture(s, "outputs/degradation/figures/anomaly_health_index.png", 0.92, 2.45, 8.15, 3.1, align="left")
    cards = [
        (f"{AN['n_detected']}/{AN['n_engines']}", "động cơ được phát hiện trước hỏng hóc", BLUE, WHITE, ICE),
        (vn(AN["lead_median"], 0), f"cycle phát hiện trước hỏng hóc (trung vị); IQR {vn(AN['lead_q1'], 0)}–{vn(AN['lead_q3'], 0)}",
         TINT, BLUE, SLATE),
        (f"{vn(AN['detection_life_fraction_median'] * 100, 0)}%", "vòng đời tại thời điểm phát hiện (trung vị)", TINT, BLUE, SLATE),
    ]
    for i, (value, label, fill, fg, sub) in enumerate(cards):
        y = 2.5 + i * 1.03
        rect(s, 9.3, y, 3.12, 0.93, fill)
        text(s, 9.5, y + 0.06, 2.8, 0.48, value, size=24, bold=True, color=fg)
        text(s, 9.5, y + 0.53, 2.85, 0.38, label, size=10.5, color=sub)
    rect(s, 0.92, 5.7, 11.5, 0.85, GOLD_TINT)
    assert SENS_ALL_DETECTED
    text(s, 1.15, 5.74, 11.05, 0.78, f"Độ nhạy: với 9 cấu hình (2/3/4σ × 3/5/10 cycle), quy tắc đều phát hiện {AN['n_engines']}/{AN['n_engines']} động cơ, "
         f"lead time nhỏ nhất ≥ {vn(SENS_LEAD_MIN, 0)} cycle. Không khử nhiễu: trung vị còn {vn(AN_RAW['lead_median'], 1)} cycle. "
         "Giới hạn: chưa ước lượng được tỷ lệ dương tính giả do thiếu nhãn thời điểm bắt đầu suy giảm.",
         size=12, anchor=MSO_ANCHOR.MIDDLE)


# ============================================================================ 13. Protocol
def slide_protocol():
    s = content(5, 13, "Đánh giá theo động cơ để tránh data leakage",
                "5-fold GroupKFold: mọi window của một động cơ nằm cùng một phía của phép chia",
                "outputs/modeling/tables/evaluation_protocol.csv; report/sections/methodology.tex",
                "Thời lượng: 40 giây.\n100 động cơ train chia thành 5 nhóm 20 động cơ; mỗi fold dùng một nhóm làm "
                "validation.\nVì sao không chia ngẫu nhiên theo dòng: hai window liền kề của cùng động cơ trùng 29/30 cycle. "
                "Nếu một window ở train và window kề nó ở validation, metric sẽ đẹp giả tạo.\nBốn bước bên phải: chọn mô hình "
                "bằng MAE trung bình theo động cơ; cân trọng số để mỗi động cơ đóng góp như nhau; refit trên 17.731 window; "
                "dự đoán một điểm mỗi động cơ test rồi mới ghép nhãn.")
    for f in range(5):
        y = 2.6 + f * 0.66
        text(s, 0.92, y, 0.9, 0.5, f"Fold {f + 1}", size=13, bold=True, anchor=MSO_ANCHOR.MIDDLE)
        for g in range(5):
            rect(s, 1.9 + g * 1.07, y, 1.0, 0.5, GOLD if g == f else ICE)
            if g == f:
                text(s, 1.9 + g * 1.07, y, 1.0, 0.5, "validation", size=11, bold=True, color=INK, align=PP_ALIGN.CENTER,
                     anchor=MSO_ANCHOR.MIDDLE)
    text(s, 0.92, 5.95, 6.3, 0.3, "Mỗi ô là một nhóm 20 động cơ train; màu nhạt là phần fit.", size=11.5, color=SLATE)
    for i, (head, body) in enumerate((("Chọn mô hình", "MAE trung bình theo động cơ trên dự đoán OOF"),
                                      ("Cân bằng động cơ", "tổng trọng số mỗi động cơ bằng 1 trong fold"),
                                      ("Refit", f"toàn bộ {vn(P['feature_rows']['rolling_features_train'], 0)} window train"),
                                      ("Dự đoán test", "1 điểm mỗi động cơ, sau đó mới ghép nhãn"))):
        y = 2.55 + i * 0.82
        circle(s, 7.6, y, 0.48, str(i + 1), size=14)
        text(s, 8.25, y - 0.03, 4.2, 0.32, head, size=15, bold=True, color=BLUE)
        text(s, 8.25, y + 0.29, 4.2, 0.34, body, size=13)
    rect(s, 7.6, 5.85, 4.82, 0.7, GOLD_TINT)
    text(s, 7.78, 5.88, 4.5, 0.64, "Hai window liền kề trùng 29/30 cycle; chia theo dòng làm metric đẹp giả tạo.",
         size=12.5, anchor=MSO_ANCHOR.MIDDLE)


# ============================================================================ 14. Kết quả RUL
def slide_rul():
    f = M["final"]
    cv = {r["model_name"]: r for r in M["cv"]}
    s = content(5, 13, "Mô hình dự đoán RUL",
                f"HistGradientBoosting, lựa chọn bằng GroupKFold 5 fold theo động cơ; MAE trên tập kiểm tra: {vn(f['mean_engine_mae'])} cycle",
                "outputs/modeling/tables/{model_metrics.csv,final_metrics.json}; figures/final_true_vs_predicted.png",
                "Thời lượng: 1 phút 10 giây.\nCách đánh giá: GroupKFold 5 fold theo động cơ, mọi window của một động cơ nằm cùng một phía; chia ngẫu nhiên theo dòng sẽ làm metric đẹp giả tạo vì hai window liền kề trùng 29/30 cycle. Mỗi động cơ có tổng trọng số bằng 1.\nBảng trên: so sánh ba mô hình bằng GroupKFold. HGB có MAE 20,21 cycle, thấp hơn Ridge "
                "3,78 cycle và thấp hơn median baseline hơn một nửa.\nTrên test: MAE "
                f"{vn(f['mean_engine_mae'], 4)} cycle, khoảng tin cậy 95% [{vn(M['ci_mae']['ci_lower'], 2)}; "
                f"{vn(M['ci_mae']['ci_upper'], 2)}], RMSE {vn(f['rmse'], 2)}, NASA score {vn(f['nasa_score'], 0)}.\nMAE test "
                "thấp hơn MAE cross-validation vì test chỉ có một điểm cuối mỗi chuỗi, RUL thật từ 7 đến 145; "
                "cross-validation tính cả đầu đời có RUL rất lớn.\nĐọc hình: điểm càng gần đường chéo càng đúng; vùng RUL "
                "thấp bám đường chéo tốt hơn.")
    table(s, [
        ["Mô hình (GroupKFold 5 fold)", "MAE", "Độ lệch chuẩn"],
        ["Median baseline", vn(cv["median_baseline"]["mean_engine_mae"]), vn(cv["median_baseline"]["std_engine_mae"])],
        ["Ridge (α = 1)", vn(cv["ridge_alpha_1"]["mean_engine_mae"]), vn(cv["ridge_alpha_1"]["std_engine_mae"])],
        ["HistGradientBoosting", vn(cv["hist_gradient_boosting"]["mean_engine_mae"]),
         vn(cv["hist_gradient_boosting"]["std_engine_mae"])],
    ], 0.92, 2.5, [3.0, 1.2, 1.5], row_h=0.42, size=12.5, highlight=3, align_right_from=1)
    cards = [
        (vn(f["mean_engine_mae"]), "MAE (cycle)", f"CI 95%: {vn(M['ci_mae']['ci_lower'])}–{vn(M['ci_mae']['ci_upper'])}",
         BLUE, WHITE, ICE),
        (vn(f["rmse"]), "RMSE (cycle)", "nhạy với sai số lớn", TINT, BLUE, SLATE),
        (vn(f["nasa_score"], 0), "NASA score", "phạt nặng hơn khi ước lượng cao", TINT, BLUE, SLATE),
    ]
    for i, (value, label, desc, fill, fg, sub) in enumerate(cards):
        x = 0.92 + i * 1.95
        rect(s, x, 4.45, 1.83, 2.1, fill)
        text(s, x + 0.15, 4.55, 1.6, 0.6, value, size=24, bold=True, color=fg)
        text(s, x + 0.15, 5.2, 1.6, 0.35, label, size=12.5, bold=True, color=fg)
        text(s, x + 0.15, 5.6, 1.6, 0.85, desc, size=11, color=sub)
    picture(s, "outputs/modeling/figures/final_true_vs_predicted.png", 6.95, 2.45, 5.47, 4.1)


# ============================================================================ 15. Sai số
def slide_errors():
    w = M["worst"]
    total = M["n_over"] + M["n_under"]
    s = content(5, 14, "Phân tích sai số dự đoán",
                "Sai số giảm khi RUL nhỏ; mô hình có xu hướng ước lượng cao, tương ứng với cảnh báo muộn",
                "outputs/modeling/figures/final_residuals_by_rul.png; outputs/diagnostics/error_diagnostics.json",
                "Thời lượng: 45 giây.\nHình: sai số bằng dự đoán trừ thật, theo RUL thật. MAE chỉ 4,38 cycle với 25 động cơ "
                "có RUL thật ≤ 30, tức đúng vùng cần cảnh báo; khoảng 21 cycle khi RUL trên 60.\n"
                f"{M['n_over']} dự đoán cao hơn thật, {M['n_under']} thấp hơn. Ví dụ động cơ {w['engine_id']}: RUL thật "
                f"{vn(w['y_true'], 0)}, dự đoán {vn(w['y_pred'], 1)}, sai {vn(w['abs_error'], 1)} cycle. Mười trường hợp sai "
                f"lớn nhất chiếm {vn(DG['top10_nasa_share'] * 100, 1)}% NASA score.\nNguyên nhân: khi động cơ còn khỏe, "
                "Health Index đi ngang nên khó phân biệt còn 160 hay 300 cycle; trên OOF dự đoán dừng quanh 145 cycle.")
    picture(s, "outputs/modeling/figures/final_residuals_by_rul.png", 0.92, 2.45, 5.9, 4.1, align="left")
    text(s, 7.3, 2.5, 5.1, 0.34, "Phân bố dấu của sai số (100 động cơ kiểm tra)", size=15, bold=True, color=BLUE)
    width = 5.1
    rect(s, 7.3, 2.95, width * M["n_over"] / total, 0.55, GOLD, rounded=False)
    rect(s, 7.3 + width * M["n_over"] / total, 2.95, width * M["n_under"] / total, 0.55, TEAL, rounded=False)
    text(s, 7.42, 2.95, 2.6, 0.55, f"{M['n_over']} ước lượng cao", size=14, bold=True, color=INK, anchor=MSO_ANCHOR.MIDDLE)
    text(s, 7.3 + width * M["n_over"] / total, 2.95, width * M["n_under"] / total, 0.55, f"{M['n_under']} thấp",
         size=12, bold=True, color=WHITE, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    rect(s, 7.3, 3.75, 5.12, 1.85, TINT)
    text(s, 7.52, 3.88, 4.7, 0.34, f"Sai số lớn nhất: động cơ {w['engine_id']}", size=14, bold=True, color=BLUE)
    for i, (label, value) in enumerate((("RUL thật", vn(w["y_true"], 0)), ("Dự đoán", vn(w["y_pred"], 1)),
                                        ("Sai số", vn(w["abs_error"], 1)))):
        text(s, 7.52 + i * 1.6, 4.3, 1.55, 0.6, value, size=24, bold=True, color=GOLD_DARK if i == 2 else BLUE)
        text(s, 7.52 + i * 1.6, 4.95, 1.55, 0.4, f"{label} (cycle)", size=11.5, color=SLATE)
    text(s, 7.3, 5.75, 5.12, 0.8, f"MAE 4,38 cycle khi RUL thật ≤ 30. Mười sai số lớn nhất chiếm "
         f"{vn(DG['top10_nasa_share'] * 100, 1)}% NASA score.", size=13, bold=True, color=BLUE)


# ============================================================================ 16. Cảnh báo
def slide_warning():
    wg = M["warning"]
    tm, ld = wg["timing"], wg["lead"]
    w30, w40 = WARN["hgb126_30"], WARN["hgb126_40"]
    s = content(5, 15, "Cơ chế cảnh báo sớm hai mức",
                "Ngưỡng RUL ≤ 30: không bỏ sót động cơ, lead time trung vị 30 cycle; 49% cảnh báo muộn so với ngưỡng",
                "outputs/modeling/figures/warning_trajectory.png; outputs/course_alignment/20261007/warning_fixed_policies.csv",
                "Thời lượng: 1 phút 30 giây.\nĐây là biểu đồ xu hướng cảnh báo sớm, sản phẩm đặc thù của đề tài. Đường xanh "
                "là RUL thật, đường đỏ là RUL dự đoán OOF, đường cam là ngưỡng 30 cycle; động cơ 1 được cảnh báo ở cycle "
                "162, hỏng ở 192.\n"
                f"Ngưỡng 30 trên 100 động cơ: {tm['before_threshold']} cảnh báo trước khi RUL thật chạm 30, "
                f"{tm['at_threshold']} đúng lúc, {tm['after_threshold']} trễ nhưng vẫn trước khi hỏng, không bỏ lỡ. Lead "
                f"time trung vị {vn(ld['median'], 0)} cycle, ngắn nhất {vn(ld['min'], 0)}.\nCảnh báo trễ tập trung ở động cơ "
                "suy giảm nhanh. Ngưỡng 40 nâng tỷ lệ kịp thời từ 51% lên 94%, đổi lại thời gian báo sớm tăng khoảng bốn "
                "lần. Chọn ngưỡng thật cần chi phí bảo trì, dữ liệu không có.\nGhép hai mức trên động cơ 1: cycle 84 bất "
                "thường, cycle 162 RUL ≤ 30, cycle 192 hỏng.")
    _, py, _, ph = picture(s, "outputs/modeling/figures/warning_trajectory.png", 0.92, 2.45, 6.35, 3.6, align="left")
    text(s, 0.92, py + ph + 0.06, 6.35, 0.3, "Động cơ train 1: cảnh báo mức 2 tại cycle 162; hỏng hóc tại cycle 192.", size=11,
         color=SLATE)
    table(s, [
        ["Ngưỡng RUL", "Kịp thời", "Muộn", "Báo sớm"],
        ["≤ 30 cycle", f"{vn(w30['on_time_rate'] * 100, 0)}%", f"{vn(w30['late_or_missed_rate'] * 100, 0)}%",
         f"{vn(w30['early_alarm_burden'] * 100, 1)}%"],
        ["≤ 40 cycle", f"{vn(w40['on_time_rate'] * 100, 0)}%", f"{vn(w40['late_or_missed_rate'] * 100, 0)}%",
         f"{vn(w40['early_alarm_burden'] * 100, 1)}%"],
    ], 7.6, 2.5, [1.4, 1.1, 0.95, 1.37], row_h=0.42, size=12.5, highlight=1, align_right_from=1)
    text(s, 7.6, 3.82, 4.82, 0.45, "Kịp thời: lead time ≥ 30 cycle. Báo sớm: tỷ lệ cycle có cảnh báo khi RUL thực tế "
         "lớn hơn 30.", size=10.5, color=SLATE)
    rect(s, 7.6, 4.4, 4.82, 2.15, TINT)
    text(s, 7.82, 4.5, 4.4, 0.32, "Hai mức cảnh báo trên động cơ 1", size=14, bold=True, color=BLUE)
    events = [("84", "Mức 1 · Health Index bất thường", GOLD), ("162", "Mức 2 · RUL dự đoán ≤ 30", BLUE),
              ("192", "Hỏng hóc", RED)]
    for i, (cycle, label, color) in enumerate(events):
        y = 4.95 + i * 0.5
        rect(s, 7.82, y, 0.95, 0.38, color)
        text(s, 7.82, y, 0.95, 0.38, f"cycle {cycle}", size=11, bold=True, color=INK if color == GOLD else WHITE,
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        text(s, 8.9, y, 3.4, 0.38, label, size=12.5, anchor=MSO_ANCHOR.MIDDLE)


# ============================================================================ 16. Lịch bảo trì
def slide_maintenance():
    ev, counts, plan = MNT["evaluation"], MNT["status_counts"], MNT["planning"]
    s = content(5, 16, "Báo cáo dự đoán thời điểm bảo trì",
                f"{ev['n_correct_needs_maintenance']}/{ev['n_flagged_needs_maintenance']} động cơ được xếp mức cần bảo trì có RUL thực tế ≤ 30; "
                f"không có trường hợp dương tính giả",
                "outputs/maintenance/tables/maintenance_schedule.csv; outputs/maintenance/summary.json",
                "Thời lượng: 1 phút 15 giây.\nĐây là sản phẩm đặc thù của đề tài: mỗi động cơ test có một dòng gồm cycle "
                "hiện tại, RUL dự đoán, cycle dự kiến cần bảo trì (lúc RUL dự đoán chạm 30) và trạng thái. Ví dụ động cơ 90 "
                "ở cycle 146, RUL dự đoán 31,2, nên cần bảo trì ở cycle 147.\nTrạng thái: cần bảo trì nếu RUL dự đoán ≤ 30; "
                "theo dõi tăng cường nếu Health Index đã báo bất thường; còn lại bình thường.\nHình trái: "
                f"{counts['Cần bảo trì']} cần bảo trì ngay, {counts['Theo dõi tăng cường']} theo dõi, {counts['Bình thường']} "
                f"bình thường. Kế hoạch: thêm {plan['1–25']} động cơ đến hạn trong 25 cycle tới và {plan['26–50']} trong "
                "26–50 cycle.\nKiểm tra bằng nhãn test sau khi đã ghi lịch: "
                f"{ev['n_actual_needs_maintenance']} động cơ thật sự cần bảo trì; {ev['n_correct_needs_maintenance']} được "
                "báo đúng; 4 động cơ còn lại có RUL dự đoán 31–38 nên đến hạn sau 1–8 cycle và vẫn ở mức theo dõi. Cả 20 "
                "động cơ bình thường có RUL thật trên 60. Giới hạn: mức theo dõi còn rộng, 41/59 động cơ có RUL thật trên 60.")
    picture(s, "outputs/maintenance/figures/fleet_status.png", 0.92, 2.45, 7.45, 2.8, align="left")
    for i, (value, label, fill, fg, sub) in enumerate((
            (str(plan["Ngay bây giờ"]), "cần bảo trì ngay", GOLD, INK, INK),
            (str(plan["1–25"]), "đến hạn trong 1–25 cycle tới", TINT, BLUE, SLATE),
            (str(plan["26–50"]), "đến hạn trong 26–50 cycle", TINT, BLUE, SLATE))):
        x = 0.92 + i * 2.5
        rect(s, x, 5.4, 2.35, 1.15, fill)
        text(s, x + 0.18, 5.47, 2.0, 0.55, value, size=26, bold=True, color=fg)
        text(s, x + 0.18, 6.02, 2.05, 0.45, label, size=11.5, color=sub)
    text(s, 8.65, 2.5, 3.77, 0.34, "Đối chiếu với RUL thực tế", size=14, bold=True, color=BLUE)
    head, *body = MNT_CHECK
    table(s, [["Trạng thái", *head[1:]]] + [[row[0], *row[1:]] for row in body],
          8.65, 2.92, [1.62, 0.72, 0.72, 0.71], row_h=0.42, size=11, highlight=1, align_right_from=1)
    rect(s, 8.65, 4.85, 3.77, 1.7, BLUE)
    text(s, 8.85, 4.95, 3.4, 1.55, [
        [(f"{ev['n_correct_needs_maintenance']}/{ev['n_actual_needs_maintenance']}", {"bold": True, "color": GOLD, "size": 22}),
         ("  động cơ có RUL thực tế ≤ 30 được xếp mức cần bảo trì", {"color": WHITE})],
        [("4 trường hợp còn lại được xếp đến hạn sau 1–8 cycle, ở mức theo dõi tăng cường.", {"color": ICE})],
    ], size=12, space_after=6)


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
                f"Khả năng tổng quát hóa sang FD002–FD004 (cửa sổ {D['transfer']['window']} cycle) và các phương án cải tiến; kết luận chính không đổi",
                "outputs/cross_dataset/matrix.json; outputs/experiments/; outputs/course_alignment/20261007/summary.json",
                "Thời lượng: 1 phút.\nBiểu đồ: cùng pipeline với window 10 vì FD004 có động cơ test ngắn nhất 19 cycle. Cột "
                "xanh là huấn luyện và test trong cùng subset; cột vàng là giữ mô hình FD001 áp dụng sang subset khác. MAE "
                "tăng khoảng 2–3 lần. Với FD002 và FD004 chỉ khoảng 15% giá trị sensor nằm trong miền FD001 nên dự đoán gần "
                "như hằng số; FD003 có thêm dạng lỗi Fan.\nBên phải là ba thử nghiệm cải thiện, không cái nào thay mô hình "
                f"chính: nested CV quay về cấu hình gốc, cảnh báo trễ {o['baseline']['late']} → {o['weights']['late']}; thêm "
                f"42 đặc trưng lịch sử giảm MAE OOF {vn(t126)} → {vn(t169)} nhưng tỷ lệ cảnh báo kịp thời ở ngưỡng 30 giảm "
                f"51% → 45%; ARIMA dự báo Health Index MAE {vn(ts['mae']['ARIMA'], 2)} nhưng chỉ tốt hơn drift ở 7/10 động cơ.")
    bar_chart(s, 0.92, 2.45, 6.5, 4.1, targets,
              [("Huấn luyện nội bộ", [round(v, 1) for v in within]),
               ("Áp dụng mô hình FD001", [round(v, 1) for v in cross])],
              [BLUE, GOLD], title="MAE trên tập kiểm tra (cycle)", max_value=72)
    items = [
        ("Nested CV: window, siêu tham số, trọng số",
         f"Quy trình lựa chọn hội tụ về cấu hình gốc; số cảnh báo muộn {o['baseline']['late']} → {o['weights']['late']}/100."),
        ("Thêm 42 đặc trưng lịch sử (169 đặc trưng)",
         f"MAE OOF {vn(t126)} → {vn(t169)}; tỷ lệ cảnh báo kịp thời (ngưỡng 30) giảm 51% → 45%."),
        ("ARIMA dự báo Health Index",
         f"MAE {vn(ts['mae']['ARIMA'])} so với {vn(ts['mae']['drift'])} của phương pháp drift; vượt trội ở 7/10 động cơ."),
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
                "Trả lời năm câu hỏi nghiên cứu; kết luận giới hạn trong dữ liệu mô phỏng C-MAPSS FD001",
                "report/sections/conclusion.tex; các artifact đã dẫn ở slide 6–16",
                "Thời lượng: 1 phút.\nQ1: dữ liệu đạt mọi kiểm tra; Pandas và PySpark khớp tuyệt đối.\nQ2: EWMA giảm khoảng "
                "86% dao động; window 30 tạo 126 đặc trưng.\nQ3: 15/21 sensor thay đổi; Health Index phát hiện bất thường ở "
                f"100/100 động cơ, trung vị {vn(AN['lead_median'], 0)} cycle trước khi hỏng.\nQ4: MAE test "
                f"{vn(f['mean_engine_mae'])} cycle, nhỏ nhất ở vùng gần hỏng.\nQ5: không bỏ lỡ, lead time trung vị 30 cycle, "
                "nhưng 49% trễ; ngưỡng 40 giảm trễ xuống 6%. Lịch bảo trì báo cần bảo trì 21 động cơ, đúng cả 21.\nSản phẩm: lịch dự đoán thời điểm cần bảo trì kèm biểu đồ xu hướng cảnh báo sớm.")
    cards = [
        ("Q1", "Pipeline dữ liệu", "0 sai khác", "Pandas và PySpark trùng khớp; PySpark nhanh hơn ~3,4× ở quy mô 10×"),
        ("Q2", "Khử nhiễu, window", f"−{vn(DEG['noise_reduction_pct_median'], 0)}%", "độ dao động sau EWMA; 126 đặc trưng cửa sổ 30 cycle"),
        ("Q3", "Xu hướng, bất thường", f"{AN['n_detected']}/{AN['n_engines']}",
         f"động cơ được phát hiện, trung vị {vn(AN['lead_median'], 0)} cycle trước hỏng hóc"),
        ("Q4", "Dự đoán RUL", f"{vn(f['mean_engine_mae'])}", "MAE trên tập kiểm tra (cycle); 4,38 khi RUL ≤ 30"),
        ("Q5", "Cảnh báo, lịch bảo trì", f"{MNT['evaluation']['n_correct_needs_maintenance']}/{MNT['evaluation']['n_flagged_needs_maintenance']}",
         "dự báo cần bảo trì chính xác; không bỏ sót; lead time trung vị 30 cycle"),
    ]
    for i, (q, head, value, desc) in enumerate(cards):
        x, y = 0.92 + (i % 3) * 3.9, 2.5 + (i // 3) * 2.1
        rect(s, x, y, 3.7, 1.95, TINT)
        circle(s, x + 0.2, y + 0.18, 0.46, q, fill=GOLD, color=INK, size=11)
        text(s, x + 0.8, y + 0.2, 2.8, 0.42, head, size=14, bold=True, color=BLUE, anchor=MSO_ANCHOR.MIDDLE)
        text(s, x + 0.2, y + 0.76, 3.3, 0.5, value, size=24, bold=True, color=BLUE)
        text(s, x + 0.2, y + 1.3, 3.35, 0.6, desc, size=11.5)
    rect(s, 8.72, 4.6, 3.7, 1.95, BLUE)
    text(s, 8.92, 4.75, 3.3, 0.36, "Sản phẩm đặc thù", size=14, bold=True, color=GOLD)
    text(s, 8.92, 5.15, 3.3, 1.3, "Lịch dự đoán thời điểm cần bảo trì cho từng động cơ, kèm biểu đồ xu hướng cảnh báo sớm.",
         size=13, color=WHITE)


# ============================================================================ 19. Khó khăn
def slide_lessons():
    s = content(6, 19, "Khó khăn, bài học và giới hạn",
                "Các vấn đề phát sinh trong quá trình thực hiện, cách xử lý và bài học rút ra",
                "report/sections/discussion.tex (Mục Khó khăn và bài học rút ra)",
                "Thời lượng: 1 phút.\nMột: dữ liệu không ghi lúc bắt đầu hỏng, nên so mỗi động cơ với chính nó lúc khỏe và nói "
                "rõ chưa đo được báo động giả.\nHai: dữ liệu chuỗi thời gian dễ rò rỉ khi chia tập; đơn vị chia phải là động "
                "cơ, window và EWMA chỉ dùng quá khứ.\nBa: PySpark trên Windows thiếu winutils nên ghi qua Arrow, và luôn đối "
                "chiếu từng giá trị với Pandas.\nBốn: dễ sa vào tối ưu mô hình; với bảo trì dự đoán, pipeline đúng và cảnh "
                "báo dễ hiểu quan trọng hơn vài cycle MAE.\nGiới hạn: dữ liệu mô phỏng, chưa có chi phí bảo trì, Spark chỉ "
                "chạy local.")
    table(s, [
        ["Khó khăn", "Cách xử lý", "Bài học"],
        ["Thiếu nhãn thời điểm bất thường", "Lấy 30 cycle đầu của từng động cơ làm tham chiếu",
         "Xác định rõ trạng thái tham chiếu; nêu rõ phần chưa kiểm chứng được"],
        ["Rò rỉ dữ liệu khi chia tập", "GroupKFold theo động cơ; window, EWMA chỉ dùng quá khứ",
         "Với dữ liệu chuỗi thời gian, đơn vị phân chia là thực thể, không phải bản ghi"],
        ["Giới hạn ghi file của PySpark trên Windows", "JRE portable, ghi qua Arrow, đối chiếu sau mỗi bước",
         "Hai cách triển khai cần được đối chiếu ở mức từng giá trị"],
        ["Nguy cơ tập trung quá mức vào tối ưu mô hình", "Cố định tiêu chí đánh giá trước; tách thử nghiệm sang phần mở rộng",
         "Tính đúng đắn của pipeline và khả năng diễn giải cảnh báo được ưu tiên hơn mức giảm nhỏ của MAE"],
    ], 0.92, 2.5, [2.7, 4.3, 4.5], row_h=0.66, size=12)
    rect(s, 0.92, 5.95, 11.5, 0.6, GOLD_TINT)
    text(s, 1.15, 5.95, 11.05, 0.6, [[("Giới hạn: ", {"bold": True, "color": BLUE}),
                                      ("dữ liệu mô phỏng; chưa có dữ liệu chi phí để tối ưu ngưỡng cảnh báo; PySpark mới chạy cục bộ, "
                                       "chưa triển khai cluster hay streaming.", {})]], size=12.5, anchor=MSO_ANCHOR.MIDDLE)


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
        "Cảm ơn thầy/cô. Nếu còn thời gian, mở dashboard theo phần demo trong handoff/SPEECH_SCRIPT_VI.pdf (khoảng 3 phút); không chạy lại "
        "huấn luyện khi trình bày."
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
