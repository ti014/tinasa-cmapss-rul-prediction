// Dựng deck thuyết trình 18–20 phút (20 slide) từ slides/slide_data.json và hình trong outputs/.
// Chạy trước: uv run --frozen python slides/extract_slide_data.py
// Sau đó:     node slides/build_presentation.cjs  (cần pptxgenjs và sharp trong NODE_PATH)
const fs = require("node:fs");
const path = require("node:path");
const pptxgen = require("pptxgenjs");
const sharp = require("sharp");

const ROOT = path.resolve(__dirname, "..");
const OUT = process.env.DECK_OUT || path.join(ROOT, "slides", "cmapss_fd001_presentation.pptx");
const D = JSON.parse(fs.readFileSync(path.join(ROOT, "slides", "slide_data.json"), "utf8"));

const C = {
  navy: "0B2239", ink: "1B2733", slate: "5A6B7B", mist: "EEF3F6", line: "C9D4DD",
  teal: "157A7F", tealL: "DCEFEF", amber: "C8701E", amberL: "FBEBD9", red: "B23A3A",
  white: "FFFFFF", ice: "BFD9E3", grey: "9AA8B5",
};
const FONT = "Arial";
const W = 13.333;
const PARTS = [
  "Bài toán và dữ liệu",
  "Pipeline Big Data",
  "Phân tích chuỗi thời gian",
  "Mô hình và đánh giá",
  "Mở rộng và kết luận",
];

// Số kiểu Việt Nam: dấu phẩy thập phân, dấu chấm hàng nghìn.
function vn(x, d = 2) {
  const [i, f] = Number(x).toFixed(d).split(".");
  const int = i.replace(/\B(?=(\d{3})+(?!\d))/g, ".");
  return f ? `${int},${f}` : int;
}
const pct = (x, d = 2) => `${vn(x * 100, d)}%`;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.author = "Phan Nguyễn Mai Phương";
pres.title = "Dự đoán RUL và cảnh báo sớm trên NASA C-MAPSS";

function T(slide, text, o) {
  slide.addText(text, {
    isTextBox: true, margin: 0, fontFace: FONT, fontSize: 14, color: C.ink,
    valign: "top", ...o,
  });
}
function box(slide, x, y, w, h, fill, line) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x, y, w, h, rectRadius: 0.08, fill: { color: fill },
    line: line ? { color: line, width: 0.75 } : { type: "none" },
  });
}
function rect(slide, x, y, w, h, fill, line) {
  slide.addShape(pres.shapes.RECTANGLE, {
    x, y, w, h, fill: { color: fill }, line: line ? { color: line, width: 0.75 } : { type: "none" },
  });
}
function arrow(slide, x1, y1, x2, y2, color = C.slate) {
  const flipH = x2 < x1;
  const flipV = y2 < y1;
  slide.addShape(pres.shapes.LINE, {
    x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1), h: Math.abs(y2 - y1),
    flipH, flipV, line: { color, width: 1.5, endArrowType: "triangle" },
  });
}
function circleNum(slide, x, y, d, label, fill = C.navy, color = C.white, size = 16) {
  slide.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: fill }, line: { type: "none" } });
  T(slide, label, { x, y, w: d, h: d, align: "center", valign: "middle", bold: true, fontSize: size, color });
}

// Khung chung cho slide nội dung: nhãn phần, tiêu đề nêu kết luận, dòng diễn giải, nguồn, số trang.
function content(part, title, takeaway, n, source, notes) {
  const s = pres.addSlide();
  s.background = { color: C.white };
  const tag = `PHẦN ${part} · ${PARTS[part - 1].toUpperCase()}`;
  box(s, 0.6, 0.34, 0.3 + 0.1 * tag.length, 0.32, C.tealL);
  T(s, tag, {
    x: 0.72, y: 0.34, w: 6, h: 0.32, valign: "middle", fontSize: 10.5, bold: true, color: C.teal, charSpacing: 1,
  });
  T(s, title, { x: 0.6, y: 0.78, w: 12.13, h: 0.62, fontSize: 27, bold: true, color: C.navy, valign: "middle" });
  T(s, takeaway, { x: 0.6, y: 1.42, w: 12.13, h: 0.42, fontSize: 15, color: C.slate, valign: "middle" });
  T(s, `Nguồn: ${source}`, { x: 0.6, y: 7.05, w: 11.2, h: 0.26, fontSize: 9, color: C.slate, valign: "middle" });
  T(s, `${n} / 20`, { x: 11.93, y: 7.05, w: 0.8, h: 0.26, fontSize: 10, color: C.slate, align: "right", valign: "middle" });
  s.addNotes(notes);
  return s;
}

function img(slide, rel, x, y, w, h, alt) {
  slide.addImage({ path: path.join(ROOT, rel), x, y, w, h, altText: alt });
}
async function cropImg(slide, rel, region, x, y, w, alt) {
  const buf = await sharp(path.join(ROOT, rel)).extract(region).png().toBuffer();
  const h = (w * region.height) / region.width;
  slide.addImage({ data: "image/png;base64," + buf.toString("base64"), x, y, w, h, altText: alt });
  return h;
}

function table(slide, rows, o) {
  const [head, ...body] = rows;
  const fs0 = o.fontSize || 13;
  const hl = o.highlight ?? -1;
  const data = [
    head.map((t) => ({ text: t, options: { bold: true, color: C.white, fill: { color: C.navy } } })),
    ...body.map((r, i) =>
      r.map((t) => ({
        text: String(t),
        options: {
          color: C.ink, bold: i === hl,
          fill: { color: i === hl ? C.tealL : i % 2 ? C.mist : C.white },
        },
      })),
    ),
  ];
  slide.addTable(data, {
    x: o.x, y: o.y, w: o.w, colW: o.colW, rowH: o.rowH || 0.42, fontFace: FONT, fontSize: fs0,
    valign: "middle", margin: [0.04, 0.08, 0.04, 0.08], border: { type: "solid", color: C.line, pt: 0.75 },
    align: "left",
  });
}

function stat(slide, x, y, w, value, label, color = C.navy, size = 34) {
  T(slide, value, { x, y, w, h: 0.62, fontSize: size, bold: true, color, valign: "bottom" });
  T(slide, label, { x, y: y + 0.66, w, h: 0.5, fontSize: 12, color: C.slate });
}

const M = D.modeling;
const P = D.pipeline;
const I = D.improvement;
const DG = D.diagnostics;
const A = D.alignment;
const inr = (t) => pct(DG.transfer[t].inside_range, 0);
const fd = Object.fromEntries(D.subsets.map((r) => [r.name, r]));
const cvBy = Object.fromEntries(M.cv.map((r) => [r.model_name, r]));
const tr = (s, t) => D.transfer.rows.find((r) => r.source === s && r.target === t);
const fixedWarning = Object.fromEntries(A.warning_policies.map((r) => [`${r.model}_${r.threshold}`, r]));

async function main() {
  // ---------------------------------------------------------------- 1. Bìa
  {
    const s = pres.addSlide();
    s.background = { color: C.navy };
    T(s, "BÁO CÁO CUỐI KỲ · PHÂN TÍCH DỮ LIỆU LỚN", { x: 0.8, y: 0.8, w: 10, h: 0.35, fontSize: 13, bold: true, color: C.ice, charSpacing: 2 });
    T(s, "Dự đoán tuổi thọ còn lại và cảnh báo sớm cho động cơ turbofan", { x: 0.8, y: 1.45, w: 11.2, h: 1.6, fontSize: 40, bold: true, color: C.white });
    T(s, "Pipeline dữ liệu chuỗi thời gian trên NASA C-MAPSS FD001–FD004", { x: 0.8, y: 3.15, w: 11, h: 0.5, fontSize: 20, color: C.ice });
    const items = [
      [String(fd.FD001.train_engines), "train engines FD001"],
      ["21", "sensor mỗi cycle"],
      ["169", "features HGB169 · FD001"],
      [vn(A.trajectory.hgb169.mean_engine_mae), "MAE trajectory candidate"],
    ];
    items.forEach(([v, l], i) => {
      const x = 0.8 + i * 2.9;
      T(s, v, { x, y: 4.25, w: 2.6, h: 0.6, fontSize: 30, bold: true, color: C.white });
      T(s, l, { x, y: 4.88, w: 2.6, h: 0.35, fontSize: 13, color: C.ice });
    });
    T(s, "Phan Nguyễn Mai Phương · MSHV 25002711", { x: 0.8, y: 6.05, w: 8, h: 0.38, fontSize: 17, bold: true, color: C.white });
    T(s, "Học viên cao học ngành Khoa học máy tính · Tháng 10/2026", { x: 0.8, y: 6.45, w: 8, h: 0.35, fontSize: 13, color: C.ice });
    s.addNotes(
      "Thời lượng: 20 giây.\n" +
        "Chào thầy/cô. Giới thiệu tên, môn học. Nêu một câu mục tiêu: xây dựng pipeline từ raw data đến dự đoán RUL và cảnh báo sớm, có thể kiểm tra và tái lập.\n" +
        "Bốn con số ở dưới là khung của bài: 100 train engines, 21 sensor, 169 đặc trưng của predictor candidate và MAE trajectory 17,0009 cycles trên panel OOF chung. Đây là kết quả alignment audit có điều kiện, không phải metric test độc lập sau refit.\n" +
        "Metric test lịch sử của HGB126 được giữ riêng ở slide 14. Nguồn: outputs/course_alignment/20261007/summary.json; outputs/course_alignment/20261007/model_metrics.csv.",
    );
  }

  // ---------------------------------------------------------------- 2. Lộ trình
  {
    const s = content(1, "Nội dung trình bày và sáu câu hỏi nghiên cứu", "18–20 phút gồm demo dashboard; kết luận ở slide 19 trả lời lần lượt Q1–Q6", 2,
      "report/main.pdf; handoff/SPEECH_SCRIPT_VI.pdf",
      "Thời lượng: 25 giây.\nĐọc nhanh 5 phần và thời lượng. Nhấn mạnh sáu câu hỏi nghiên cứu; slide 19 sẽ trả lời lại từng câu bằng bằng chứng tương ứng. Demo dashboard dự kiến 3 phút ở cuối, chỉ trình diễn artifact đã tạo.\nQ1 kiểm tra dữ liệu; Q2 xu hướng sensor; Q3 parity ETL; Q4 so sánh HGB126/HGB169; Q5 cảnh báo; Q6 chuyển miền.");
    const rows = [
      ["Bài toán và dữ liệu", "Slide 2–5", "3 phút"],
      ["Pipeline Big Data", "Slide 6–7", "3 phút"],
      ["Phân tích chuỗi thời gian", "Slide 8–11", "3 phút"],
      ["Mô hình và đánh giá", "Slide 12–16", "5 phút"],
      ["Mở rộng và kết luận", "Slide 17–20", "3 phút"],
      ["Demo dashboard", "Artifact đã tạo", "3 phút"],
    ];
    rows.forEach(([name, sl, t], i) => {
      const y = 2.12 + i * 0.68;
      circleNum(s, 0.6, y, 0.48, String(i + 1), i === 5 ? C.amber : C.navy, C.white, 12);
      T(s, name, { x: 1.25, y: y - 0.01, w: 4.8, h: 0.28, fontSize: 15.5, bold: true, color: C.ink });
      T(s, `${sl} · ${t}`, { x: 1.25, y: y + 0.29, w: 4.8, h: 0.23, fontSize: 11.5, color: C.slate });
    });
    box(s, 6.75, 2.1, 5.98, 4.65, C.mist);
    T(s, "Câu hỏi nghiên cứu", { x: 7.1, y: 2.35, w: 5.3, h: 0.4, fontSize: 18, bold: true, color: C.navy });
    const qs = [
      ["Q1", "FD001 có hợp lệ về schema, khóa và tính liên tục của cycle không?"],
      ["Q2", "Sensor thay đổi thế nào giữa đầu và cuối vòng đời engine?"],
      ["Q3", "ETL bằng Pandas và PySpark có cho kết quả giống nhau không?"],
      ["Q4", "HGB126 hay HGB169 ổn định hơn trên cùng panel và các mốc vòng đời?"],
      ["Q5", "RUL có chuyển thành cảnh báo sớm với chất lượng chấp nhận được không?"],
      ["Q6", "Model huấn luyện trên FD001 hoạt động thế nào khi áp dụng FD002–FD004?"],
    ];
    qs.forEach(([q, t], i) => {
      const y = 2.86 + i * 0.61;
      circleNum(s, 7.1, y, 0.42, q, C.amber, C.white, 10.5);
      T(s, t, { x: 7.72, y: y - 0.01, w: 4.62, h: 0.5, fontSize: 12.5, color: C.ink, valign: "middle" });
    });
  }

  // ---------------------------------------------------------------- 3. Bài toán
  {
    const s = content(1, "Bài toán: ước lượng số cycle còn lại trước failure",
      "Đề tài 8 yêu cầu phát hiện dấu hiệu suy giảm và dự đoán thời điểm cần bảo trì từ dữ liệu cảm biến", 3,
      "references/context/assignment-brief.png; report/sections/problem-data.tex; data/CONTRACTS.md",
      "Thời lượng: 35 giây.\n" +
        "Remaining Useful Life (RUL, tuổi thọ hữu ích còn lại) là số cycle vận hành còn lại trước failure. Đây là supervised regression.\n" +
        "Ví dụ: engine kết thúc ở cycle 200, tại cycle 170 thì RUL = 30 cycles. Đơn vị là cycle; dữ liệu không có thông tin quy đổi ra giờ hoặc ngày.\n" +
        "Bảng bên phải đối chiếu từng yêu cầu kỹ thuật của đề tài với phần đã làm và slide trình bày. Nói rõ: xử lý nhiễu được thực hiện qua thống kê trên window và Health Index; repository chưa dùng bộ lọc tín hiệu riêng.");
    T(s, "Ví dụ trên một train engine", { x: 0.6, y: 2.15, w: 5.2, h: 0.32, fontSize: 15, bold: true, color: C.navy });
    const x0 = 0.75, x1 = 5.55, y = 3.05, tx = x0 + (x1 - x0) * (170 / 200);
    rect(s, x0, y, x1 - x0, 0.2, C.tealL);
    rect(s, tx, y, x1 - tx, 0.2, C.amber);
    s.addShape(pres.shapes.OVAL, { x: x1 - 0.1, y: y - 0.05, w: 0.3, h: 0.3, fill: { color: C.red }, line: { type: "none" } });
    T(s, "cycle 1", { x: x0 - 0.1, y: y + 0.32, w: 1, h: 0.28, fontSize: 11, color: C.slate });
    T(s, "t = 170", { x: tx - 0.5, y: y - 0.42, w: 1, h: 0.3, fontSize: 12, bold: true, color: C.amber, align: "center" });
    T(s, "failure\ncycle 200", { x: x1 - 0.55, y: y + 0.32, w: 1.1, h: 0.5, fontSize: 11, color: C.red, align: "center" });
    T(s, "RUL(t) = cycle cuối − t", { x: 0.6, y: 4.1, w: 5.4, h: 0.45, fontSize: 22, bold: true, color: C.navy });
    T(s, "RUL(170) = 200 − 170 = 30 cycles", { x: 0.6, y: 4.6, w: 5.4, h: 0.4, fontSize: 16, color: C.amber, bold: true });
    T(s, "Supervised regression: input là chuỗi 21 sensor theo cycle, output là một số RUL. Đơn vị là cycle, không quy đổi ra giờ.",
      { x: 0.6, y: 5.25, w: 5.3, h: 1.2, fontSize: 14, color: C.ink });
    table(s, [
      ["Yêu cầu của đề tài", "Cách đáp ứng trong báo cáo", "Slide"],
      ["Xử lý chuỗi thời gian cảm biến", "Kiểm tra schema, khóa, cycle liên tục; window theo từng engine", "6, 10"],
      ["Xử lý nhiễu tín hiệu", "Thống kê trên window 30 và Health Index PCA; chưa dùng bộ lọc riêng", "10, 11"],
      ["Chuẩn hóa, tổng hợp theo window", "Chuẩn hóa schema; StandardScaler trong fold; 126 đặc trưng", "10"],
      ["Trực quan xu hướng suy giảm", "Quỹ đạo sensor, kiểm định đầu/cuối vòng đời, Health Index", "9, 11"],
      ["Dự đoán thời điểm bảo trì, cảnh báo sớm", "Dự đoán RUL; cảnh báo khi RUL dự đoán ≤ 30 cycles", "14, 16"],
    ], { x: 6.35, y: 2.15, w: 6.38, colW: [2.2, 3.43, 0.75], rowH: 0.7, fontSize: 12 });
  }

  // ---------------------------------------------------------------- 4. Dữ liệu
  {
    const s = content(1, "NASA C-MAPSS: bốn subset, hai trục khác biệt",
      "Hai trục là số điều kiện vận hành và số fault mode; FD001 là protocol chính, FD002–FD004 dùng ở slide 18", 4,
      "data/raw/readme.txt; outputs/cross_dataset/FD00x/etl/run_info.json; data/CONTRACTS.md",
      "Thời lượng: 50 giây.\n" +
        "C-MAPSS là dữ liệu mô phỏng động cơ turbofan của NASA. Mỗi subset có train (chạy đến failure), test (bị cắt trước failure) và file RUL_FD00x.txt chứa nhãn test.\n" +
        "FD001: 1 điều kiện vận hành, 1 fault mode (HPC degradation), 100 train và 100 test engines, 33.727 observations.\n" +
        "FD002 thêm 6 điều kiện vận hành; FD003 thêm fault mode Fan; FD004 có cả hai.\n" +
        "Mỗi dòng có 26 trường: engine_id, cycle, 3 operational settings, 21 sensor. Khóa trong một subset là (split, engine_id, cycle).");
    const meta = {
      FD001: ["1 điều kiện", "1 fault mode (HPC)"], FD002: ["6 điều kiện", "1 fault mode (HPC)"],
      FD003: ["1 điều kiện", "2 fault mode (HPC, Fan)"], FD004: ["6 điều kiện", "2 fault mode (HPC, Fan)"],
    };
    D.subsets.forEach((r, i) => {
      const x = 0.6 + i * 3.08, main = i === 0;
      const fg = main ? C.white : C.ink, sub = main ? C.ice : C.slate;
      box(s, x, 2.1, 2.9, 2.75, main ? C.navy : C.mist);
      T(s, r.name, { x: x + 0.25, y: 2.25, w: 2.4, h: 0.5, fontSize: 24, bold: true, color: main ? C.white : C.navy });
      T(s, main ? "Protocol chính" : "Kiểm tra chuyển miền", { x: x + 0.25, y: 2.75, w: 2.4, h: 0.3, fontSize: 11.5, bold: true, color: main ? C.ice : C.teal });
      T(s, `${meta[r.name][0]} vận hành\n${meta[r.name][1]}`, { x: x + 0.25, y: 3.15, w: 2.5, h: 0.62, fontSize: 13, color: fg });
      T(s, `${r.train_engines} train / ${r.test_engines} test engines\n${vn(r.observations, 0)} observations`, { x: x + 0.25, y: 3.9, w: 2.5, h: 0.62, fontSize: 13, color: sub });
    });
    T(s, "Một dòng observation có 26 trường", { x: 0.6, y: 5.12, w: 8, h: 0.34, fontSize: 15, bold: true, color: C.navy });
    const chips = [["engine_id", 1.35, C.navy], ["cycle", 1.0, C.navy], ["setting_1 … setting_3", 2.55, C.teal], ["sensor_1 … sensor_21", 7.0, C.amber]];
    let cx = 0.6;
    chips.forEach(([t, w, col]) => {
      box(s, cx, 5.55, w, 0.5, col);
      T(s, t, { x: cx, y: 5.55, w, h: 0.5, align: "center", valign: "middle", fontSize: 13, bold: true, color: C.white });
      cx += w + 0.08;
    });
    T(s, "Khóa trong một subset: (split, engine_id, cycle). Bốn subset được xử lý riêng vì khóa chưa có dataset_id.",
      { x: 0.6, y: 6.25, w: 12.1, h: 0.4, fontSize: 13, color: C.slate });
  }

  // ---------------------------------------------------------------- 5. Nhãn và ranh giới
  {
    const s = content(1, "Nhãn RUL tạo từ train; nhãn test chỉ mở sau khi đã dự đoán",
      "Ranh giới này ngăn data leakage (rò rỉ dữ liệu) từ test vào EDA, đặc trưng và chọn mô hình", 5,
      "data/CONTRACTS.md; outputs/etl/parity.json; outputs/modeling/run_info.json; outputs/modeling/tables/final_metrics.json",
      "Thời lượng: 55 giây.\n" +
        "Train: mỗi engine chạy đến failure nên nhãn suy ra trực tiếp: RUL = cycle cuối − cycle hiện tại. FD001 có 20.631 nhãn train.\n" +
        "Test: chuỗi bị cắt trước failure. Không được lấy cycle cuối quan sát làm failure. Model tạo đúng một dự đoán tại cycle cuối quan sát của mỗi test engine; sau khi ghi final predictions mới ghép RUL_FD001.txt để chấm điểm. run_info ghi test_label_join.status = completed_after_final_predictions.\n" +
        "Bên phải: ba kiểm soát cho nhãn test. final_metrics.json ghi test_label_sha256 để đối chiếu file nhãn đã dùng.");
    const lane = (y, label, sub, full) => {
      T(s, label, { x: 0.6, y, w: 1.6, h: 0.34, fontSize: 17, bold: true, color: C.navy });
      T(s, sub, { x: 0.6, y: y + 0.36, w: 1.7, h: 0.6, fontSize: 11.5, color: C.slate });
      const x0 = 2.4, x1 = full ? 8.1 : 6.3;
      rect(s, x0, y + 0.12, x1 - x0, 0.26, C.tealL);
      if (full) {
        s.addShape(pres.shapes.OVAL, { x: x1 - 0.15, y: y + 0.07, w: 0.36, h: 0.36, fill: { color: C.red }, line: { type: "none" } });
        T(s, "failure", { x: x1 - 0.5, y: y + 0.5, w: 1.0, h: 0.28, fontSize: 11.5, color: C.red, align: "center" });
      } else {
        s.addShape(pres.shapes.LINE, { x: x1, y: y + 0.25, w: 1.8, h: 0, line: { color: C.grey, width: 1.5, dashType: "dash" } });
        s.addShape(pres.shapes.OVAL, { x: x1 - 0.13, y: y + 0.1, w: 0.3, h: 0.3, fill: { color: C.amber }, line: { type: "none" } });
        T(s, "cycle cuối quan sát", { x: x1 - 1.6, y: y + 0.5, w: 1.8, h: 0.28, fontSize: 11.5, color: C.amber, align: "right" });
        T(s, "failure chưa biết", { x: x1 + 0.3, y: y + 0.5, w: 1.6, h: 0.28, fontSize: 11.5, color: C.grey, align: "right" });
      }
    };
    lane(2.2, "Train", `${fd.FD001.train_engines} engines chạy đến failure`, true);
    lane(3.65, "Test", `${fd.FD001.test_engines} engines bị cắt trước failure`, false);
    box(s, 0.6, 5.0, 7.9, 1.5, C.mist);
    T(s, [
      { text: "Train: ", options: { bold: true, color: C.navy } },
      { text: `RUL(t) = cycle cuối − t → ${vn(P.parity[1].rows, 0)} nhãn`, options: { breakLine: true } },
      { text: "Test: ", options: { bold: true, color: C.navy } },
      { text: "1 dự đoán/engine tại cycle cuối → ghi final predictions → mới ghép RUL_FD001.txt", options: {} },
    ], { x: 0.85, y: 5.18, w: 7.5, h: 1.45, fontSize: 14.5, paraSpaceAfter: 8 });
    box(s, 9.0, 2.1, 3.73, 4.65, C.navy);
    T(s, "Kiểm soát nhãn test", { x: 9.3, y: 2.3, w: 3.2, h: 0.35, fontSize: 16, bold: true, color: C.white });
    [
      ["Không dùng", "trong EDA, đặc trưng, chọn model"],
      ["Ghép muộn", "sau khi đã ghi final predictions"],
      ["Có dấu vết", "run_info ghi trạng thái join; final_metrics ghi SHA-256 của nhãn"],
    ].forEach(([a, b], i) => {
      const yy = 2.9 + i * 1.25;
      T(s, a, { x: 9.3, y: yy, w: 3.2, h: 0.4, fontSize: 19, bold: true, color: "F2B880" });
      T(s, b, { x: 9.3, y: yy + 0.42, w: 3.2, h: 0.7, fontSize: 12.5, color: C.ice });
    });
  }

  // ---------------------------------------------------------------- 6. Kiến trúc pipeline
  {
    const s = content(2, "Pipeline Bronze → Silver → Gold → đặc trưng → mô hình",
      "Mỗi bước ghi artifact kèm hash và run_info để truy vết và chạy lại", 6,
      "assets/diagrams/fd001-pipeline.drawio; handoff/DATA_WALKTHROUGH.md; outputs/*/run_info.json",
      "Thời lượng: 55 giây.\n" +
        "Đọc sơ đồ từ trái sang phải. Bronze là file raw. Reader kiểm tra đúng 26 trường, kiểu dữ liệu, khóa duy nhất và cycle liên tục; train và test được union theo dòng (Pandas concat hoặc PySpark unionByName), không phải join.\n" +
        "Silver lưu observations và train_targets dạng Parquet. Gold lưu engine_summary dạng Parquet và SQLite. Từ Silver, bước đặc trưng tạo trailing window 30 cycles.\n" +
        "Modeling dùng GroupKFold, refit và tạo 100 dự đoán test. Hai nhánh phụ EDA/thống kê và Health Index chỉ dùng train.\n" +
        "Phần Big Data nằm ở batch ETL nhiều file, Parquet Silver/Gold và parity Pandas–PySpark. Modeling và rolling features dùng Pandas/scikit-learn.");
    const steps = [
      ["Bronze", "Raw TXT", "train, test,\nRUL_FD00x"],
      ["Validation", "Reader", "26 trường, kiểu,\nkhóa, cycle liên tục"],
      ["Silver", "Parquet", `observations ${vn(fd.FD001.observations, 0)}\ntrain_targets ${vn(P.parity[1].rows, 0)}`],
      ["Gold", "Parquet + SQLite", `engine_summary\n${P.parity[2].rows} engines`],
      ["Đặc trưng", "Window 30", `${vn(P.feature_rows.rolling_features, 0)} dòng\n× ${P.feature_columns} cột`],
      ["Mô hình", "HGB", "GroupKFold → refit\n→ 100 dự đoán test"],
    ];
    const bw = 1.78, gap = 0.29, y = 2.55, bh = 1.75;
    rect(s, 0.6 + 2 * (bw + gap) - 0.1, 2.02, 2 * bw + gap + 0.2, 0.36, C.amberL);
    T(s, "Pandas và PySpark: kết quả khớp", { x: 0.6 + 2 * (bw + gap) - 0.1, y: 2.02, w: 2 * bw + gap + 0.2, h: 0.36, fontSize: 12, bold: true, color: C.amber, align: "center", valign: "middle" });
    steps.forEach(([a, b, c], i) => {
      const x = 0.6 + i * (bw + gap);
      const fill = i === 5 ? C.navy : i === 4 ? C.teal : C.mist;
      const dark = i >= 4;
      box(s, x, y, bw, bh, fill);
      T(s, a, { x: x + 0.12, y: y + 0.12, w: bw - 0.24, h: 0.34, fontSize: 15, bold: true, color: dark ? C.white : C.navy });
      T(s, b, { x: x + 0.12, y: y + 0.47, w: bw - 0.24, h: 0.3, fontSize: 12, bold: true, color: dark ? C.ice : C.teal });
      T(s, c, { x: x + 0.12, y: y + 0.85, w: bw - 0.2, h: 0.8, fontSize: 11.5, color: dark ? C.white : C.ink });
      if (i < 5) arrow(s, x + bw + 0.03, y + bh / 2, x + bw + gap - 0.03, y + bh / 2);
    });
    const branch = [
      [2, "EDA, kiểm định, Health Index–ARIMA", "chỉ dùng train · slide 8–11", 3.9],
      [5, "Metric, hình, dashboard", "slide 14–16", 2.6],
    ];
    branch.forEach(([col, a, b, bwid]) => {
      const cxm = 0.6 + col * (bw + gap) + bw / 2;
      const x = Math.min(Math.max(cxm - bwid / 2, 0.6), 12.73 - bwid);
      arrow(s, cxm, y + bh + 0.05, cxm, 5.02);
      box(s, x, 5.05, bwid, 0.9, C.white, C.line);
      T(s, a, { x: x + 0.15, y: 5.13, w: bwid - 0.3, h: 0.4, fontSize: 13, bold: true, color: C.navy });
      T(s, b, { x: x + 0.15, y: 5.53, w: bwid - 0.3, h: 0.3, fontSize: 11.5, color: C.slate });
    });
    T(s, "Khóa observation: (split, engine_id, cycle). Feature và target được join bằng đủ ba cột khóa và kiểm tra quan hệ 1:1.",
      { x: 0.6, y: 6.3, w: 12.1, h: 0.45, fontSize: 13, color: C.slate });
  }

  // ---------------------------------------------------------------- 7. Big Data evidence
  {
    const b = D.stage_benchmark;
    const s = content(2, "Pandas và PySpark cho cùng kết quả tại ba boundary",
      "Đã bổ sung parity 126 rolling features và benchmark local 1×/10×: đọc, tổng hợp, tạo feature, ghi file", 7,
      "outputs/etl/parity.json; outputs/benchmarks/20260929/{measurements.csv,run_info.json}",
      "Thời lượng: 1 phút 5 giây.\n" +
        "Bảng trái: cùng input, Pandas pipeline và PySpark pipeline cho cùng schema, khóa, số dòng và giá trị ở ba boundary; sai khác tuyệt đối lớn nhất bằng 0.\n" +
        "Biểu đồ phải: trung vị ba lượt đo trong process riêng, cùng input Parquet, 33.727 và 337.270 dòng. Pipeline gồm đọc, engine summary, rolling features, ghi file; startup Spark báo riêng. Peak RSS gồm Python và JVM.\n" +
        "Spark local[4], bốn partition, driver/executor heap 4 GiB. Lượt heap mặc định bị OOM ở 10x được lưu riêng, không đưa vào median. Ghi Parquet qua PyArrow trên driver; thời gian Spark gồm cả toArrow. Chưa đo distributed writer hoặc cluster. Không xóa OS cache giữa lượt đo.");
    table(s, [
      ["Boundary", "Số dòng", "Sai khác lớn nhất", "Kết quả"],
      ...P.parity.map((p) => [p.name, vn(p.rows, 0), vn(p.max_abs_diff, 0), p.status === "passed" ? "Khớp" : p.status]),
    ], { x: 0.6, y: 2.2, w: 6.2, colW: [2.35, 1.2, 1.55, 1.1], rowH: 0.55, fontSize: 13 });
    box(s, 0.6, 4.65, 6.2, 1.75, C.mist);
    T(s, "Kiểm tra cùng một input bằng hai engine xử lý", { x: 0.85, y: 4.8, w: 5.8, h: 0.34, fontSize: 14, bold: true, color: C.navy });
    T(s, "Rolling features: so toàn bộ 27.927 dòng × 126 features, tolerance 10⁻⁷. Cùng window 30, std ddof=1 và slope nhân quả theo từng engine.",
      { x: 0.85, y: 5.2, w: 5.75, h: 1.4, fontSize: 13, color: C.ink });
    s.addChart(pres.charts.BAR, ["pandas", "spark"].map((engine) => ({
      name: engine === "pandas" ? "Pandas" : "PySpark local[4]",
      labels: ["1× · 33.727 dòng", "10× · 337.270 dòng"],
      values: [1, 10].map((scale) => Math.round(b.find((r) => r.engine === engine && r.scale === scale).pipeline_seconds)),
    })), {
      x: 7.2, y: 2.1, w: 5.53, h: 3.9, barDir: "col", barGrouping: "clustered", chartColors: [C.teal, C.amber],
      showTitle: true, title: "Pipeline gồm I/O · trung vị 3 lượt (giây)", titleFontSize: 12, titleColor: C.ink, titleFontFace: FONT,
      showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 13, dataLabelColor: C.ink, dataLabelFontFace: FONT, dataLabelFormatCode: "0",
      catAxisLabelFontSize: 11, catAxisLabelColor: C.slate, valAxisLabelFontSize: 10, valAxisLabelColor: C.slate,
      valGridLine: { color: "E3E8EC", size: 0.5 }, catGridLine: { style: "none" }, showLegend: true, legendPos: "b", legendFontSize: 11, catAxisLabelFontFace: FONT, valAxisLabelFontFace: FONT,
    });
    T(s, "Chưa gồm startup Spark. Ghi qua Arrow trên driver; chưa suy rộng hiệu năng sang cluster.",
      { x: 7.2, y: 6.1, w: 5.53, h: 0.6, fontSize: 13, bold: true, color: C.navy });
  }

  // ---------------------------------------------------------------- 8. EDA
  {
    const E = D.eda;
    const s = content(3, "EDA: tuổi thọ phân tán rộng và 7 cột không đổi",
      "EDA chỉ dùng train để hiểu dữ liệu trước khi xây dựng đặc trưng", 8,
      "outputs/eda/figures/engine_lifetimes.png; outputs/eda/tables/constant_engine_counts.csv; lifetime_summary.csv",
      "Thời lượng: 30 giây.\n" +
        "Histogram và boxplot: tuổi thọ train từ 128 đến 362 cycles, trung vị 199, có vài engine dài bất thường ở đuôi phải.\n" +
        "Bảy cột không đổi trong cả 100 engine: sensor 1, 5, 10, 16, 18, 19 và setting_3. sensor_6 không đổi trong 38/100 engine.\n" +
        "Repository chưa làm feature selection dựa trên EDA; các cột này vẫn nằm trong 126 đặc trưng. Đây là quyết định có chủ đích để không dùng kết quả EDA làm bước chọn biến ngoài training fold.");
    img(s, "outputs/eda/figures/engine_lifetimes.png", 0.6, 2.15, 7.6, 7.6 * 617 / 1817, "Histogram và boxplot tuổi thọ 100 train engines");
    const L = E.lifetime;
    [["Ngắn nhất", L.min], ["Trung vị", L["50%"]], ["Trung bình", L.mean], ["Dài nhất", L.max]].forEach(([l, v], i) => {
      stat(s, 0.6 + i * 1.95, 5.05, 1.9, vn(v, v % 1 ? 2 : 0), `${l} (cycles)`, C.navy, 26);
    });
    box(s, 8.6, 2.1, 4.13, 4.65, C.mist);
    T(s, "Không đổi trong cả 100 engine", { x: 8.85, y: 2.3, w: 3.7, h: 0.35, fontSize: 15, bold: true, color: C.navy });
    E.always_constant.forEach((m, i) => {
      const x = 8.85 + (i % 2) * 1.85, y = 2.8 + Math.floor(i / 2) * 0.55;
      box(s, x, y, 1.72, 0.44, m.startsWith("setting") ? C.teal : C.slate);
      T(s, m, { x, y, w: 1.72, h: 0.44, align: "center", valign: "middle", fontSize: 12.5, bold: true, color: C.white });
    });
    const p6 = E.partially_constant[0];
    T(s, `${p6.measurement} không đổi trong ${p6.constant_engines}/100 engine.`, { x: 8.85, y: 5.1, w: 3.7, h: 0.4, fontSize: 13, color: C.ink });
    T(s, "Chưa loại cột nào: 126 đặc trưng vẫn gồm đủ 21 sensor; không làm feature selection từ EDA.", { x: 8.85, y: 5.6, w: 3.7, h: 0.95, fontSize: 12.5, color: C.slate });
  }

  // ---------------------------------------------------------------- 9. Xu hướng suy giảm
  {
    const st = D.statistics;
    const s = content(3, "Nhiều sensor thay đổi rõ khi engine tiến gần failure",
      `So sánh 30 cycle đầu và 30 cycle cuối trên cùng engine: ${st.n_reject}/${st.n_sensors} sensor có thay đổi sau hiệu chỉnh FDR`, 9,
      "outputs/eda/figures/sensor_trajectories_1.png; outputs/statistics/tables/sensor_inference.csv; report/sections/methodology.tex",
      "Thời lượng: 40 giây.\n" +
        "Hai hình là quỹ đạo gốc của ba train engine có tuổi thọ 178, 199 và 229 cycles. sensor_4 tăng dần, sensor_7 giảm dần khi tiến gần failure; tín hiệu có nhiễu theo từng cycle.\n" +
        "Kiểm định: với mỗi train engine, lấy mean 30 cycle cuối trừ mean 30 cycle đầu. Bootstrap theo engine 10.000 lần cho khoảng tin cậy, Wilcoxon signed-rank cho p-value, Benjamini–Hochberg với q = 0,05 cho 21 sensor.\n" +
        "Kết quả: 15 sensor bác bỏ giả thuyết không; 6 sensor hằng số nên không kiểm định.\n" +
        "Giới hạn: đây là mô tả khác biệt đầu/cuối vòng đời, không chứng minh nhân quả và không phải feature selection.");
    const src = "outputs/eda/figures/sensor_trajectories_1.png";
    const h1 = await cropImg(s, src, { left: 905, top: 492, width: 912, height: 438 }, 0.6, 2.0, 4.45, "Quỹ đạo sensor_4 của ba train engine");
    await cropImg(s, src, { left: 0, top: 1372, width: 905, height: 445 }, 0.6, 2.0 + h1 + 0.08, 4.45, "Quỹ đạo sensor_7 của ba train engine");
    T(s, "Engine 87 (178 cycles), 26 (199), 75 (229); giá trị gốc chưa chuẩn hóa.", { x: 0.6, y: 6.66, w: 4.6, h: 0.3, fontSize: 10, color: C.slate });
    const steps = [
      ["Ghép cặp trong engine", "Δ = mean(30 cycle cuối) − mean(30 cycle đầu)"],
      ["Bootstrap và Wilcoxon", `Lấy mẫu lại engine ${vn(st.resamples, 0)} lần; Wilcoxon signed-rank`],
      ["Hiệu chỉnh đa kiểm định", "Benjamini–Hochberg cho 21 sensor, q = 0,05"],
    ];
    steps.forEach(([a, b], i) => {
      const y = 2.15 + i * 0.95;
      circleNum(s, 5.6, y, 0.5, String(i + 1), C.teal, C.white, 14);
      T(s, a, { x: 6.3, y: y - 0.03, w: 6.4, h: 0.32, fontSize: 15, bold: true, color: C.navy });
      T(s, b, { x: 6.3, y: y + 0.3, w: 6.4, h: 0.4, fontSize: 13.5, color: C.ink });
    });
    box(s, 5.6, 5.0, 3.3, 1.7, C.navy);
    T(s, `${st.n_reject}/${st.n_sensors}`, { x: 5.85, y: 5.1, w: 2.9, h: 0.75, fontSize: 40, bold: true, color: C.white });
    T(s, "sensor bác bỏ giả thuyết không sau hiệu chỉnh", { x: 5.85, y: 5.88, w: 2.9, h: 0.65, fontSize: 12.5, color: C.ice });
    box(s, 9.1, 5.0, 3.63, 1.7, C.mist);
    T(s, `${st.n_degenerate} sensor hằng số nên không kiểm định. Kết quả mô tả khác biệt đầu/cuối vòng đời, không chứng minh nhân quả.`, { x: 9.3, y: 5.15, w: 3.25, h: 1.45, fontSize: 13, color: C.ink });
  }

  // ---------------------------------------------------------------- 10. Window features
  {
    const fr = P.feature_rows;
    const s = content(3, `Trailing window ${P.window} cycle biến chuỗi sensor thành ${P.feature_columns} đặc trưng`,
      "Mỗi dòng đặc trưng chỉ dùng quá khứ của chính engine đó; không dùng cycle tương lai hoặc engine khác", 10,
      "outputs/features/run_info.json; src/cmapss/features/rolling.py; handoff/DATA_WALKTHROUGH.md",
      "Thời lượng: 45 giây.\n" +
        "Ví dụ tại cycle 170: window gồm cycles 141–170. Cycle 171 trở đi không được dùng.\n" +
        "Với mỗi sensor tính 6 thống kê: last, mean, sample std, min, max, slope tuyến tính. 21 × 6 = 126 cột.\n" +
        "29 cycle đầu mỗi engine chưa đủ window nên không tạo dòng; không padding. FD001 có 17.731 train windows và 10.196 test windows.\n" +
        "Về yêu cầu xử lý nhiễu và chuẩn hóa: mean, min, max, slope trên 30 cycle tóm tắt dao động từng cycle; repository chưa dùng bộ lọc tín hiệu riêng. ETL chuẩn hóa schema và kiểu dữ liệu. StandardScaler fit trong training fold cho Ridge và trên 90 engine cho Health Index; HGB dùng thang đo gốc.");
    const n = 40, start = 136, cw = 0.2, x0 = 0.6, y0 = 2.45;
    for (let k = 0; k < n; k++) {
      const c = start + k;
      const fill = c === 170 ? C.amber : c >= 141 && c <= 170 ? C.teal : c > 170 ? "E6EAEE" : C.tealL;
      rect(s, x0 + k * cw, y0, cw - 0.03, 0.42, fill);
    }
    T(s, "t = 170", { x: x0 + 34.5 * cw - 0.45, y: y0 - 0.34, w: 0.9, h: 0.3, fontSize: 12, bold: true, color: C.amber, align: "center" });
    T(s, "window: cycles 141–170", { x: x0 + 5 * cw, y: y0 + 0.5, w: 30 * cw, h: 0.3, fontSize: 12.5, bold: true, color: C.teal, align: "center" });
    T(s, "171+: không dùng", { x: x0 + 35 * cw - 0.6, y: y0 + 0.5, w: 5 * cw + 0.6, h: 0.3, fontSize: 11.5, color: C.slate, align: "right" });
    T(s, "21 sensor × 6 thống kê = 126 cột", { x: 0.6, y: 3.6, w: 8, h: 0.4, fontSize: 18, bold: true, color: C.navy });
    const stats = [["last", "trạng thái hiện tại"], ["mean", "mức trung bình"], ["std", "độ dao động"], ["min", "cận dưới"], ["max", "cận trên"], ["slope", "xu hướng tăng/giảm"]];
    stats.forEach(([a, b], i) => {
      const x = 0.6 + (i % 3) * 2.65, y = 4.15 + Math.floor(i / 3) * 0.95;
      box(s, x, y, 2.5, 0.8, C.mist);
      T(s, a, { x: x + 0.15, y: y + 0.08, w: 2.2, h: 0.34, fontSize: 15, bold: true, color: C.teal });
      T(s, b, { x: x + 0.15, y: y + 0.43, w: 2.2, h: 0.3, fontSize: 12, color: C.ink });
    });
    T(s, `${vn(fr.rolling_features_train, 0)} train windows · ${vn(fr.rolling_features_test, 0)} test windows · 29 cycle đầu mỗi engine chưa đủ window, không padding`,
      { x: 0.6, y: 6.2, w: 8, h: 0.5, fontSize: 12.5, color: C.slate });
    box(s, 8.9, 2.1, 3.83, 4.65, C.navy);
    T(s, "Nhiễu và chuẩn hóa", { x: 9.15, y: 2.3, w: 3.4, h: 0.36, fontSize: 16, bold: true, color: C.white });
    T(s, [
      { text: "Thống kê trên 30 cycle tóm tắt dao động từng cycle; chưa dùng bộ lọc tín hiệu riêng.", options: { bullet: true, breakLine: true } },
      { text: "ETL chuẩn hóa schema và kiểu dữ liệu.", options: { bullet: true, breakLine: true } },
      { text: "StandardScaler fit trong training fold (Ridge) và trên 90 engine (Health Index).", options: { bullet: true, breakLine: true } },
      { text: "HGB dùng thang đo gốc.", options: { bullet: true } },
    ], { x: 9.15, y: 2.8, w: 3.4, h: 3.8, fontSize: 13, color: C.white, paraSpaceAfter: 8 });
  }

  // ---------------------------------------------------------------- 11. Health Index
  {
    const ts = D.timeseries;
    const s = content(3, "Health Index gom 21 sensor thành một đường suy giảm",
      "PCA fit trên 90 train engines; ARIMA dự báo 30% cuối chuỗi của 10 engine giữ riêng", 11,
      "outputs/timeseries/figures/health_index_selected_engines.png; tables/forecast_metrics.csv; tables/baseline_metrics.csv",
      "Thời lượng: 35 giây.\n" +
        "Health Index là PC1 sau StandardScaler; scaler, PCA và hướng PC1 fit trên 90 train engines, 10 engine còn lại chỉ được transform. Hình cho thấy chỉ số giảm dần và giảm nhanh ở cuối vòng đời: đây là biểu đồ xu hướng suy giảm mà đề tài yêu cầu.\n" +
        "ARIMA fit 70% đầu chuỗi, dự báo 30% cuối. MAE trung bình 0,4110, thấp hơn drift 0,7469 và last-value 0,9478 (thang Health Index, không phải cycle).\n" +
        "Giới hạn: ARIMA chỉ thấp hơn drift ở 7/10 engine; coverage khoảng dự đoán 95% dao động 21,43%–97,03%. Health Index forecast không phải RUL forecast; đây là phân tích phụ.");
    img(s, "outputs/timeseries/figures/health_index_selected_engines.png", 0.6, 2.05, 7.4, 7.4 * 960 / 1760, "Health Index của 10 validation engines");
    const labels = { ARIMA: "ARIMA", drift: "Drift", last_value: "Last-value" };
    const keys = ["ARIMA", "drift", "last_value"];
    s.addChart(pres.charts.BAR, [{ name: "MAE", labels: keys.map((k) => `${labels[k]} · ${vn(ts.mae[k], 4)}`), values: keys.map((k) => Number(ts.mae[k].toFixed(4))) }], {
      x: 8.3, y: 2.05, w: 4.43, h: 3.0, barDir: "bar", chartColors: [C.teal],
      showTitle: true, title: `MAE trung bình trên ${ts.n_engines} engine (thang Health Index)`, titleFontSize: 11.5, titleColor: C.ink, titleFontFace: FONT,
      showValue: false,
      catAxisLabelFontSize: 13, catAxisLabelColor: C.ink, valAxisHidden: true, valGridLine: { style: "none" }, catGridLine: { style: "none" },
      showLegend: false, catAxisOrientation: "maxMin", catAxisLabelFontFace: FONT, valAxisMinVal: 0, valAxisMaxVal: 1,
    });
    box(s, 8.3, 5.25, 4.43, 1.2, C.amberL);
    T(s, `Coverage khoảng dự đoán 95%: ${pct(ts.coverage_min)}–${pct(ts.coverage_max)} giữa các engine. Health Index không phải RUL; đây là phân tích phụ.`,
      { x: 8.5, y: 5.33, w: 4.05, h: 1.05, fontSize: 12.5, color: C.ink, valign: "middle" });
  }

  // ---------------------------------------------------------------- 12. Evaluation protocol
  {
    const s = content(4, "Đánh giá theo engine để tránh data leakage",
      "5-fold GroupKFold: mọi window của một engine nằm cùng một phía của phép chia", 12,
      "outputs/modeling/tables/evaluation_protocol.csv; outputs/modeling/run_info.json; report/sections/methodology.tex",
      "Thời lượng: 45 giây.\n" +
        "Sơ đồ trái: 100 train engines chia thành 5 nhóm 20 engine. Mỗi fold dùng một nhóm làm validation, bốn nhóm còn lại để fit.\n" +
        "Vì sao không chia ngẫu nhiên theo dòng: hai window liền kề của cùng engine chồng lấn 29/30 cycle. Nếu một window ở train và window kề nó ở validation, metric sẽ lạc quan.\n" +
        "Bốn bước bên phải: chọn model bằng mean engine-level MAE; đặt trọng số để tổng trọng số mỗi engine bằng 1, engine dài không chi phối loss; refit trên toàn bộ 17.731 train windows; dự đoán một điểm cho mỗi test engine rồi mới ghép nhãn.");
    for (let f = 0; f < 5; f++) {
      const y = 2.35 + f * 0.72;
      T(s, `Fold ${f + 1}`, { x: 0.6, y, w: 0.9, h: 0.5, fontSize: 13, bold: true, color: C.ink, valign: "middle" });
      for (let g = 0; g < 5; g++) {
        const val = g === f;
        box(s, 1.55 + g * 1.12, y, 1.05, 0.5, val ? C.amber : C.tealL);
        if (val) T(s, "validation", { x: 1.55 + g * 1.12, y, w: 1.05, h: 0.5, fontSize: 11, bold: true, color: C.white, align: "center", valign: "middle" });
      }
    }
    T(s, "Mỗi ô là một nhóm 20 train engines; màu nhạt là phần fit.", { x: 0.6, y: 6.0, w: 6.6, h: 0.3, fontSize: 11.5, color: C.slate });
    const steps = [
      ["Chọn model", "mean engine-level MAE trên dự đoán OOF"],
      ["Cân bằng engine", "tổng sample weight mỗi engine bằng 1 trong fold"],
      ["Refit", `toàn bộ ${vn(P.feature_rows.rolling_features_train, 0)} train windows`],
      ["Dự đoán test", "1 điểm/engine tại cycle cuối, sau đó mới ghép nhãn"],
    ];
    steps.forEach(([a, b], i) => {
      const y = 2.2 + i * 0.95;
      circleNum(s, 7.55, y, 0.5, String(i + 1), C.navy, C.white, 14);
      T(s, a, { x: 8.25, y: y - 0.03, w: 4.4, h: 0.32, fontSize: 15, bold: true, color: C.navy });
      T(s, b, { x: 8.25, y: y + 0.3, w: 4.45, h: 0.36, fontSize: 13, color: C.ink });
    });
    box(s, 7.55, 6.0, 5.18, 0.78, C.amberL);
    T(s, "Hai window liền kề trùng 29/30 cycle; chia ngẫu nhiên theo dòng làm metric lạc quan.", { x: 7.72, y: 6.07, w: 4.9, h: 0.65, fontSize: 12.5, color: C.ink, valign: "middle" });
  }

  // ---------------------------------------------------------------- 13. Model alignment
  {
    const t126 = A.trajectory.hgb126, t169 = A.trajectory.hgb169;
    const e126 = A.endpoint.hgb126, e169 = A.endpoint.hgb169;
    const s = content(4, "HGB169 là predictor candidate trên panel chung",
      "HGB169 giảm MAE trajectory 3,2134 cycles; audit có điều kiện", 13,
      "outputs/course_alignment/20261007/summary.json; outputs/course_alignment/20261007/paired_bootstrap_ci.csv",
      "Thời lượng: 55 giây.\n" +
        "Alignment audit chấm lại HGB126 và HGB169 trên cùng 17.731 dòng train, cùng khóa engine và cùng grouped folds. HGB126 gồm 126 rolling features; HGB169 thêm cycle và 42 history features, thành 169 features.\n" +
        `Trajectory: MAE ${vn(t126.mean_engine_mae, 4)} → ${vn(t169.mean_engine_mae, 4)} và RMSE ${vn(t126.rmse, 4)} → ${vn(t169.rmse, 4)} cycles. Endpoint tại các mốc 0,5/0,7/0,9: MAE ${vn(e126.mean_engine_mae, 4)} → ${vn(e169.mean_engine_mae, 4)} cycles.\n` +
        `Paired bootstrap trên prediction đã lưu: delta trajectory MAE ${vn(A.delta.trajectory_mae.value, 4)} cycles, CI 95% [${vn(A.delta.trajectory_mae.ci_lower, 4)}; ${vn(A.delta.trajectory_mae.ci_upper, 4)}]. Đây chưa phải refit mù và chưa bao gồm seed/model-selection uncertainty.\n` +
        "Vì vậy HGB169 là predictor candidate chính trong phạm vi audit; HGB126 vẫn giữ vai trò baseline lịch sử để báo cáo metric test ở slide 14.");
    s.addChart(pres.charts.BAR, [{
      name: "HGB126 baseline", labels: ["Trajectory MAE", "Endpoint MAE"],
      values: [Number(t126.mean_engine_mae.toFixed(2)), Number(e126.mean_engine_mae.toFixed(2))],
    }, {
      name: "HGB169 candidate", labels: ["Trajectory MAE", "Endpoint MAE"],
      values: [Number(t169.mean_engine_mae.toFixed(2)), Number(e169.mean_engine_mae.toFixed(2))],
    }], {
      x: 0.6, y: 2.05, w: 7.4, h: 4.75, barDir: "col", barGrouping: "clustered", chartColors: [C.ice, C.teal],
      showTitle: true, title: "MAE trên panel chung (cycles)", titleFontSize: 13, titleColor: C.ink, titleFontFace: FONT,
      showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 13, dataLabelColor: C.ink,
      catAxisLabelFontSize: 14, catAxisLabelColor: C.ink, catAxisLabelFontFace: FONT, valAxisHidden: true,
      valGridLine: { style: "none" }, catGridLine: { style: "none" }, showLegend: true, legendPos: "b", legendFontSize: 12, legendFontFace: FONT, valAxisMinVal: 0, valAxisMaxVal: 25,
    });
    table(s, [
      ["Panel / chỉ số", "HGB126", "HGB169", "Delta"],
      ["Trajectory MAE", vn(t126.mean_engine_mae), vn(t169.mean_engine_mae), vn(A.delta.trajectory_mae.value)],
      ["Trajectory RMSE", vn(t126.rmse), vn(t169.rmse), vn(A.delta.trajectory_rmse.value)],
      ["Endpoint MAE", vn(e126.mean_engine_mae), vn(e169.mean_engine_mae), vn(A.delta.endpoint_mae.value)],
      ["Endpoint RMSE", vn(e126.rmse), vn(e169.rmse), vn(A.delta.endpoint_rmse.value)],
    ], { x: 8.15, y: 2.1, w: 4.58, colW: [1.65, 0.95, 0.95, 1.03], rowH: 0.48, fontSize: 11.5, highlight: 2 });
    box(s, 8.15, 4.65, 4.58, 1.95, C.mist);
    T(s, "Cách đọc kết quả", { x: 8.38, y: 4.83, w: 4.0, h: 0.34, fontSize: 14, bold: true, color: C.navy });
    T(s, "Âm ở cột Delta nghĩa là HGB169 thấp hơn HGB126. Hai panel dùng cùng engine, fold và prediction đã lưu; chưa có blind refit độc lập, còn historical test readout chỉ dùng để báo cáo.", { x: 8.38, y: 5.25, w: 4.0, h: 1.1, fontSize: 12.5, color: C.ink });
  }

  // ---------------------------------------------------------------- 14. Test result
  {
    const f = M.final;
    const s = content(4, `Metric test lịch sử của HGB126: MAE ${vn(f.mean_engine_mae)} cycles`,
      "HGB126 là baseline lịch sử; metric test được giữ riêng vì HGB169 chưa có refit mù độc lập",
      14,
      "outputs/modeling/tables/final_metrics.json; outputs/modeling/figures/final_true_vs_predicted.png",
      "Thời lượng: 1 phút 5 giây.\n" +
        `Đây là metric test lịch sử của HGB126, không phải kết quả test độc lập của HGB169: mean engine MAE ${vn(f.mean_engine_mae, 4)} cycles, RMSE ${vn(f.rmse, 4)} cycles, NASA score ${vn(f.nasa_score, 4)}.\n` +
        `Bootstrap theo engine ${vn(M.bootstrap_resamples, 0)} lần: khoảng tin cậy 95% cho MAE là [${vn(M.ci_mae.ci_lower, 4)}; ${vn(M.ci_mae.ci_upper, 4)}].\n` +
        "HGB169 được báo cáo ở slide 13 trên panel trajectory và endpoint chung; không trộn các metric đó với metric test lịch sử này.\n" +
        "Đọc hình: trục ngang là RUL thật, trục dọc là RUL dự đoán; điểm càng gần đường chéo càng đúng. Vùng RUL thấp bám sát đường chéo hơn vùng RUL cao.\n" +
        "Trên slide làm tròn 2 chữ số; số đầy đủ nằm trong notes và báo cáo. NASA score bất đối xứng, không có đơn vị cycle.");
    const cards = [
      [`${vn(f.mean_engine_mae)}`, "MAE (cycles)", `CI 95%: [${vn(M.ci_mae.ci_lower)}; ${vn(M.ci_mae.ci_upper)}]`, C.navy, C.white, C.ice],
      [`${vn(f.rmse)}`, "RMSE (cycles)", "Phạt sai số lớn mạnh hơn MAE", C.mist, C.navy, C.slate],
      [`${vn(f.nasa_score)}`, "NASA score", "Phạt dự đoán trễ nặng hơn; không có đơn vị", C.mist, C.navy, C.slate],
    ];
    cards.forEach(([v, l, d, fill, fg, sub], i) => {
      const y = 2.1 + i * 1.6;
      box(s, 0.6, y, 5.2, 1.45, fill);
      T(s, v, { x: 0.85, y: y + 0.12, w: 2.6, h: 0.75, fontSize: 36, bold: true, color: fg, valign: "middle" });
      T(s, l, { x: 3.4, y: y + 0.2, w: 2.3, h: 0.35, fontSize: 15, bold: true, color: fg });
      T(s, d, { x: 3.4, y: y + 0.58, w: 2.3, h: 0.75, fontSize: 12, color: sub });
    });
    img(s, "outputs/modeling/figures/final_true_vs_predicted.png", 6.3, 2.05, 6.43, 6.43 * 800 / 1120, "RUL thật và RUL dự đoán trên 100 test engines");
  }

  // ---------------------------------------------------------------- 15. Error analysis
  {
    const w = M.worst;
    const s = content(4, "Phân tích sai số của HGB126 baseline lịch sử",
      "Dự đoán cao hơn RUL thật tương ứng cảnh báo muộn; NASA score phạt loại sai số này nặng hơn", 15,
      "outputs/modeling/predictions/final_evaluation.parquet; outputs/modeling/figures/final_residuals_by_rul.png; outputs/diagnostics/error_diagnostics.json",
      "Thời lượng: 45 giây.\n" +
        "Hình là residual của metric test lịch sử HGB126: residual = dự đoán − thật theo RUL thật. Residual dương nghĩa là model nghĩ engine còn sống lâu hơn thực tế.\n" +
        `${M.n_over} dự đoán cao hơn RUL thật, ${M.n_under} dự đoán thấp hơn. Sai số tăng khi RUL thật lớn: ở đầu vòng đời tín hiệu suy giảm còn yếu.\n` +
        `Ví dụ engine ${w.engine_id}: RUL thật ${vn(w.y_true, 0)}, dự đoán ${vn(w.y_pred, 4)}, sai ${vn(w.abs_error, 4)} cycles. Không loại engine này khỏi metric.\n` +
        `Mười trường hợp sai lớn nhất chiếm ${pct(DG.top10_nasa_share, 1)} NASA score. Trên OOF, khi RUL thật 150–200 dự đoán trung bình ${vn(DG.oof_mean_pred_150_200, 2)}; khi 200–400 là ${vn(DG.oof_mean_pred_200_400, 2)} cycles: model không phân biệt được các trạng thái còn khỏe (outputs/diagnostics/error_diagnostics.json).
` +
        "Thông điệp: metric trung bình không thay thế việc xem từng trường hợp sai lớn.");
    img(s, "outputs/modeling/figures/final_residuals_by_rul.png", 0.6, 2.05, 6.6, 6.6 * 800 / 1120, "Residual theo RUL thật trên 100 test engines");
    T(s, "Hướng sai số trên 100 test engines", { x: 7.6, y: 2.15, w: 5.1, h: 0.34, fontSize: 15, bold: true, color: C.navy });
    const total = M.n_over + M.n_under, bw = 5.1;
    rect(s, 7.6, 2.65, bw * M.n_over / total, 0.6, C.amber);
    rect(s, 7.6 + bw * M.n_over / total, 2.65, bw * M.n_under / total, 0.6, C.teal);
    T(s, `${M.n_over} cao hơn`, { x: 7.7, y: 2.65, w: 2.5, h: 0.6, fontSize: 15, bold: true, color: C.white, valign: "middle" });
    T(s, `${M.n_under} thấp hơn`, { x: 7.6 + bw * M.n_over / total, y: 2.65, w: bw * M.n_under / total, h: 0.6, fontSize: 12.5, bold: true, color: C.white, align: "center", valign: "middle" });
    box(s, 7.6, 3.65, 5.13, 2.0, C.mist);
    T(s, `Trường hợp sai lớn nhất: engine ${w.engine_id}`, { x: 7.85, y: 3.8, w: 4.7, h: 0.34, fontSize: 14, bold: true, color: C.navy });
    [["RUL thật", vn(w.y_true, 0)], ["Dự đoán", vn(w.y_pred)], ["Sai số", vn(w.abs_error)]].forEach(([l, v], i) => {
      T(s, v, { x: 7.85 + i * 1.6, y: 4.3, w: 1.55, h: 0.6, fontSize: 26, bold: true, color: i === 2 ? C.amber : C.navy });
      T(s, `${l} (cycles)`, { x: 7.85 + i * 1.6, y: 4.95, w: 1.55, h: 0.5, fontSize: 11.5, color: C.slate });
    });
    T(s, `10 trường hợp sai lớn nhất chiếm ${pct(DG.top10_nasa_share, 1)} NASA score. Trên OOF, dự đoán dừng quanh ${vn(DG.oof_mean_pred_200_400, 0)}–${vn(DG.oof_mean_pred_150_200, 0)} cycles khi RUL thật > 150.`,
      { x: 7.6, y: 5.85, w: 5.13, h: 0.9, fontSize: 13, bold: true, color: C.navy });
  }

  // ---------------------------------------------------------------- 16. Early warning
  {
    const wg = M.warning, tm = wg.timing, ld = wg.lead;
    const s = content(4, "Warning policy phải được đánh giá riêng với predictor",
      "Threshold 30: HGB126 51%, HGB169 45%; threshold 40: 94% và 96%", 16,
      "outputs/course_alignment/20261007/warning_fixed_policies.csv; outputs/modeling/figures/warning_trajectory.png",
      "Thời lượng: 1 phút 30 giây.\n" +
        "Đây là biểu đồ xu hướng của HGB126 baseline ở threshold 30: đường xanh là RUL thật, đường đỏ là RUL dự đoán OOF, đường cam là ngưỡng 30 cycles. Cảnh báo đầu tiên tại cycle 162, trước failure ở cycle 192 đúng 30 cycles.\n" +
        `Trên HGB126 threshold 30: ${tm.before_threshold} cảnh báo trước khi RUL thật chạm 30, ${tm.at_threshold} đúng ngưỡng, ${tm.after_threshold} sau ngưỡng nhưng vẫn trước failure. Lead time trung vị ${vn(ld.median, 0)} cycles.\n` +
        "Bảng bên phải là fixed-policy sensitivity trên cùng 100 engine: threshold 30 và 40, persistence 1, lead requirement 30. HGB126 và HGB169 không được chọn lại bằng warning policy này.\n" +
        "Không tuyên bố cải thiện toàn bộ quy trình: threshold 30 HGB169 kém đúng thời điểm hơn; threshold 40 HGB169 nhỉnh hơn nhưng burden tăng nhẹ. Chưa có chi phí bảo trì nên đây chưa phải chính sách triển khai.");
    img(s, "outputs/modeling/figures/warning_trajectory.png", 0.6, 2.05, 6.9, 6.9 * 720 / 1280, "Quỹ đạo RUL thật, RUL dự đoán OOF và ngưỡng cảnh báo của engine 1");
    const warningRows = [
      ["Model", "Ngưỡng", "Đúng lúc", "Trễ/bỏ lỡ", "Burden"],
      ["HGB126", "30", pct(fixedWarning.hgb126_30.on_time_rate, 0), pct(fixedWarning.hgb126_30.late_or_missed_rate, 0), pct(fixedWarning.hgb126_30.early_alarm_burden, 3)],
      ["HGB169", "30", pct(fixedWarning.hgb169_30.on_time_rate, 0), pct(fixedWarning.hgb169_30.late_or_missed_rate, 0), pct(fixedWarning.hgb169_30.early_alarm_burden, 3)],
      ["HGB126", "40", pct(fixedWarning.hgb126_40.on_time_rate, 0), pct(fixedWarning.hgb126_40.late_or_missed_rate, 0), pct(fixedWarning.hgb126_40.early_alarm_burden, 3)],
      ["HGB169", "40", pct(fixedWarning.hgb169_40.on_time_rate, 0), pct(fixedWarning.hgb169_40.late_or_missed_rate, 0), pct(fixedWarning.hgb169_40.early_alarm_burden, 3)],
    ];
    table(s, warningRows, { x: 7.65, y: 2.05, w: 5.08, colW: [1.0, 0.75, 1.1, 1.15, 1.08], rowH: 0.46, fontSize: 10.5, highlight: 4 });
    T(s, "Fixed-policy sensitivity trên cùng prediction panel", { x: 7.65, y: 4.55, w: 5.0, h: 0.3, fontSize: 13.5, bold: true, color: C.navy });
    T(s, "Burden = tỷ lệ cảnh báo sớm; threshold chưa được hiệu chuẩn theo chi phí. Predictor selection và warning policy là hai quyết định riêng.", { x: 7.65, y: 4.95, w: 5.0, h: 0.7, fontSize: 12.5, color: C.ink });
    box(s, 7.65, 5.75, 5.08, 0.92, C.amberL);
    T(s, [
      { text: `HGB126 threshold 30: ${tm.before_threshold + tm.at_threshold}/${fd.FD001.test_engines} đúng lúc; median lead ${vn(ld.median, 0)} cycles.`, options: { bold: true, breakLine: true } },
      { text: "Hình bên trái là mô tả nội bộ trên OOF.", options: {} },
    ], { x: 7.88, y: 5.84, w: 4.6, h: 0.75, fontSize: 11.5, color: C.ink, valign: "middle", paraSpaceAfter: 3 });
    T(s, "Engine 1: cảnh báo tại cycle 162, failure tại cycle 192.", { x: 0.6, y: 2.05 + 6.9 * 720 / 1280 + 0.1, w: 6.9, h: 0.3, fontSize: 11.5, color: C.slate });
  }

  // ---------------------------------------------------------------- 17. Improvement
  {
    const o = I.outer, dci = I.low_delta_ci;
    const s = content(5, "Nested CV: kiểm tra cải thiện trên engine được giữ riêng",
      "Kết quả gốc và thí nghiệm bổ sung; chỉ thay baseline khi đạt tiêu chí đã chốt trước", 17,
      `${D.provenance.improvement_run}/; ${D.provenance.followup_run}/`,
      "Thời lượng: 1 phút 15 giây.\n" +
        `Mục tiêu: giảm sai số khi RUL ≤ 30 và giảm cảnh báo trễ, không làm tăng MAE toàn bộ. Nested GroupKFold ${I.outer_splits}×${I.inner_splits}: vòng ngoài giữ riêng một nhóm engine để đánh giá; vòng trong chia phần còn lại thành 3 nhóm để chọn cấu hình.\n` +
        `Vòng ngoài trên ${vn(o.baseline.rows, 0)} dòng: MAE RUL thấp ${vn(o.baseline.low_mae, 4)} → ${vn(o.weights.low_mae, 4)} cycles (Δ +${vn(dci[0], 4)}, CI 95% [${vn(dci[1], 4)}; ${vn(dci[2], 4)}]); cảnh báo trễ/bỏ lỡ ${o.baseline.late} → ${o.weights.late}. Không đạt tiêu chí, giữ baseline; metric test lịch sử HGB126 vẫn 15,3490.\n` +
        "Bảng dưới là outer CV của đợt bổ sung: FD001 capped RUL/history, FD002/FD004 settings/condition normalization. FD002/FD004 là within-subset, không phải transfer. Test lịch sử đã được xem nên đây là exploratory follow-up.\n" +
        D.followup.map((r) => `${r.dataset}: outer MAE ${vn(r.baseline_outer_mae, 4)} → ${vn(r.selected_outer_mae, 4)}, gate ${r.outer_gate_passed ? "đạt" : "chưa đạt"}; test MAE ${vn(r.test_mae, 4)}.`).join("\n"));
    T(s, "Nested GroupKFold", { x: 0.6, y: 2.1, w: 3.4, h: 0.34, fontSize: 15, bold: true, color: C.navy });
    const nest = [
      [`${I.outer_splits} vòng ngoài`, "giữ riêng một nhóm engine", C.navy],
      [`${I.inner_splits} vòng trong`, "chọn cấu hình chỉ từ train", C.teal],
      ["CI nghiên cứu gốc", `ΔMAE RUL ≤ 30: +${vn(dci[0], 4)}\nCI 95% [${vn(dci[1], 4)}; ${vn(dci[2], 4)}]`, C.amber],
    ];
    nest.forEach(([a, b, col], i) => {
      const y = 2.6 + i * 1.35;
      box(s, 0.6, y, 3.4, 1.05, col);
      T(s, a, { x: 0.8, y: y + 0.12, w: 3.0, h: 0.36, fontSize: 15, bold: true, color: C.white });
      T(s, b, { x: 0.8, y: y + 0.48, w: 3.1, h: 0.52, fontSize: 12, color: C.white });
      if (i < 2) arrow(s, 2.3, y + 1.07, 2.3, y + 1.33);
    });
    T(s, "FD001 gốc: thử window, tham số và trọng số", { x: 4.4, y: 2.1, w: 8.3, h: 0.34, fontSize: 14, bold: true, color: C.navy });
    table(s, [
      ["Chỉ số", "Baseline", "Quy trình chọn mới"],
      ["MAE toàn bộ (cycles)", vn(o.baseline.mae), vn(o.weights.mae)],
      ["MAE khi RUL ≤ 30 (cycles)", vn(o.baseline.low_mae, 4), vn(o.weights.low_mae, 4)],
      ["Cảnh báo trễ hoặc bỏ lỡ", `${o.baseline.late}/100`, `${o.weights.late}/100`],
    ], { x: 4.4, y: 2.5, w: 8.33, colW: [3.73, 2.3, 2.3], rowH: 0.4, fontSize: 12.5 });
    T(s, "Bổ sung 29/09: outer MAE của quy trình chọn (cycles)", { x: 4.4, y: 4.35, w: 8.3, h: 0.34, fontSize: 14, bold: true, color: C.navy });
    table(s, [
      ["Dataset", "Baseline", "Quy trình chọn", "Tiêu chí"],
      ...D.followup.map((r) => [r.dataset, vn(r.baseline_outer_mae), vn(r.selected_outer_mae), r.outer_gate_passed ? "Đạt" : "Chưa đạt"]),
    ], { x: 4.4, y: 4.75, w: 8.33, colW: [2.93, 1.8, 1.8, 1.8], rowH: 0.4, fontSize: 12.5 });
    T(s, "FD001: window 30; FD002/FD004: window 10, huấn luyện trong từng subset.",
      { x: 4.4, y: 6.45, w: 8.33, h: 0.35, fontSize: 13, bold: true, color: C.amber });
  }

  // ---------------------------------------------------------------- 18. Transfer
  {
    const tg = ["FD002", "FD003", "FD004"];
    const within = tg.map((t) => tr(t, t).mae), cross = tg.map((t) => tr("FD001", t).mae);
    const ratios = tg.map((_, i) => cross[i] / within[i]);
    const s = content(5, "Model FD001 không chuyển trực tiếp sang FD002–FD004",
      `Cùng pipeline với window ${D.transfer.window}; MAE tăng ${vn(Math.min(...ratios), 1)}–${vn(Math.max(...ratios), 1)} lần khi đổi điều kiện vận hành hoặc fault mode`, 18,
      "outputs/cross_dataset/matrix.json; outputs/diagnostics/error_diagnostics.json",
      "Thời lượng: 55 giây.\n" +
        "Câu hỏi: model FD001 có dùng được cho subset khác không. Chạy cùng pipeline với window 10 vì FD004 có test engine ngắn nhất chỉ 19 cycles. Không so các số này với MAE 15,3490 dùng window 30.\n" +
        `Cột xanh nhạt: train và test cùng subset (FD002 ${vn(within[0], 4)}, FD003 ${vn(within[1], 4)}, FD004 ${vn(within[2], 4)}). Cột cam: giữ model FD001 áp dụng sang test subset khác (${vn(cross[0], 4)}, ${vn(cross[1], 4)}, ${vn(cross[2], 4)}). FD001 nội bộ với window 10 là ${vn(tr("FD001", "FD001").mae, 4)}.\n` +
        `Chẩn đoán: với FD002 và FD004 chỉ khoảng ${inr("FD002")} và ${inr("FD004")} giá trị sensor test nằm trong miền train FD001, dự đoán gần như hằng số (độ lệch chuẩn ${vn(DG.transfer.FD002.pred_std, 1)} và ${vn(DG.transfer.FD004.pred_std, 1)}). FD003 có ${inr("FD003")} giá trị trong miền nhưng dự đoán lệch cao trung bình ${vn(DG.transfer.FD003.mean_signed_error, 1)} cycles.
` +
        "Diễn giải: khác biệt điều kiện vận hành (FD002) và fault mode (FD003) đều làm MAE tăng; FD004 có cả hai. Kết luận thực tế: cần huấn luyện lại hoặc domain adaptation trước khi áp dụng. Đây là negative result nhưng trả lời trực tiếp câu hỏi tổng quát hóa.");
    s.addChart(pres.charts.BAR, [
      { name: "Train và test cùng subset", labels: tg, values: within.map((v) => Number(v.toFixed(2))) },
      { name: "Train FD001 → test subset", labels: tg, values: cross.map((v) => Number(v.toFixed(2))) },
    ], {
      x: 0.6, y: 2.0, w: 7.6, h: 4.8, barDir: "col", barGrouping: "clustered", chartColors: [C.ice, C.amber],
      showTitle: true, title: `Mean engine MAE trên test (cycles, làm tròn), window ${D.transfer.window}`, titleFontSize: 13, titleColor: C.ink, titleFontFace: FONT,
      showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 13, dataLabelColor: C.ink, dataLabelFormatCode: "0",
      catAxisLabelFontSize: 14, catAxisLabelColor: C.ink, catAxisLabelFontFace: FONT, valAxisHidden: true,
      valGridLine: { style: "none" }, catGridLine: { style: "none" }, showLegend: true, legendPos: "b", legendFontSize: 12, legendFontFace: FONT, valAxisMaxVal: 70,
    });
    const why = [["FD002", `6 điều kiện · ${inr("FD002")} trong miền FD001`], ["FD003", `fault Fan · ${inr("FD003")} trong miền FD001`], ["FD004", `cả hai · ${inr("FD004")} trong miền FD001`]];
    why.forEach(([n, d], i) => {
      const y = 2.15 + i * 0.95;
      box(s, 8.6, y, 4.13, 0.8, C.mist);
      T(s, n, { x: 8.8, y, w: 1.1, h: 0.8, fontSize: 16, bold: true, color: C.navy, valign: "middle" });
      T(s, `${d}\n${vn(within[i])} → ${vn(cross[i])} cycles (×${vn(ratios[i], 1)})`, { x: 9.9, y: y + 0.08, w: 2.75, h: 0.66, fontSize: 11.5, color: C.ink, valign: "middle" });
    });
    box(s, 8.6, 5.1, 4.13, 1.65, C.navy);
    T(s, "Cần huấn luyện lại hoặc domain adaptation trước khi dùng model FD001 cho subset khác.", { x: 8.8, y: 5.22, w: 3.8, h: 1.4, fontSize: 14, bold: true, color: C.white, valign: "middle" });
  }

  // ---------------------------------------------------------------- 19. Kết luận
  {
    const f = M.final, ld = M.warning.lead;
    const s = content(5, "Trả lời sáu câu hỏi nghiên cứu", "Kết luận chỉ áp dụng cho protocol và dữ liệu mô phỏng đã nêu", 19,
      "report/sections/conclusion.tex; các artifact đã dẫn ở slide 9–18",
      "Thời lượng: 1 phút 15 giây.\n" +
        "Q1: FD001 đạt kiểm tra schema, khóa và cycle; Q2: 15/21 sensor thay đổi giữa đầu và cuối vòng đời, Health Index giảm rõ ở cuối vòng đời.\n" +
        "Q3: parity Pandas–PySpark đạt trên ba artifact với sai khác tuyệt đối tối đa bằng 0.\n" +
        `Q4: HGB169 là predictor candidate chính có điều kiện; trajectory MAE ${vn(A.trajectory.hgb169.mean_engine_mae, 4)} so với ${vn(A.trajectory.hgb126.mean_engine_mae, 4)} của HGB126.\n` +
        `Q5: warning policy threshold 30 đúng thời điểm ${pct(fixedWarning.hgb126_30.on_time_rate, 0)} với HGB126 và ${pct(fixedWarning.hgb169_30.on_time_rate, 0)} với HGB169; threshold 40 lần lượt ${pct(fixedWarning.hgb126_40.on_time_rate, 0)} và ${pct(fixedWarning.hgb169_40.on_time_rate, 0)}. Q6: FD001 không chuyển trực tiếp sang FD002–FD004.\n` +
        "Đóng góp Big Data: batch ETL nhiều file, Parquet Silver/Gold, parity Pandas–PySpark, artifact có hash và run_info.");
    const cols = [
      ["Q1", "Dữ liệu hợp lệ", "Đạt", "Schema, khóa và cycle của FD001 qua kiểm tra"],
      ["Q2", "Xu hướng sensor", `${D.statistics.n_reject}/${D.statistics.n_sensors}`, "sensor thay đổi đầu/cuối; Health Index giảm rõ"],
      ["Q3", "Parity ETL", "0 sai khác", "Ba artifact Pandas–PySpark khớp tuyệt đối"],
      ["Q4", "HGB comparison", `${vn(A.delta.trajectory_mae.value, 2)}`, "delta trajectory MAE; HGB169 là candidate có điều kiện"],
      ["Q5", "Warning policy", "51% / 45%", "threshold 30: HGB126 / HGB169; policy tách khỏi predictor"],
      ["Q6", "Chuyển miền", "Không", "FD001 cần huấn luyện lại hoặc domain adaptation"],
    ];
    cols.forEach(([q, h, v, d], i) => {
      const colIndex = i % 3, rowIndex = Math.floor(i / 3), x = 0.6 + colIndex * 4.1, y = 2.05 + rowIndex * 2.15;
      box(s, x, y, 3.9, 1.92, C.mist);
      circleNum(s, x + 0.22, y + 0.2, 0.48, q, C.amber, C.white, 11);
      T(s, h, { x: x + 0.85, y: y + 0.2, w: 2.8, h: 0.38, fontSize: 14, bold: true, color: C.navy, valign: "middle" });
      T(s, v, { x: x + 0.22, y: y + 0.78, w: 3.4, h: 0.46, fontSize: 24, bold: true, color: C.navy });
      T(s, d, { x: x + 0.22, y: y + 1.28, w: 3.42, h: 0.48, fontSize: 11.5, color: C.ink });
    });
    T(s, "Đóng góp Big Data: batch ETL nhiều file, Parquet Silver/Gold, parity Pandas–PySpark và artifact có hash, run_info để tái lập.",
      { x: 0.6, y: 6.45, w: 12.1, h: 0.35, fontSize: 12.5, bold: true, color: C.navy });
  }

  // ---------------------------------------------------------------- 20. Giới hạn và hỏi đáp
  {
    const s = pres.addSlide();
    s.background = { color: C.navy };
    T(s, "PHẦN 5 · MỞ RỘNG VÀ KẾT LUẬN", { x: 0.6, y: 0.4, w: 6, h: 0.3, fontSize: 10.5, bold: true, color: C.ice, charSpacing: 1 });
    T(s, "Giới hạn và hướng tiếp theo", { x: 0.6, y: 0.8, w: 12, h: 0.62, fontSize: 27, bold: true, color: C.white, valign: "middle" });
    const lim = [
      "Dữ liệu mô phỏng; FD001 chỉ có một điều kiện vận hành và một fault mode.",
      "Mỗi test engine chỉ được đánh giá tại một điểm cắt.",
      "Không có chi phí, downtime, intervention records: chưa tối ưu được lịch bảo trì.",
      "Benchmark Pandas/PySpark chạy local; ghi qua driver, chưa đo cluster.",
    ];
    const next = [
      "Refit mù HGB169 trên cùng protocol để xác nhận candidate bằng một lần chạy độc lập.",
      "Hiệu chuẩn warning policy riêng với chi phí báo sớm, báo trễ và tiêu chí vận hành.",
      "Đánh giá sai số theo toàn quỹ đạo trên engine chưa thấy và kèm khoảng không chắc chắn.",
      "Benchmark PySpark trên cluster thực; benchmark local hiện chỉ là bằng chứng khả năng chạy.",
    ];
    const col = (x, head, items, color) => {
      T(s, head, { x, y: 1.75, w: 5.8, h: 0.4, fontSize: 17, bold: true, color });
      T(s, items.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < items.length - 1 } })),
        { x, y: 2.3, w: 5.7, h: 3.0, fontSize: 15, color: C.white, paraSpaceAfter: 10 });
    };
    col(0.6, "Giới hạn đã biết", lim, "F2B880");
    col(6.95, "Đã bổ sung và hướng tiếp theo", next, C.ice);
    T(s, "Xin cảm ơn thầy/cô. Xin nhận câu hỏi.", { x: 0.6, y: 5.75, w: 12.1, h: 0.6, fontSize: 26, bold: true, color: C.white });
    T(s, "Dashboard: uv run --extra dashboard streamlit run app.py · Kịch bản demo: handoff/DEMO_SCRIPT.md", { x: 0.6, y: 6.45, w: 12.1, h: 0.35, fontSize: 13, color: C.ice });
    T(s, "20 / 20", { x: 11.93, y: 7.05, w: 0.8, h: 0.26, fontSize: 10, color: C.ice, align: "right", valign: "middle" });
    s.addNotes(
      "Thời lượng: 1 phút, sau đó hỏi đáp.\n" +
        "HGB169 hiện là predictor candidate có điều kiện từ alignment audit; bước xác nhận cần refit mù độc lập. Warning policy phải được hiệu chuẩn riêng, vì threshold 30 và 40 cho trade-off khác nhau. Capped RUL, history features, condition normalization và benchmark local có I/O đã được kiểm tra; chưa mở thêm nhánh CNN/LSTM hoặc training mới.\n" +
        "Nếu còn thời gian hoặc cô yêu cầu: mở dashboard theo handoff/DEMO_SCRIPT.md (khoảng 4 phút); không chạy lại training khi trình bày.\n" +
        "Nguồn: report/sections/discussion.tex.");
  }

  await pres.writeFile({ fileName: OUT });
  console.log(`wrote ${path.relative(ROOT, OUT)}`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
