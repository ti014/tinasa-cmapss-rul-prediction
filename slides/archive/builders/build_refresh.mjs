import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, PresentationFile } from "@oai/artifact-tool";

const workspaceDir = process.env.WORKSPACE_DIR ?? "D:/IUH/BigData/Final_Project";
const sourcePath = path.join(workspaceDir, "slides", "archive", "decks", "cmapss_fd001_presentation_before_refresh.pptx");
const buildDir = process.env.TMP_DIR ?? path.join(workspaceDir, "tmp", "slides-refresh-20260924");
const candidatePath = path.join(buildDir, "candidate.pptx");
const previewDir = path.join(buildDir, "renders-candidate");

const artifact = (relativePath) => path.join(workspaceDir, relativePath);
const readBytes = async (relativePath) => new Uint8Array(await fs.readFile(artifact(relativePath)));

const presentation = await PresentationFile.importPptx(await FileBlob.load(sourcePath));
const byId = (id) => presentation.resolve(id);

function updateText(id, text, style = {}) {
  const shape = byId(id);
  shape.text = text;
  if (style.fontSize || style.color || style.bold || style.alignment || style.autoFit || style.typeface || style.lineSpacing) {
    shape.text.style = style;
  }
  return shape;
}

function setPosition(id, position) {
  byId(id).position = position;
  return byId(id);
}

function addText(slide, text, position, style = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    position,
    fill: "none",
    line: { fill: "none", width: 0 },
  });
  shape.text = text;
  shape.text.style = {
    typeface: "Arial",
    fontSize: 18,
    color: "#16202A",
    alignment: "left",
    verticalAlignment: "top",
    autoFit: "shrinkText",
    insets: { left: 0, right: 0, top: 0, bottom: 0 },
    ...style,
  };
  return shape;
}

async function replaceImage(id, relativePath, position, alt, fit = "contain") {
  const image = byId(id);
  await image.replace({
    blob: await readBytes(relativePath),
    contentType: "image/png",
    alt,
    fit,
  });
  image.frame = position;
  image.fit = fit;
  image.crop = { left: 0, top: 0, right: 0, bottom: 0 };
  return image;
}

function deleteById(id) {
  byId(id).delete();
}

// Correct the folio boxes. The original width was too narrow and wrapped 02–10.
for (const id of [
  "sh/s72xofmh", "sh/sfix03y9", "sh/ih8ju9sn", "sh/58vehgvy", "sh/h4bupgn6",
  "sh/n6x0fqlo", "sh/o7ydoret", "sh/98ji147e", "sh/d87uxg3m",
]) {
  const folio = setPosition(id, { left: 1168, top: 684, width: 48, height: 18 });
  folio.text.alignment = "right";
  folio.text.autoFit = "shrinkText";
}

// Slide 1: state both the RUL task and the early-warning companion analysis.
updateText("sh/k3yl0zql", "Dự đoán RUL và cảnh báo sớm", {
  typeface: "Arial", fontSize: 46, color: "#FFFFFF", bold: true, alignment: "left", autoFit: "shrinkText",
});
updateText("sh/65g3298r", "Pipeline tái lập và đánh giá theo engine", {
  typeface: "Arial", fontSize: 21, color: "#FFFFFF", alignment: "left", autoFit: "shrinkText",
});
byId("nt/y90nupkv").textFrame.setText(
  "Nguồn: report/main.pdf, outputs/modeling/tables/final_metrics.json và outputs/modeling/tables/warning_summary.json. Warning dùng selected-model OOF train với ngưỡng RUL ≤30; không phải maintenance policy.",
);

// Slide 2: separate held-out RUL evaluation from the train-only warning evidence.
updateText("sh/x4r21kru", "Cảnh báo sớm dùng selected-model OOF trên train trajectories với ngưỡng minh họa RUL ≤30. Test labels chỉ dùng ở bước đánh giá cuối.", {
  typeface: "Arial", fontSize: 20, color: "#16202A", alignment: "left", autoFit: "shrinkText", lineSpacing: 1.04,
});
byId("nt/hwbqtkby").textFrame.setText(
  "Nguồn: outputs/etl/run_info.json, outputs/features/run_info.json, data/CONTRACTS.md và outputs/modeling/tables/warning_summary.json. RUL final đánh giá trên 100 test engines; warning đánh giá trên train OOF.",
);

// Slide 3: make the data boundary and current review evidence explicit.
updateText("sh/3ah8rqlg", "train/test\nobservations", {
  typeface: "Arial", fontSize: 18, color: "#FFFFFF", alignment: "left", autoFit: "shrinkText",
});
updateText("sh/f29gbyx0", "RUL metrics\nHealth Index · OOF warning", {
  typeface: "Arial", fontSize: 17, color: "#FFFFFF", alignment: "left", autoFit: "shrinkText",
});
updateText("sh/6d0fytg3", "18 bảng khớp ngữ nghĩa · 5 hình byte-identical · raw bất biến · dashboard loader", {
  typeface: "Arial", fontSize: 18, color: "#16202A", alignment: "left", autoFit: "shrinkText",
});
addText(presentation.slides.getItem(2), "test labels: final evaluation only", {
  left: 790, top: 390, width: 350, height: 24,
}, { fontSize: 16, color: "#F28E2B", bold: true });
byId("nt/ofy9wn61").textFrame.setText(
  "Nguồn: data/CONTRACTS.md; outputs/etl, outputs/features, outputs/modeling, outputs/timeseries; tmp/e2e-review-20260924/validation.json. Test labels chỉ join ở final evaluation.",
);

// Slide 4: use one readable EDA chart and keep the statistical interpretation concise.
await replaceImage(
  "im/ed4nel0n",
  "outputs/eda/figures/engine_lifetimes.png",
  { left: 70, top: 155, width: 720, height: 285 },
  "Train engine lifetime distribution",
);
deleteById("im/1gfm90ze");
addText(presentation.slides.getItem(3), "Sensor-level paired analysis", {
  left: 835, top: 180, width: 330, height: 30,
}, { fontSize: 22, color: "#1F7A8C", bold: true });
addText(presentation.slides.getItem(3), "Effect size, confidence interval và FDR được lưu trong artifact. Kết quả mô tả khác biệt đầu/cuối vòng đời trên train, không suy ra quan hệ nhân quả.", {
  left: 835, top: 230, width: 330, height: 150,
}, { fontSize: 19, color: "#16202A", lineSpacing: 1.05 });
updateText("sh/sna103ap", "Sensor-level paired analysis lưu effect size, confidence interval và FDR trong artifact.", {
  typeface: "Arial", fontSize: 18, color: "#16202A", alignment: "left", autoFit: "shrinkText",
});
byId("nt/jyx0ra1s").textFrame.setText(
  "Nguồn: outputs/eda/figures/engine_lifetimes.png; outputs/statistics/figures/effect_size_ci.png; outputs/statistics/tables/sensor_inference.csv; report/sections/statistical-protocol.tex.",
);

// Slide 5: tighten the leakage boundary wording.
updateText("sh/v2twb6dw", "Preprocessing và model selection chỉ fit trên training folds.", {
  typeface: "Arial", fontSize: 19, color: "#16202A", alignment: "left", autoFit: "shrinkText",
});

// Slide 7: replace the superseded interval with the engine-cluster bootstrap result.
updateText("sh/l0vuh0rm", "15,3490", { typeface: "Arial", fontSize: 27, color: "#FFFFFF", bold: true, autoFit: "shrinkText" });
updateText("sh/fi9c369c", "20,5413", { typeface: "Arial", fontSize: 27, color: "#FFFFFF", bold: true, autoFit: "shrinkText" });
updateText("sh/a9ojq5kz", "1.513,2577", { typeface: "Arial", fontSize: 27, color: "#FFFFFF", bold: true, autoFit: "shrinkText" });
updateText("sh/xcz2l03q", "Bootstrap theo engine, 10.000 lần:\nMAE CI 95% [12,7286; 18,0866]", {
  typeface: "Arial", fontSize: 15, color: "#FFFFFF", alignment: "left", autoFit: "shrinkText", lineSpacing: 1.02,
});
byId("nt/gnmp4jqx").textFrame.setText(
  "Nguồn: outputs/modeling/tables/final_metrics.json; outputs/modeling/figures/final_true_vs_predicted.png; outputs/modeling/figures/final_residuals_by_rul.png. CI dùng engine-cluster resampling có hoàn lại và giữ duplicate engine draws.",
);

// Slide 8: keep one readable trajectory and add editable baseline evidence.
await replaceImage(
  "im/d4vqd4j2",
  "outputs/timeseries/figures/health_index_selected_engines.png",
  { left: 70, top: 145, width: 600, height: 300 },
  "Health Index trajectories for selected train engines",
);
deleteById("im/c32pkj2x");
const slide8 = presentation.slides.getItem(7);
const baselineTable = slide8.tables.add({
  rows: 4,
  columns: 3,
  left: 700,
  top: 145,
  width: 500,
  height: 220,
  values: [
    ["Model", "MAE", "RMSE"],
    ["ARIMA", "0,4110", "0,5027"],
    ["Drift", "0,7469", "0,9128"],
    ["Last-value", "0,9478", "1,1553"],
  ],
});
baselineTable.styleOptions = { headerRow: true, bandedRows: true };
for (let column = 0; column < 3; column += 1) {
  const cell = baselineTable.getCell(0, column);
  cell.fill = "#0F243E";
  cell.text.style = { typeface: "Arial", fontSize: 17, color: "#FFFFFF", bold: true };
}
for (let row = 1; row < 4; row += 1) {
  for (let column = 0; column < 3; column += 1) {
    baselineTable.getCell(row, column).text.style = {
      typeface: "Arial", fontSize: 17, color: "#16202A", bold: row === 1 && column === 0,
    };
  }
}
addText(slide8, "Cùng holdout Health Index: mô tả MAE/RMSE, không phải RUL forecast.", {
  left: 700, top: 385, width: 500, height: 42,
}, { fontSize: 16, color: "#52606D" });
updateText("sh/sjix0zy1", "PCA/scaler fit trên 90 engine; 10 engine được chọn chỉ transform; split prefix 70% / suffix 30%.", {
  typeface: "Arial", fontSize: 18, color: "#16202A", alignment: "left", autoFit: "shrinkText", lineSpacing: 1.03,
});
updateText("sh/je9g3ahk", "ARIMA, drift và last-value dùng cùng holdout; baseline không có prediction interval.", {
  typeface: "Arial", fontSize: 17, color: "#52606D", alignment: "left", autoFit: "shrinkText",
});
byId("nt/fu1gfa1s").textFrame.setText(
  "Nguồn: outputs/timeseries/run_info.json; outputs/timeseries/tables/forecast_metrics.csv; outputs/timeseries/tables/baseline_metrics.csv; outputs/timeseries/tables/diagnostics.csv. Health Index forecast là phân tích phụ, không thay thế RUL.",
);

// Slide 9: repurpose the old validation slide for the new warning evidence and current review proof.
const slide9 = presentation.slides.getItem(8);
slide9.shapes.deleteAll();
for (const image of [...slide9.images.items]) image.delete();
addText(slide9, "Cảnh báo sớm, benchmark và tái lập", { left: 64, top: 38, width: 1150, height: 50 }, {
  fontSize: 32, color: "#0F243E", bold: true,
});
addText(slide9, "Bằng chứng mô tả trên selected-model OOF và Pandas in-memory", { left: 66, top: 88, width: 1140, height: 30 }, {
  fontSize: 16, color: "#52606D",
});
slide9.images.add({
  blob: await readBytes("outputs/modeling/figures/warning_trajectory.png"),
  contentType: "image/png",
  alt: "Train OOF early-warning trajectory for engine 1",
  fit: "contain",
  position: { left: 64, top: 145, width: 700, height: 394 },
});
addText(slide9, "Ngưỡng", { left: 805, top: 150, width: 120, height: 28 }, { fontSize: 19, color: "#1F7A8C", bold: true });
addText(slide9, "RUL dự đoán ≤30 cycles", { left: 805, top: 185, width: 380, height: 36 }, { fontSize: 24, color: "#0F243E", bold: true });
addText(slide9, "Timing theo engine", { left: 805, top: 245, width: 250, height: 28 }, { fontSize: 19, color: "#1F7A8C", bold: true });
addText(slide9, "44 trước ngưỡng\n7 đúng ngưỡng\n49 sau ngưỡng nhưng trước failure", { left: 805, top: 280, width: 390, height: 90 }, { fontSize: 20, color: "#16202A", lineSpacing: 1.08 });
addText(slide9, "Lead time (min / Q1 / median / Q3 / max)", { left: 805, top: 385, width: 390, height: 26 }, { fontSize: 17, color: "#52606D" });
addText(slide9, "19 / 26 / 30 / 35 / 55 cycles", { left: 805, top: 414, width: 390, height: 30 }, { fontSize: 22, color: "#F28E2B", bold: true });
addText(slide9, "OOF đã dùng để chọn model, nên đây là evidence mô tả nội bộ. False-alarm rate không xác định vì mọi train trajectory đều có failure event.", { left: 805, top: 465, width: 390, height: 76 }, { fontSize: 16, color: "#52606D", lineSpacing: 1.05 });
addText(slide9, "Fresh review: 18 bảng · 5 hình · 151 tests. Benchmark Pandas in-memory: 3.372.700 rows ở scale 100× trong 1,67 s; chưa có PySpark timing.", { left: 64, top: 570, width: 1140, height: 32 }, { fontSize: 16, color: "#16202A", autoFit: "shrinkText" });
addText(slide9, "NASA C-MAPSS FD001 · Phan Nguyễn Mai Phương · 2026", { left: 64, top: 684, width: 900, height: 18 }, { fontSize: 11, color: "#52606D" });
addText(slide9, "09", { left: 1168, top: 684, width: 48, height: 18 }, { fontSize: 11, color: "#52606D", bold: true, alignment: "right" });
slide9.speakerNotes.textFrame.setText(
  "Nguồn: outputs/modeling/tables/warning_summary.json; outputs/modeling/tables/warning_events.csv; docs/product/warning-evaluation.md; outputs/etl/benchmark.json; tmp/e2e-review-20260924/validation.json. Ngưỡng RUL ≤30 là giả định minh họa. OOF đã dùng trong model selection nên warning chỉ là evidence mô tả nội bộ; false-alarm rate không được suy ra là 0%.",
);

// Slide 10: make the limitations and next research gate current.
updateText("sh/ydcnatkn", "FD001 chỉ có một điều kiện vận hành và một dạng lỗi.\n\nHealth Index forecast không phải RUL forecast.\n\nWarning dùng OOF đã chọn model, chưa phải đánh giá độc lập hoặc maintenance policy.\n\nChưa có cost, downtime hoặc intervention records.", {
  typeface: "Arial", fontSize: 18, color: "#16202A", alignment: "left", autoFit: "shrinkText", lineSpacing: 1.02,
});
updateText("sh/725onyl4", "slides/cmapss_fd001_presentation.pptx\nhandoff/FINAL_HANDOFF.md\nreport/main.pdf\nsrc/cmapss/ và tests/", {
  typeface: "Arial", fontSize: 20, color: "#16202A", alignment: "left", autoFit: "shrinkText", lineSpacing: 1.05,
});
updateText("sh/m1cnetkj", "Tiếp theo: sensitivity window 10/20/30 và outer warning evaluation.", {
  typeface: "Arial", fontSize: 17, color: "#F28E2B", bold: true, alignment: "left", autoFit: "shrinkText",
});
byId("nt/m90b6t0r").textFrame.setText(
  "Nguồn: data/CONTRACTS.md; outputs/modeling/tables/warning_summary.json; docs/product/warning-evaluation.md; report/sections/discussion.tex; handoff/FINAL_HANDOFF.md.",
);

await fs.mkdir(buildDir, { recursive: true });
await fs.mkdir(previewDir, { recursive: true });
await (await PresentationFile.exportPptx(presentation)).save(candidatePath);

for (let index = 0; index < presentation.slides.items.length; index += 1) {
  const slide = presentation.slides.getItem(index);
  const png = await slide.export({ format: "png", scale: 2 });
  await fs.writeFile(path.join(previewDir, `slide-${String(index + 1).padStart(2, "0")}.png`), new Uint8Array(await png.arrayBuffer()));
  const layout = await slide.export({ format: "layout" });
  await fs.writeFile(path.join(previewDir, `slide-${String(index + 1).padStart(2, "0")}.layout.json`), await layout.text(), "utf8");
}
const montage = await presentation.export({ format: "webp", montage: true, scale: 1 });
await fs.writeFile(path.join(buildDir, "deck-candidate-montage.webp"), new Uint8Array(await montage.arrayBuffer()));
const snapshot = await presentation.inspect({ kind: "slide,textbox,shape,image,table,notes", maxChars: 30000 });
await fs.writeFile(path.join(buildDir, "deck-candidate-inspect.ndjson"), snapshot.ndjson, "utf8");
