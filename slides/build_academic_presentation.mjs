import fs from "node:fs/promises";
import path from "node:path";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const root = "D:/IUH/BigData/Final_Project";
const build = path.join(root, "tmp", "slides-rul-20260927");
const candidate = path.join(build, "candidate.pptx");
const preview = path.join(build, "renders");
await fs.mkdir(preview, { recursive: true });

const ppt = Presentation.create({ slideSize: { width: 1280, height: 720 } });
const C = { navy: "#0F243E", blue: "#245B8F", teal: "#1F7A8C", orange: "#D56B11", ink: "#16202A", muted: "#52606D", white: "#FFFFFF", pale: "#EAF4F5" };

function text(slide, value, x, y, w, h, size = 22, color = C.ink, bold = false, options = {}) {
  const s = slide.shapes.add({ geometry: "textbox", position: { left: x, top: y, width: w, height: h }, fill: "none", line: { fill: "none", width: 0 } });
  s.text = value;
  s.text.style = { typeface: "Arial", fontSize: size, color, bold, alignment: options.align ?? "left", verticalAlignment: "top", autoFit: "shrinkText", lineSpacing: options.lineSpacing ?? 1.08, insets: { left: 0, right: 0, top: 0, bottom: 0 } };
  return s;
}

function base(title, subtitle, number, source, notes) {
  const slide = ppt.slides.add();
  slide.background.fill = C.white;
  text(slide, title, 64, 38, 1150, 52, 34, C.navy, true);
  text(slide, subtitle, 66, 96, 1140, 35, 19, C.muted);
  text(slide, `Nguồn: ${source}`, 64, 656, 1080, 30, 12, C.muted);
  text(slide, String(number).padStart(2, "0"), 1165, 671, 50, 25, 14, C.muted, true, { align: "right" });
  slide.speakerNotes.textFrame.setText(notes);
  return slide;
}

function block(slide, label, body, x, y, w, bodyHeight = 100) {
  text(slide, label, x, y, w, 32, 20, C.teal, true);
  text(slide, body, x, y + 40, w, bodyHeight, 23, C.ink);
}

async function picture(slide, relative, alt, x, y, w, h) {
  slide.images.add({ blob: new Uint8Array(await fs.readFile(path.join(root, relative))), contentType: "image/png", alt, fit: "contain", position: { left: x, top: y, width: w, height: h } });
}

function table(slide, values, x, y, w, h, font = 19) {
  const t = slide.tables.add({ rows: values.length, columns: values[0].length, left: x, top: y, width: w, height: h, values });
  t.styleOptions = { headerRow: true, bandedRows: true };
  for (let r = 0; r < values.length; r++) for (let c = 0; c < values[0].length; c++) {
    const cell = t.getCell(r, c);
    cell.text.style = { typeface: "Arial", fontSize: font, color: r === 0 ? C.white : C.ink, bold: r === 0 };
    if (r === 0) cell.fill = C.navy;
  }
  return t;
}

// 1. Cover.
{
  const s = ppt.slides.add();
  s.background.fill = C.navy;
  text(s, "Dự đoán tuổi thọ còn lại và cảnh báo sớm", 80, 135, 1120, 120, 48, C.white, true);
  text(s, "Phân tích dữ liệu động cơ mô phỏng NASA C-MAPSS FD001–FD004", 82, 280, 1090, 55, 29, "#BFE7EA");
  text(s, "Báo cáo cuối kỳ môn Phân tích dữ liệu lớn", 82, 390, 1000, 40, 23, C.white);
  text(s, "Phan Nguyễn Mai Phương | MSHV 25002711", 82, 590, 900, 38, 22, C.white);
  s.speakerNotes.textFrame.setText("Mục tiêu chính là Remaining Useful Life (RUL, tuổi thọ hữu ích còn lại) theo cycle. Phần cảnh báo dùng kết quả dự đoán để mô tả thời điểm phát hiện RUL thấp. Nguồn: report/main.pdf.");
}

// 2. Problem and concrete target.
{
  const s = base("Bài toán nghiên cứu", "Dự đoán số cycle còn lại trước khi một engine đi đến failure", 2, "report/sections/problem-data.tex; data/CONTRACTS.md", "Giải thích RUL bằng ví dụ. Trong train, RUL tại cycle t bằng cycle cuối của engine trừ t. Test bị cắt sớm; nhãn NASA chỉ dùng ở điểm quan sát cuối để đánh giá. Không quy đổi cycle ra giờ.");
  block(s, "Đầu vào", "Chuỗi đo của 21 sensor theo từng cycle và từng engine.", 76, 175, 500, 100);
  block(s, "Đầu ra", "Remaining Useful Life (RUL): số cycle còn lại tại thời điểm dự đoán.", 670, 175, 510, 115);
  text(s, "Ví dụ: engine kết thúc ở cycle 200. Tại cycle 170, RUL thật bằng 30 cycles.", 76, 405, 1100, 80, 27, C.navy, true);
  text(s, "Câu hỏi chính: đặc trưng từ 30 cycle gần nhất dự đoán RUL tốt đến đâu trên engine chưa thấy?", 76, 520, 1100, 75, 23, C.ink);
}

// 3. Data contract and label split.
{
  const s = base("Dữ liệu C-MAPSS và ranh giới nhãn", "FD001 là protocol chính; FD002–FD004 dùng cho kiểm tra chuyển miền", 3, "data/raw/DATA_SOURCE.md; outputs/etl/run_info.json; data/CONTRACTS.md", "Kho raw có đủ bốn subset C-MAPSS. Phần chính của báo cáo dùng FD001 với một operating condition và một fault mode. Ma trận bổ sung dùng FD002–FD004 để kiểm tra transfer. Trong train, nhãn RUL suy từ cycle cuối. Nhãn test của từng subset chỉ đọc sau khi ghi final predictions.");
  table(s, [["Tập", "Engine", "Dòng quan sát", "Vai trò"], ["Train", "100", "20.631", "Học và chọn model"], ["Test", "100", "13.096", "Đánh giá cuối"]], 78, 175, 1115, 235, 19);
  text(s, "Train: RUL = cycle cuối của engine − cycle hiện tại", 82, 455, 1110, 52, 26, C.navy, true);
  text(s, "Test: mỗi engine có một dự đoán tại cycle cuối được quan sát; sau đó mới ghép nhãn NASA của subset tương ứng để chấm điểm.", 82, 525, 1110, 85, 23, C.ink);
}

// 4. Data flow.
{
  const s = base("Quy trình xử lý dữ liệu", "Mỗi bước ghi artifact để có thể kiểm tra và chạy lại", 4, "assets/diagrams/fd001-pipeline.png; handoff/DATA_WALKTHROUGH.md", "Đọc sơ đồ từ trái sang phải. Trong một subset, reader đọc train/test cùng schema, thêm split rồi union theo dòng. Silver giữ observations typed trong Parquet; SQLite chỉ có engine_summary và train_targets. Rolling features dùng lịch sử của từng engine, GroupKFold chọn model, final evaluation ghép nhãn test ở bước cuối. PySpark hiện tái hiện ETL local để parity; chưa có query engine trên Parquet, Airflow hoặc Power BI.");
  await picture(s, "assets/diagrams/fd001-pipeline.png", "Sơ đồ pipeline FD001", 58, 150, 1165, 390);
  text(s, "FD001: 33.727 observations → Silver Parquet → Gold summary/features. Nhãn test chỉ xuất hiện sau khi model đã tạo dự đoán cuối.", 75, 550, 1110, 74, 23, C.navy, true);
}

// 5. Statistical sensor analysis.
{
  const s = base("Sensor thay đổi ra sao theo vòng đời?", "So sánh 30 cycle đầu và 30 cycle cuối trong cùng một train engine", 5, "outputs/statistics/tables/sensor_inference.csv; report/sections/statistical-protocol.tex", "Mỗi engine đóng góp một cặp cho từng sensor. Tính mean đầu và mean cuối, rồi lấy chênh lệch. Bootstrap theo engine để có CI, Wilcoxon signed-rank và Benjamini–Hochberg cho 21 sensor. 15 sensor bác bỏ giả thuyết không sau BH; không được hiểu là bằng chứng nhân quả hoặc giá trị dự báo ngoài mẫu.");
  block(s, "Cách so sánh", "Với mỗi sensor: chênh lệch = trung bình 30 cycle cuối trừ trung bình 30 cycle đầu.", 75, 170, 1050, 100);
  block(s, "Cách kiểm tra", "Ghép cặp trong cùng engine; bootstrap lấy mẫu lại engine; Wilcoxon và hiệu chỉnh Benjamini–Hochberg cho 21 sensor.", 75, 325, 1050, 125);
  text(s, "Kết quả: 15 sensor có bằng chứng thay đổi theo protocol này.", 75, 530, 1080, 48, 27, C.navy, true);
  text(s, "Điều này mô tả khác biệt đầu/cuối vòng đời; chưa chứng minh sensor đó dự đoán tốt trên engine mới.", 75, 585, 1080, 42, 18, C.muted);
}

// 6. Feature representation.
{
  const s = base("Từ chuỗi sensor đến đặc trưng", "Một dự đoán dùng 30 cycle gần nhất của chính engine đó", 6, "outputs/features/run_info.json; handoff/DATA_WALKTHROUGH.md", "Window dài 30. Ví dụ tại cycle 170 dùng 141–170, không dùng 171. Với mỗi trong 21 sensor, tính last, mean, standard deviation, min, max, slope. Tổng 126 đặc trưng. Artifact hiện có 27.927 window. Observation và feature dùng cùng khóa; modeling join feature với train target one-to-one. Ba setting không đưa vào X của model RUL. Không nối window qua hai engine.");
  text(s, "Ví dụ tại cycle 170: chỉ đọc cycles 141–170", 77, 175, 1100, 58, 30, C.navy, true);
  table(s, [["Mỗi sensor", "Ý nghĩa ngắn"], ["Giá trị cuối", "Trạng thái tại cycle hiện tại"], ["Mean, độ lệch chuẩn", "Mức và biến thiên trong window"], ["Min, max, slope", "Biên và xu hướng tăng/giảm"]], 78, 270, 1110, 255, 19);
  text(s, "21 sensor × 6 phép tóm tắt = 126 đặc trưng; tạo 27.927 window hợp lệ.", 78, 560, 1110, 58, 25, C.teal, true);
}

// 7. Evaluation protocol.
{
  const s = base("Đánh giá theo engine", "Các window của một engine luôn nằm cùng phía của phép chia", 7, "outputs/modeling/tables/evaluation_protocol.csv; outputs/modeling/run_info.json", "Không chia ngẫu nhiên các window vì các cycle trong cùng engine có tương quan. Dùng 5-fold GroupKFold theo engine_id để chọn model bằng mean engine MAE. Sau khi chọn, fit lại trên toàn train và tạo một dự đoán cho mỗi test engine. Nhãn test chỉ dùng để chấm điểm cuối.");
  table(s, [["Giai đoạn", "Dữ liệu", "Mục đích"], ["5-fold GroupKFold", "100 train engines", "Chọn model bằng MAE trung bình theo engine"], ["Refit", "Toàn bộ train", "Tạo model cuối"], ["Final test", "100 test engines", "Một dự đoán ở cycle cuối mỗi engine"]], 75, 185, 1130, 300, 17);
  text(s, "Vì sao cần GroupKFold? Cùng một engine có nhiều window gần nhau; để chúng ở cả train và validation sẽ làm kết quả lạc quan.", 78, 535, 1110, 90, 22, C.navy, true);
}

// 8. Model selection.
{
  const s = base("Chọn mô hình dự đoán RUL", "So sánh ba cách dự đoán trên cùng 5 fold và cùng metric", 8, "outputs/modeling/tables/model_metrics.csv", "Median baseline là mốc hằng đơn giản. Ridge là hồi quy tuyến tính có regularization. HistGradientBoosting là mô hình cây boosting, nắm quan hệ phi tuyến. Cùng protocol, model boosting có mean engine MAE thấp nhất 20,2143. Đây là metric cross-validation, khác với 15,3490 trên final test.");
  table(s, [["Model", "Mean engine MAE (cycles)", "Diễn giải"], ["Median baseline", "46,9543", "Mốc dự đoán hằng"], ["Ridge", "23,9922", "Quan hệ tuyến tính"], ["HistGradientBoosting", "20,2143", "Được chọn"]], 80, 185, 1110, 300, 19);
  text(s, "HistGradientBoosting được chọn vì có MAE thấp nhất trong phép so sánh đã định trước.", 80, 535, 1100, 80, 26, C.navy, true);
}

// 9. Held-out result.
{
  const s = base("Kết quả trên 100 engine test", "Mỗi engine đóng góp một dự đoán tại cycle cuối được quan sát", 9, "outputs/modeling/tables/final_metrics.json; outputs/modeling/figures/final_true_vs_predicted.png", "Đọc hình: gần đường chéo nghĩa là dự đoán gần RUL thật. MAE 15,3490 cycles, RMSE 20,5413 cycles. Bootstrap lấy lại 100 engine có hoàn lại 10.000 lần, khoảng tin cậy 95% cho MAE 12,7286–18,0866. NASA score 1.513,2577 là score bất đối xứng, không có đơn vị cycle.");
  await picture(s, "outputs/modeling/figures/final_true_vs_predicted.png", "RUL thật và RUL dự đoán trên 100 test engines", 70, 160, 650, 420);
  block(s, "Sai số trung bình", "MAE = 15,3490 cycles", 765, 195, 405, 55);
  block(s, "Sai số lớn bị phạt mạnh hơn", "RMSE = 20,5413 cycles", 765, 300, 405, 55);
  block(s, "Độ bất định", "CI 95% của MAE: [12,7286; 18,0866]", 765, 405, 420, 80);
  text(s, "NASA score: 1.513,2577 (không đo bằng cycle)", 765, 550, 420, 48, 19, C.muted);
}

// 10. Error analysis.
{
  const s = base("Mô hình sai ở đâu?", "Sai số lớn nhất xảy ra khi RUL dự đoán cao hơn RUL thật", 10, "outputs/modeling/figures/final_residuals_by_rul.png; report/sections/current-results.tex", "74 trên 100 dự đoán cao hơn RUL thật và 26 thấp hơn. Median absolute error là 13,3654 cycles. Engine 23 có RUL thật 113, dự đoán 170,4768, sai 57,4768 cycles. Đây là trường hợp minh họa sai số lớn; không loại khỏi metric. Hình thể hiện prediction minus true theo RUL thật.");
  await picture(s, "outputs/modeling/figures/final_residuals_by_rul.png", "Sai số dự đoán theo RUL thật", 65, 160, 685, 420);
  block(s, "Phân bố hướng sai số", "74 dự đoán cao hơn RUL thật; 26 dự đoán thấp hơn.", 790, 185, 400, 100);
  block(s, "Ví dụ cần lưu ý", "Engine 23: RUL thật 113, dự đoán 170,48. Sai số 57,48 cycles.", 790, 330, 400, 145);
  text(s, "Metric trung bình không thay thế việc xem các trường hợp sai lớn.", 790, 530, 400, 75, 22, C.navy, true);
}

// 11. Secondary Health Index analysis.
{
  const s = base("Phân tích phụ: Health Index và ARIMA", "Health Index tóm tắt sensor; giá trị của nó không đo bằng cycle", 11, "outputs/timeseries/run_info.json; outputs/timeseries/tables/baseline_metrics.csv; outputs/timeseries/tables/forecast_metrics.csv", "StandardScaler và PCA fit trên 90 train engine không được chọn; 10 engine còn lại chỉ transform. ARIMA fit 70% đầu và forecast 30% cuối mỗi chuỗi. So sánh cùng holdout với drift và last-value. Trung bình MAE/RMSE của ARIMA thấp hơn, nhưng chỉ 10 engine và coverage 21,43–97,03%, vì vậy đây là so sánh mô tả.");
  await picture(s, "outputs/timeseries/figures/health_index_selected_engines.png", "Quỹ đạo Health Index trên 10 engine train được chọn", 55, 155, 670, 345);
  table(s, [["Dự báo", "MAE", "RMSE"], ["ARIMA", "0,4110", "0,5027"], ["Drift", "0,7469", "0,9128"], ["Last-value", "0,9478", "1,1553"]], 750, 175, 450, 285, 18);
  text(s, "PCA fit trên 90 engine khác; ARIMA học 70% đầu, đánh giá 30% cuối của 10 engine.", 75, 530, 1120, 55, 21, C.navy, true);
  text(s, "Kết quả này mô tả Health Index, không suy ra RUL hoặc thời điểm bảo trì.", 75, 590, 1120, 35, 18, C.muted);
}

// 12. Early warning.
{
  const s = base("Cảnh báo sớm từ RUL dự đoán", "Cảnh báo tại cycle đầu tiên mà RUL dự đoán không quá 30", 12, "outputs/modeling/tables/warning_summary.json; outputs/modeling/figures/warning_trajectory.png", "Warning dùng selected-model out-of-fold predictions trên train. Mỗi engine được dự đoán bởi model không fit engine đó, nhưng cùng OOF đã dùng để chọn model. Timing 44 trước mốc RUL thật 30, 7 đúng mốc, 49 sau mốc nhưng trước failure. Lead time min/Q1/median/Q3/max 19/26/30/35/55 cycles. False alarm rate không xác định vì mọi trajectory train đều đi đến failure.");
  await picture(s, "outputs/modeling/figures/warning_trajectory.png", "Ví dụ quỹ đạo cảnh báo engine 1", 55, 160, 705, 390);
  block(s, "Thời điểm cảnh báo đầu", "44 trước ngưỡng; 7 đúng ngưỡng; 49 sau ngưỡng.", 790, 175, 390, 120);
  block(s, "Thời gian đến failure", "Trung vị 30 cycles; miền 19–55 cycles.", 790, 340, 390, 95);
  text(s, "OOF đã dùng chọn model: kết quả chỉ mô tả nội bộ. Chưa đánh giá utility bảo trì.", 790, 505, 390, 115, 19, C.muted);
}

// 13. ETL, parity and reproducibility.
{
  const s = base("ETL và khả năng tái lập", "Đối chiếu kết quả xử lý; đo runtime cục bộ bằng Pandas", 13, "outputs/etl/parity.json; outputs/etl/benchmark.json; tmp/e2e-review-20260924/validation.json", "Pandas/PySpark cho cùng kết quả tại ba boundary Silver/Gold, max absolute difference 0.0. Benchmark chỉ đo build_engine_summary in-memory bằng Pandas. Scale 100x có 3.372.700 rows, hai lần 1,6752 và 1,6732 giây; không bao gồm I/O và không có PySpark timing. Fresh E2E đối chiếu 18 bảng, 5 hình và raw hash; 175 tests pass.");
  block(s, "Độ đúng của ETL", "Pandas và PySpark khớp schema, khóa, số dòng và giá trị ở ba bảng Silver/Gold.", 75, 170, 1100, 100);
  block(s, "Thử nghiệm quy mô", "Pandas tổng hợp 3.372.700 dòng nhân bản trong bộ nhớ: 1,6752 và 1,6732 giây.", 75, 320, 1100, 105);
  block(s, "Tái lập", "Chạy lại từ raw: 18 bảng khớp nội dung, 5 hình byte-identical, raw hash không đổi; 175 tests pass.", 75, 475, 1100, 120);
}

// 14-16. Completed nested RUL improvement study.
{
  const s = base("Thiết kế thử nghiệm cải thiện RUL", "100 train engines, cùng 17.731 khóa từ cycle 30", 14, "experiments/README.md; 20260927_full/search_manifest.json", "Outer 5-fold được giữ từ baseline. Trong mỗi outer train, inner 3-fold chọn window, tham số và trọng số. Chỉ chấm outer validation sau khi chọn xong. Test đã được xem trước đợt này, không dùng để chọn lại cấu hình.");
  block(s, "Phạm vi đã chạy", "Window 10 / 20 / 30, 24 cấu hình HistGradientBoosting và 3 phương án trọng số RUL.", 75, 170, 1100, 90);
  block(s, "Tách lựa chọn và đánh giá", "5 outer folds đánh giá trên engine được giữ riêng. Mỗi outer train dùng 3 inner folds để chọn cấu hình.", 75, 315, 1100, 110);
  block(s, "Tiêu chí đã chốt", "Ưu tiên MAE khi RUL ≤ 30. MAE toàn bộ, tỷ lệ cảnh báo trễ và thời gian báo sớm phải không tăng.", 75, 490, 1100, 110);
}
{
  const s = base("Kết quả chưa đạt tiêu chí cải thiện", "Outer evaluation của cả quy trình chọn cấu hình", 15, "20260927_full/ablation_metrics.csv; paired_comparison.csv", "Các stage dùng cùng 100 train engines. Window 30 được chọn ở cả 5 vòng. Hai vòng chọn cấu hình HGB khác, ba vòng giữ baseline. Không vòng nào chọn thêm trọng số. Chênh lệch MAE RUL thấp +0,0054 cycles, CI [-0,0506;0,0639]. Tỷ lệ cảnh báo trễ tăng 4 điểm phần trăm, CI [1;8]. Các CI điều kiện trên predictions đã fit.");
  table(s, [["Phương án", "MAE toàn bộ", "MAE RUL ≤ 30", "Trễ / bỏ lỡ"], ["Baseline", "20,2143", "3,6676", "49 / 100"], ["Chọn window", "20,2143", "3,6676", "49 / 100"], ["Chỉnh tham số", "20,2459", "3,6730", "53 / 100"], ["Chọn trọng số", "20,2459", "3,6730", "53 / 100"]], 70, 170, 1140, 270, 21);
  text(s, "MAE RUL thấp thay đổi +0,0054 cycles, CI 95% [−0,0506; 0,0639].", 75, 485, 1110, 55, 23, C.navy, true);
  text(s, "Cấu hình cuối trùng baseline. Test MAE giữ nguyên 15,3490 cycles.", 75, 560, 1110, 60, 24, C.ink);
}
{
  const s = base("Giảm sai số có thể đi kèm đánh đổi", "Ví dụ trong inner CV trên toàn train, phục vụ chọn cấu hình", 16, "20260927_full/final_selection_metrics.csv", "Đây là ví dụ từ inner selection, không phải outer evaluation độc lập của ứng viên hgb_08. hgb_08 giảm MAE RUL thấp khoảng 9,75% nhưng MAE toàn bộ tăng 0,3233 cycles. Strong weights giảm low-RUL MAE và cảnh báo trễ nhưng tăng MAE toàn bộ và alarm burden. Luật đã chốt không cho phép tăng chỉ số kiểm soát. Không tuning tiếp theo test.");
  table(s, [["Ứng viên", "MAE RUL ≤ 30", "MAE toàn bộ", "Trễ / bỏ lỡ"], ["Baseline", "3,6148", "20,3613", "50 / 100"], ["hgb_08", "3,2625", "20,6846", "49 / 100"], ["Ưu tiên RUL thấp cao", "3,3901", "20,4072", "47 / 100"]], 70, 170, 1140, 230, 21);
  block(s, "Lý do chưa thay model", "hgb_08 giảm sai số RUL thấp khoảng 9,75% nhưng tăng MAE toàn bộ. Phương án trọng số cao còn tăng thời gian báo sớm.", 75, 450, 1100, 120);
  text(s, "Đợt tiếp theo cần chốt mức đánh đổi chấp nhận được trước khi thử lại.", 75, 605, 1110, 35, 21, C.navy, true);
}

// 17. Conclusion and limits.
{
  const s = base("Kết luận và giới hạn", "Kết quả chính trên FD001 và bằng chứng chuyển miền trong C-MAPSS", 17, "report/sections/conclusion.tex; outputs/cross_dataset/matrix.json", "Kết luận: HistGradientBoosting được chọn theo GroupKFold, final MAE 15,3490 cycles với window 30 trên FD001. Ma trận bổ sung dùng window 10 cho bốn subset. Transfer từ FD001 sang FD002–FD004 có MAE 60,3215; 47,3949; 58,5731 cycles. Cảnh báo OOF vẫn là mô tả nội bộ. Health Index forecast là phân tích phụ.");
  block(s, "Kết quả chính", "HistGradientBoosting đạt MAE 15,3490 cycles trên 100 test engines; CI 95% [12,7286; 18,0866].", 75, 170, 1100, 100);
  block(s, "Giới hạn cần nhớ", "Kết quả chính FD001 dùng window 30; ma trận transfer dùng window 10. Test chỉ đánh giá một điểm cắt mỗi engine. Đợt nested CV bổ sung chưa cải thiện các tiêu chí đã chốt.", 75, 330, 1100, 125);
  block(s, "Bước nghiên cứu tiếp", "Chốt mức đánh đổi cho vùng RUL thấp, thử đặc trưng mới trong training fold, rồi đánh giá bằng protocol mới đã công bố trước.", 75, 515, 1100, 95);
}

// 18. Cross-dataset transfer matrix.
{
  const s = base("Transfer giữa các subset C-MAPSS", "Window 10 cho phép giữ đủ test engines trong FD001–FD004", 18, "outputs/cross_dataset/matrix.json; outputs/cross_dataset/README.md", "Ma trận bổ sung chạy bốn evaluation nội bộ và ba phép transfer. Nội bộ: train và test cùng subset. Transfer: model fit trên toàn bộ train FD001 được giữ cố định khi dự đoán test subset khác. Window 10 khác window 30 của metric chính FD001, nên hai nhóm kết quả được trình bày riêng. Kết luận: dataset shift làm MAE tăng rõ rệt khi chuyển khỏi FD001.");
  table(s, [["Train → Test", "Loại", "MAE (cycles)", "RMSE (cycles)"], ["FD001 → FD001", "Nội bộ", "21,9258", "29,5936"], ["FD002 → FD002", "Nội bộ", "22,3556", "31,3077"], ["FD003 → FD003", "Nội bộ", "28,3255", "41,7606"], ["FD004 → FD004", "Nội bộ", "27,2218", "36,9311"], ["FD001 → FD002", "Transfer", "60,3215", "70,4603"], ["FD001 → FD003", "Transfer", "47,3949", "59,2140"], ["FD001 → FD004", "Transfer", "58,5731", "69,8148"]], 70, 145, 1140, 385, 16);
  text(s, "Model fit riêng trên FD001 không nên dùng trực tiếp cho subset khác nếu chưa retraining hoặc domain adaptation.", 75, 565, 1120, 60, 24, C.navy, true);
  text(s, "Các metric này dùng window 10; metric chính FD001 trên slide 9 dùng window 30.", 75, 625, 1120, 32, 17, C.muted);
}

await (await PresentationFile.exportPptx(ppt)).save(candidate);
for (let i = 0; i < ppt.slides.items.length; i++) {
  const slide = ppt.slides.getItem(i);
  const png = await slide.export({ format: "png", scale: 2 });
  await fs.writeFile(path.join(preview, `slide-${String(i + 1).padStart(2, "0")}.png`), new Uint8Array(await png.arrayBuffer()));
  const layout = await slide.export({ format: "layout" });
  await fs.writeFile(path.join(preview, `slide-${String(i + 1).padStart(2, "0")}.layout.json`), await layout.text(), "utf8");
}
const snapshot = await ppt.inspect({ kind: "slide,textbox,table,image,notes", maxChars: 50000 });
await fs.writeFile(path.join(build, "inspect.ndjson"), snapshot.ndjson, "utf8");
console.log(`Exported ${ppt.slides.items.length} slides to ${candidate}`);
