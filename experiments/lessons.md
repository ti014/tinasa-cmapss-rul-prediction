# Bài học từ thực nghiệm RUL

**Phan Nguyễn Mai Phương — MSHV 25002711**

- 27/09/2026: Giảm MAE RUL thấp có thể làm tăng MAE toàn bộ. Ví dụ hgb_08
  giảm khoảng 9,75% low-RUL MAE trong inner CV, nhưng tăng overall MAE 0,3233.
  Nguồn: `20260928_rebuild/final_selection_metrics.csv`.
- Window ngắn không tự động tốt hơn. Window 30 được chọn ở cả năm outer
  runs và final selection trong đợt đã xét; không suy rộng ngoài search này.
- Quy trình chọn cấu hình thắng trong inner CV vẫn có thể kém hơn baseline
  ở outer evaluation. Cảnh báo trễ tăng từ 49 lên 53/100 engines trong đợt này.
- Khi muốn chấp nhận đánh đổi, chốt tiêu chí mới trước đợt tiếp theo. Không
  đổi luật sau khi xem outer/test rồi gọi đó là xác nhận độc lập.
- Cache cần gắn với config, split, input và source. Dry-run đã chứng minh
  có thể dùng lại 30 fits và từ chối cấu hình thay đổi.

- 05/10/2026: Cùng source, raw inputs, versions và seed chưa bảo đảm feature
  values hoặc HGB predictions trùng tuyệt đối giữa Linux ARM và Windows x86.
  Rerun9 fits giữ cùng thứ tự metrics, nhưng rolling features lệch tối đa
  7.105427357601002e-15. Một baseline fold chạy trên Windows với frozen Linux
  features tái tạo đúng5.878 predictions. Kiểm tra inputs/folds/features trước
  khi quy sai khác cho model; báo riêng exact vector equality và gần nhau về
  metrics. Nguồn: `windows_linux_comparison/20261005/README.md`.

- 05/10/2026: XGBoost không tự cải thiện HGB khi giữ nguyên 169 history
  features. Trong grid đã đóng băng, outer trajectory RMSE uncapped là
  29,975 so với HGB 29,807; endpoint MAE tăng từ 9,891 lên 11,626 chu kỳ.
  CI RMSE đi qua 0. Kết luận chỉ áp dụng cho grid/protocol này.
- Cắt nhãn 125 cải thiện test endpoint nhưng làm tăng trajectory RMSE ở
  HGB. Ví dụ test MAE cap125 là 9,376, uncapped là 13,753; outer trajectory
  RMSE lần lượt 36,208 và 29,807. Giữ riêng mục tiêu và references.
- Bootstrap trên fixed outer predictions là conditional CI, không bao gồm
  refitting/seed variability. Folds/test đã xem trước đây không tạo xác nhận
  độc lập chỉ vì chạy lại. Nguồn: `tree_comparison/20261005_full/README.md`.

- 05/10/2026: TCN pilot trên raw sensor sequences cần sequence builder,
  fold-local scaler và causal convolution riêng; không thể chỉ đổi estimator
  trong pipeline tree. Với 8 epochs, TCN uncapped RMSE khoảng36,60, cao hơn
  HGB uncapped29,81. Đây là tín hiệu sàng lọc, chưa phải full nested result;
  ngân sách train phải được ghi rõ trước khi kết luận.
- 05/10/2026: LightGBM pilot có thể nhỉnh hơn HGB ở RMSE khi đổi số cây,
  nhưng full matched nested run không tái tạo lợi thế. LightGBM uncapped pooled
  RMSE29,842 so với HGB29,807; delta+0,034, CI đi qua0. Không chọn theo pilot
  hoặc một metric riêng.

### Hiệu chỉnh ghi chú pilot ngày 05/10

- TCN 8 epochs chỉ sàng lọc pipeline và có thể underfit; chưa đủ bằng chứng
  loại TCN. Raw sensor input khác age/history của HGB169 nên không gán delta
  cho riêng kiến trúc. Lượt50epochs đã dừng vì harness chưa đủ safeguards.
- LightGBM exploratory summary gộp hai settings; không dùng nó như selected
  nested metric. Kết luận accepted/rejected phải dựa study01 có endpoint panel,
  warning controls và paired gate, không dựa endpointRUL0 của script thăm dò.

### Kết luận từ LightGBM study01 ngày 05/10/2026

- LightGBM uncapped giảm pooled trajectory RMSE từ29,806846 xuống29,653172,
  nhưng paired95%CI của delta[−0,479373;0,185828] đi qua0. Đây chưa phải bằng
  chứng cải thiện. Endpoint NASA tăng từ6,883768 lên20,988312; không chọn ứng viên.
- LightGBM cap125 có CI RMSE tốt hơn HGB cùng cap, nhưng endpoint MAE/NASA
  chưa đạt safeguards. Ví dụ endpoint NASA3,859042 so với3,683556; một metric
  tốt hơn không đủ đổi model theo mục tiêu đã chốt.
- Inner procedure selection ở cả5outer folds và final fit giữ HGB uncapped.
  Test reporting thuận lợi hơn ở một vài metric không thay được gate đã đóng băng.
- Study01 mới là readout đầy đủ của LightGBM; kết quả exploratory trước đó
  không được dùng như accepted nested gate. TCN8epochs vẫn chỉ là pilot, có thể
  underfit và khác thông tin đầu vào của HGB history169; chưa loại kiến trúc.
- Nguồn: [study01](../outputs/experiments/lightgbm_comparison/20261005_study01/README.md)
  và [verifier](../outputs/experiments/lightgbm_comparison/20261005_study01/verification.json).

### Kết luận từ gated blend study02 ngày 05/10/2026

- Prediction-only gating giữ lợi thế phần xa hỏng của HGB uncapped, nhưng grid hiện tại chưa tạo selected pipeline vượt gate. Ví dụ RMSE29,802876 so với29,806846, CI delta[−0,014889;0,005437] vẫn đi qua0; endpoint và burden upper CI cũng dương.
- Diagnostic blend có RMSE29,766509 và endpoint MAE9,634579 tốt hơn baseline, nhưng early-alarm burden tăng từ0,052159 lên0,053870. Chọn theo diagnostic outer scores sẽ thay selection rule sau kết quả; readout selected pipeline phải được giữ riêng.
- Chỉ1/5 outer folds chọn blend đủ controls, dù final-inner chọn blend. Final-inner cải thiện hoặc test scoring tốt hơn không tự thông qua outer promotion gate. Kết thúc nhánh hiện tại với kết luận không chọn; published model/registry giữ nguyên.
- Atomic replace trên Windows cần retry có giới hạn cho reader-handle conflicts; lỗi kéo dài phải phát sinh và giữ receipt cũ. Snapshot với đường dẫn gốc dài cần tên ngắn và mapping kiểm hash/coverage/collision.
- Nguồn: [study02](../outputs/experiments/gated_blend_comparison/20261005_study02/README.md), [verification](../outputs/experiments/gated_blend_comparison/20261005_study02/verification.json).

- 07/10/2026: Similarity-based retrieval với block-mean sensor30cycles và k={5,10}
  thua HGB169 trên cả3 GroupKFold inner folds: RMSE37,173/36,543 so với HGB28,835;
  endpoint MAE20,890/20,844 so với10,551. Dừng nhánh theo điểm dừng đã đóng băng;
  không mở thêm DTW/PCA/k-grid chỉ để cứu kết quả. Nguồn:
  `outputs/experiments/similarity_pilot/20261007_pilot02/decision.csv`.
