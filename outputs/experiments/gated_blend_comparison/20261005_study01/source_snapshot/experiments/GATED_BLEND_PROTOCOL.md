# Protocol phối hợp hai HGB theo RUL dự đoán

**Phan Nguyễn Mai Phương — MSHV 25002711**  
**Ngày khai báo:** 05/10/2026  
**Status:** PLANNED, trước khi đánh giá grid mới trên dữ liệu thực nghiệm.

## Giả thuyết và phạm vi

HGB không cắt nhãn tốt hơn ở phần còn xa hỏng; HGB cap125 tốt hơn ở một số vùng gần hỏng. Một trọng số phụ thuộc RUL do HGB không cắt nhãn dự đoán có thể giữ lợi thế toàn quỹ đạo và cải thiện endpoint. Đây là giả thuyết cần kiểm chứng, chưa phải kết quả.

Kết quả chẩn đoán trước đó cho thấy trộn cố định 10%, 20%, 30% cap125 làm RMSE final-inner tăng từ 28,8350 lên 28,8709, 29,0915, 29,4927 dù endpoint NASA giảm. Grid mới kiểm tra trọng số thay đổi theo dự đoán. Grid không sử dụng RUL thật, lifetime, future sensor rows hoặc vùng RUL thật để quyết định trọng số.

Study là phân tích bổ sung trên engine folds và test từng được xem. Không gọi đây là xác nhận độc lập hoặc kết quả mới từ 48 lượt huấn luyện lại.

## Hai model thành phần và dữ liệu

Tái dùng đúng hai HGB references từ `outputs/experiments/lightgbm_comparison/20261005_study01`: uncapped và cap125. Mỗi HGB có 200 boosting iterations, learning rate 0,05, tối đa 31 leaves, min samples leaf 20, squared error, L2 bằng 0 và không early stopping. Cùng 169 causal history features, cửa sổ 30, engine-total-one sample weights và raw-RUL evaluation.

Nguồn đã được verifier kiểm tra. Runner mới phải xác nhận source/input/lock identity, hashes, cache coverage, raw truth, engine separation và receipt của nguồn trước khi sử dụng. Số lượt fit tái dùng là **48**: 30 outer-inner, 10 outer, 6 final-inner và 2 final models; **fresh fits = 0**. Không tái dùng lựa chọn LightGBM, test scores hoặc nhãn test để chọn gate.

Chia dữ liệu giữ nguyên 5 outer folds theo động cơ và 3 inner GroupKFold folds. Mỗi outer gate/policy chỉ được chọn từ OOF predictions của các engine trong outer fitting fold, rồi mới áp dụng cho outer validation engines. Final gate chọn từ final-inner OOF của toàn bộ training engines.

## Công thức và grid đóng băng

Gọi `u` là dự đoán HGB uncapped, `c` là dự đoán HGB cap125, đều ở đơn vị chu kỳ nguyên bản. Với `L < U` và `0 < M <= 1`:

```text
r = clip((U - u) / (U - L), 0, 1)
w = M * r² * (3 - 2r)
prediction = (1 - w) * u + w * c
```

Implementation clip `u` vào `[L,U]` trước khi trừ để tránh overflow của đầu vào hữu hạn rất lớn. Trọng số giảm liên tục theo `u`, bằng `M` khi `u <= L` và bằng 0 khi `u >= U`. Dự đoán nằm giữa hai dự đoán thành phần. Không suy ra đường dự đoán giảm đơn điệu theo thời gian; warning retractions vẫn phải được đo.

Grid gồm đúng **16 GateSpec**:

- `lower_rul`: `{60, 90}`;
- `upper_rul`: `{125, 150}`;
- `max_capped_weight`: `{0.25, 0.5, 0.75, 1.0}`.

Baseline/fallback là HGB uncapped; HGB cap125 là reference báo cáo riêng. Ví dụ `L=60, U=150, M=0.5`: khi `u` bằng 60, 105, 150 thì trọng số cap125 là 0,5; 0,25; 0. Trọng số không phụ thuộc nhãn thật.

## Inner selection và warning policy

Từng gate được đánh giá trên cùng các inner OOF rows ghép chính xác theo `(split, engine_id, cycle)`. Truth và inner/outer fold membership của hai thành phần phải trùng. Metric tính lại từ dự đoán mới, không tái dùng error columns cũ.

Calibrate warning policy riêng cho mỗi gate trên inner predictions với threshold `{30,35,40,45,50}`, persistence `{1,3,5}`, lead requirement 30 và target on-time 0,9; dùng cùng rule calibration đã kiểm trong audit study. HGB references giữ đúng policy đã được chọn từ inner predictions ở từng stage của source run; verifier tái tính policy theo cùng calibration rule. Không giả định mọi stage đều là threshold 40, persistence 1: HGB uncapped outer fold 4 của nguồn dùng threshold 40, persistence 3.

Gate feasible khi endpoint mean-engine MAE/NASA và early-alarm burden không lớn hơn HGB uncapped, đồng thời on-time rate không thấp hơn HGB uncapped. Mọi margins bằng 0. Chọn gate feasible có pooled trajectory RMSE thấp nhất, tie-break endpoint MAE rồi candidate ID. Chỉ dùng nó làm selected pipeline nếu trajectory RMSE thấp hơn HGB uncapped; nếu không, fallback HGB uncapped.

`gated_blend` lưu gate feasible tốt nhất hoặc diagnostic winner có RMSE thấp nhất nếu không có gate feasible, kèm flag rõ ràng. `selected_pipeline` lưu đúng lựa chọn đủ điều kiện hoặc fallback. Hai tên không được đánh đồng. Không chọn lại từ outer metrics.

## Outer readout và promotion gate

Các procedures được báo cáo: `hgb_uncapped`, `hgb_cap125`, `gated_blend`, `selected_pipeline`. Endpoint panel hồi cứu tại 50%, 70%, 90% lifetime; toàn quỹ đạo dùng mọi row đủ lịch sử. Scoring luôn dùng raw truth. Báo cáo thêm lỗi theo vùng và warning transitions/retractions.

So sánh paired engine bootstrap **10.000** lần, seed 42, conditional trên fixed outer predictions. Mỗi sampled engine giữ multiplicity; pooled RMSE tính từ tổng SSE và tổng số rows, không lấy trung bình RMSE engine/fold.

Promotion gate của **selected_pipeline so với HGB uncapped** giữ nguyên:

- upper 95% CI của delta trajectory RMSE `< 0`;
- upper CI của delta endpoint mean-engine MAE, endpoint mean-engine NASA, early-alarm burden `<= 0`;
- lower CI của delta on-time rate `>= 0`.

So sánh với cap125 chỉ reporting. Không nới gate, thay grid hoặc chọn lại sau outer readout. CI không bao gồm refitting/seed variability hoặc hiệu chỉnh nhiều đối chứng.

## Provenance, inference và test ordering

Trước selection, đóng băng source mới và import dependencies, protocol, config/grid, source-run identity và digests của mọi source artifact sử dụng; snapshot đủ để kiểm tra về sau. Dùng run ID mới; không ghi đè run đã có.

Lưu final gate/policy, hai base models, metadata/checksums và dự đoán test của các final procedures trước khi đọc raw test labels. Reload model và tái tạo base predictions và blended predictions chính xác từ frozen test features. Chỉ sau replay receipt hợp lệ mới mở test labels để reporting; outer decision được đóng băng trước test và giữ nguyên sau test. Verifier độc lập không fit model hoặc đọc raw test labels.

Smoke có namespace riêng và không thể đạt scientific promotion. Namespace đầy đủ: `outputs/experiments/gated_blend_comparison/20261005_study01`. Candidate không đạt thì không chọn để vận hành. Không tự thay published model/registry, commit hoặc push; không yêu cầu rerun Linux cho ứng viên chưa đạt.

## Module/package

```text
src/cmapss/modeling/gated_blend/
  contracts.py     # GateSpec và explicit grid builder
  gating.py        # prediction-only weights và exact paired alignment
  config.py        # fixed study/grid configuration
  analysis.py      # inner selection, warning controls, paired readouts
  inference.py     # saved component models và frozen gate inference
  runner.py        # source verification, nested derivation, test ordering
scripts/
  run_gated_blend_study.py
  verify_gated_blend_study.py
```

Weighted averaging tham khảo [scikit-learn VotingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingRegressor.html). Quy tắc smoothstep phụ thuộc prediction trong study này là giả thuyết riêng của project; tài liệu đó không chứng minh hiệu quả của gate. Engine-grouped evaluation tham khảo [scikit-learn Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html).
