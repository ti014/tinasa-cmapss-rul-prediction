# HGB multi-scale feature study

**Phan Nguyễn Mai Phương — MSHV 25002711**  
**Run:** `20261006_study02` · **Status:** completed

## Kết luận

Study đủ protocol không đạt promotion gate. Outer `selected_pipeline` fallback
về `hgb_baseline169`; không thay published model/registry.

`hgb_multiscale10_30` giảm trajectory RMSE điểm từ **29,806846** xuống
**29,480160**, nhưng 95% CI của delta là **[-0,688575; 0,059603]**. Endpoint
MAE tăng từ **9,891336** lên **9,987586** (CI upper **0,436050**), NASA tăng
từ **6,883768** lên **7,566429** (CI upper **3,074927**), on-time giảm từ
**0,94** xuống **0,91** và early-alarm burden tăng từ **0,052159** lên
**0,052726**. Vì vậy `promotion_gate_met=false`.

`hgb_initial_profile42` kém hơn rõ rệt: trajectory RMSE **33,128001** và
endpoint MAE **13,172342**. Final-inner chọn multi-scale để lưu artifact tái
tạo, nhưng outer decision đã freeze trước test và vẫn giữ baseline.

## Thiết kế

Ba arm dùng cùng panel `(split, engine_id, cycle)`, raw RUL, engine-total-one
weights, 5 outer × 3 inner GroupKFold theo engine, seed 42 và HGB 200
iterations, learning rate 0,05, 31 leaves, `min_samples_leaf=20`, không early
stopping:

- `hgb_baseline169`: 169 features;
- `hgb_multiscale10_30`: baseline + 126 rolling-window-10 features (295);
- `hgb_initial_profile42`: baseline + 42 mean/sample-std của 30 cycles đầu (211).

Tổng **72 fits**: 69 CV records và 3 final models. Warning policy được
calibrate trên inner OOF; paired engine bootstrap dùng 10.000 resamples và
promotion controls đã đóng băng trong [protocol](../../../../experiments/MULTISCALE_FEATURE_PROTOCOL.md).

## Test reporting

Test chỉ đọc sau khi final models, predictions và replay receipt được lưu.
Multi-scale có RMSE test **19,916635** so với baseline **20,933433**, nhưng
đây là reporting-only và không thay đổi promotion decision.

## Kiểm chứng

[Independent verification](verification.json) đạt **all_checks_passed=true**,
gồm baseline feature/prediction parity với HGB source, provenance, 72 fit
records, selection, paired bootstrap và exact model replay. Verifier không fit
model và không đọc raw test labels. Smoke run chỉ là engineering evidence.

Artifacts chính: [aggregate metrics](aggregate_metrics.csv), [paired
comparison](paired_comparison.csv), [test metrics](test_metrics.csv), [decision](decision.json),
[run info](run_info.json), [final prediction receipt](final_predictions_receipt.json).
