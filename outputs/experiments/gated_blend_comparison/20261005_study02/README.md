**Phan Nguyễn Mai Phương — MSHV 25002711**

# Kết quả phối hợp HGB theo RUL dự đoán

## Kết luận

**Không chọn gated blend để vận hành trong protocol này.** `selected_pipeline`, tức quy trình chọn blend đủ điều kiện bằng inner CV hoặc fallback về HGB uncapped, chỉ giảm trajectory RMSE từ **29,806846** xuống **29,802876**. Delta là −0,003970 chu kỳ, với paired 95% CI **[−0,014889; 0,005437]**. Chưa chứng minh được cải thiện; endpoint MAE/NASA và early-alarm burden cũng chưa đạt điều kiện không xấu hơn. `promotion_gate_met=false`.

`gated_blend` là readout chẩn đoán riêng: mỗi fold dùng gate feasible tốt nhất, hoặc gate có RMSE thấp nhất nếu không có gate feasible. Nó có RMSE và endpoint tốt hơn HGB uncapped, nhưng early-alarm burden tăng. Không dùng readout này thay thế `selected_pipeline` sau khi đã thấy outer metrics.

## Thiết lập và công thức

Lượt hoàn tất là `20261005_study02`, chạy trên Windows ngày 05/10/2026 trong **122,66 giây**. Study tái dùng **48 HGB fits**, gồm 30 outer-inner, 10 outer, 6 final-inner và 2 final models; **fresh fits = 0**. Hai HGB và predictions gốc lấy từ [LightGBM study01](../../lightgbm_comparison/20261005_study01/README.md), được kiểm source/input/lock/runtime identity, hashes, engine separation và coverage trước khi dùng.

Hai thành phần cùng 169 causal history features, cửa sổ 30, engine-total-one sample weights. Mỗi HGB có 200 boosting iterations, learning rate 0,05, tối đa 31 leaves, tối thiểu 20 samples/leaf và không early stopping. Một model dùng raw RUL labels; model còn lại cap125 chỉ ở fitting labels. Scoring luôn dùng raw truth.

Với dự đoán uncapped `u` và cap125 `c`:

```text
r = clip((U - u) / (U - L), 0, 1)
w = M * r² * (3 - 2r)
p = (1 - w) * u + w * c
```

Grid đóng băng gồm 16 gates: L ∈ {60,90}, U ∈ {125,150}, M ∈ {0,25; 0,5; 0,75; 1}. Gate chỉ dùng prediction hiện tại, không dùng RUL thật hoặc lifetime để quyết định trọng số. Output giữ đơn vị chu kỳ và không clip. Ví dụ gate final-inner L=90, U=125, M=0,5: khi `u=107,5`, trọng số cap125 bằng 0,25. Trọng số liên tục không bảo đảm trajectory giảm đơn điệu.

5 outer folds và 3 inner folds đều chia theo engine, seed 42. Warning threshold {30,35,40,45,50} và persistence {1,3,5} được calibrate bằng inner OOF; illustrative lead requirement là 30 chu kỳ. HGB reference giữ đúng policy nguồn ở từng stage, gồm threshold40/persistence3 của outer fold4.

## Outer readout

100 engines, 17.731 trajectory rows và 300 endpoint rows. Endpoint panel dựng hồi cứu tại 50%, 70%, 90% lifetime. Endpoint NASA là mean-engine score, khác tổng NASA của test table bên dưới.

| Procedure | Trajectory RMSE | Endpoint MAE | Endpoint mean-engine NASA | On-time | Early-alarm burden |
|---|---:|---:|---:|---:|---:|
| HGB uncapped | 29,806846 | 9,891336 | 6,883768 | 0,94 | 0,052159 |
| HGB cap125, reference | 36,208014 | 9,321925 | 3,683556 | 0,94 | 0,055003 |
| Gated blend, diagnostic | 29,766509 | 9,634579 | 6,771263 | 0,96 | 0,053870 |
| Selected pipeline | 29,802876 | 9,897053 | 6,897570 | 0,94 | 0,052043 |

Early-alarm burden là tỷ lệ thời gian cảnh báo trong các rows có raw RUL >30, tính riêng từng engine rồi lấy trung bình. Đây là chỉ số workload của policy minh họa. On-time nghĩa là cảnh báo đầu tiên có lead ≥30 chu kỳ. Selected pipeline có trung bình 1,74 transitions và 0,37 retractions mỗi engine, bằng HGB uncapped; diagnostic blend có 1,76 và 0,38.

### Promotion gate của selected pipeline so với HGB uncapped

Paired engine bootstrap 10.000 lần, seed42; delta = candidate − baseline. Pooled RMSE tính bằng tổng SSE/tổng rows, giữ multiplicity của engines được lấy mẫu lặp.

| Metric | Delta | Paired 95% CI | Điều kiện | Đạt |
|---|---:|---:|---|---|
| Trajectory RMSE | −0,003970 | [−0,014889; 0,005437] | upper <0 | Không |
| Endpoint MAE | 0,005717 | [−0,034595; 0,047881] | upper ≤0 | Không |
| Endpoint mean-engine NASA | 0,013802 | [−0,012821; 0,057591] | upper ≤0 | Không |
| On-time rate | 0 | [0; 0] | lower ≥0 | Có |
| Early-alarm burden | −0,000115 | [−0,000982; 0,000778] | upper ≤0 | Không |

Mọi margins bằng 0 như [protocol](../../../../experiments/GATED_BLEND_PROTOCOL.md). Ví dụ point estimate burden giảm rất nhỏ, nhưng upper CI vẫn dương nên điều kiện chưa đạt. Không nới điều kiện sau kết quả.

Diagnostic blend có delta RMSE −0,040337, CI [−0,078115; −0,007778]; endpoint MAE −0,256758 và NASA −0,112505 cũng có upper CI âm. Tuy nhiên burden tăng 0,001711, CI [−0,000627; 0,004159]. Ngay cả diagnostic này cũng chưa đáp ứng toàn bộ mục tiêu. Các gates diagnostic không feasible ở bốn folds, nên đây không phải readout của quy trình chọn model đủ điều kiện.

## Inner selection và final artifact

| Outer fold | Số gates feasible/improving | Selected procedure | Gate diagnostic |
|---|---:|---|---|
| 1 | 0/0 | HGB uncapped | L60/U125/M0,25 |
| 2 | 0/0 | HGB uncapped | L90/U125/M0,25 |
| 3 | 4/4 | Gated blend | L90/U125/M0,25 |
| 4 | 0/0 | HGB uncapped | L90/U125/M0,5 |
| 5 | 0/0 | HGB uncapped | L90/U125/M0,5 |

Final-inner trên toàn bộ training engines có 8/16 gates feasible, 5 improving; chọn **L90/U125/M0,5**, policy threshold40/persistence1. Inner RMSE giảm từ28,835035 xuống28,788977. Điều này giải thích vì sao final artifact chứa một blend dù outer promotion gate không đạt. Nó được lưu để tái tạo thí nghiệm; lựa chọn cho vận hành vẫn không được thông qua.

Hai base models, final spec/policies và bốn final prediction procedures được lưu, nạp lại và replay chính xác trước khi đọc test labels. Outer decision được đóng băng trước final test reporting và giữ nguyên sau test.

## Test reporting

100 test engines, một prediction tại endpoint mỗi engine. Test từng được xem trong lịch sử và chỉ dùng reporting.

| Procedure | MAE | RMSE | Tổng NASA score |
|---|---:|---:|---:|
| HGB uncapped | 13,753450 | 20,933433 | 3744,351401 |
| HGB cap125 | 9,375761 | 12,549297 | 212,123270 |
| Gated blend / final selected pipeline | 13,544074 | 20,896506 | 3738,899140 |

Test cải thiện nhẹ của final blend không thay đổi gate đã đóng băng. [Test receipt](test_evaluation.json) ghi `selection_used_test_labels=false` và `outer_gate_changed_by_test=false`.

## Kiểm chứng và bàn giao

63 focused tests đã đạt trong project chính. Independent verifier đạt **32/32 checks**, tái tính provenance, 46 CV caches, selections, policies, metric readouts, bootstrap và saved-model replay; không fit model hoặc đọc raw test labels. Sau copy, inference bằng code trong project chính cũng tái tạo chính xác cả 4 procedures trên 100 endpoint rows, max absolute delta bằng 0. Verification receipt và delivery receipts ghi riêng trạng thái của việc kiểm chứng và copy.

Hai lỗi kỹ thuật được xử lý trước lượt hoàn tất: [smoke đầu](../20261005_smoke/README.md) bị đường dẫn snapshot quá dài; [study01](../20261005_study01/README.md) bị Windows từ chối atomic replace file tiến độ. Snapshot đã dùng tên ngắn với mapping kiểm collision; atomic replace retry có giới hạn và vẫn báo lỗi nếu kéo dài. Smoke02 và smoke03 là engineering scope, không có scientific promotion hoặc test scoring.

## Phạm vi kết luận

Đây là follow-up trên engine folds/test đã được xem trong lịch sử. Bootstrap CI conditional trên fixed outer predictions, không bao gồm refitting variability, seed variability hoặc hiệu chỉnh nhiều phép thử. Endpoint panel là đánh giá hồi cứu; lead30 là policy minh họa. Không suy rộng kết luận này thành mọi phương pháp gating đều kém.

Kết thúc nhánh thử nghiệm này với kết luận **không chọn** theo grid và tiêu chí hiện tại. Published model/registry được giữ nguyên.

## Cấu trúc triển khai

```text
src/cmapss/modeling/gated_blend/
  contracts.py   # GateSpec và grid
  gating.py      # weights, blend và exact row alignment
  config.py      # fixed experiment contract
  analysis.py    # inner selection, warning controls, paired readouts
  inference.py   # checksum và saved model inference
  runner.py      # provenance, nested derivation, test ordering
scripts/
  run_gated_blend_study.py
  verify_gated_blend_study.py
```

## Bằng chứng

[Run info](run_info.json), [manifest](manifest.json), [reuse inventory](reuse_inventory.json), [outer selections](outer_selections.json), [final selection](selection.json), [aggregate metrics](aggregate_metrics.csv), [paired comparisons](paired_comparison.csv), [region metrics](region_metrics.csv), [warning events](warning_events.parquet), [decision](decision.json), [inference metadata](inference.json), [prediction replay receipt](final_predictions_receipt.json), [verification](verification.json), [delivery tests](tests_delivery.xml), [delivery replay](delivery_replay.json), [delivery](delivery.json).

Source identity của lượt hoàn tất: `35c5ca253b77062f4f6065d0ef90b3c4a143fa7e5c256e2305ecc4afe7230d77`. Source HGB study identity: `c10425d965e3f7d0411d14a56d15526ff33373c8e2ee077467b3aa38950ce90a`.
