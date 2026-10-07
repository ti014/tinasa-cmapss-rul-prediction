# Protocol HGB multi-scale và initial-profile

**Tác giả:** Phan Nguyễn Mai Phương — **MSHV 25002711**  
**Ngày khai báo:** 2026-10-06  
**Trạng thái:** PLANNED, trước khi chạy study đầy đủ.

## 1. Câu hỏi và giả thuyết

Các study trước đã dùng HGB trên 169 causal features với trailing window 30
cycles. Một phép thử window 10 riêng lẻ không cải thiện kết quả. Study này
kiểm tra hai biểu diễn mới trên cùng một HGB protocol:

1. giữ window 30 và bổ sung statistics của window 10 để mô hình thấy cả thay
   đổi ngắn hạn và xu hướng dài hơn;
2. giữ window 30 và bổ sung profile sensor của 30 cycles đầu để biểu diễn
   mức nền của từng engine.

Giả thuyết là thêm thông tin ngắn hạn hoặc baseline theo engine có thể giảm
trajectory RMSE ở vùng RUL cao mà không làm xấu endpoint và warning controls.
Không giả định arm nào sẽ thắng. Ví dụ, tại cycle 30, window 10 dùng các
cycles 21--30; profile dùng các cycles 1--30; cả hai đều không dùng dữ liệu
tương lai.

## 2. Ba arms cố định

| Arm | Feature model | Số feature | Thay đổi so với baseline |
| --- | --- | ---: | --- |
| `hgb_baseline169` | rolling30 (126) + `cycle` + history từ initial30 (42) | 169 | Đối chứng HGB hiện tại |
| `hgb_multiscale10_30` | baseline169 + rolling10 (126) | 295 | Thêm 6 statistics × 21 sensors ở window10 |
| `hgb_initial_profile42` | baseline169 + initial profile (42) | 211 | Thêm mean/std của 21 sensors trong cycles1--30 |

`hgb_multiscale10_30` có 21 cột `last` của window10 trùng giá trị với cột
`last` ở window30 tại cùng cycle. Các cột này vẫn giữ nguyên để phản ánh
đúng recipe đã khai báo; báo cáo phải nêu rõ 105 cột window10 còn lại là phần
thông tin mới. Initial profile dùng sample standard deviation (`ddof=1`).

Mọi arm dùng cùng panel khóa `(split, engine_id, cycle)`, chỉ giữ các dòng
cycle >= 30. Việc thêm window10 hoặc profile không được làm rơi hoặc thêm dòng.
Không feature nào được chứa `rul`, `y_true`, `lifetime`, test label hoặc
thông tin lifetime. `setting_1..3` không được dùng trong hai nhóm feature mới.

## 3. Dữ liệu, folds và fitting

- Raw RUL giữ nguyên ở validation, outer readout và test reporting. Chỉ
  clipping target nếu có arm cap, nhưng study này không có target cap.
- Sample weight là `engine_total_one`: tổng trọng số của mỗi engine trong
  fitting fold bằng một.
- Outer split: 5 folds theo engine; inner split: 3 `GroupKFold` folds theo
  engine; seed 42.
- Cấu hình `HistGradientBoostingRegressor`: squared error, 200 iterations,
  learning rate 0,05, tối đa 31 leaves, `min_samples_leaf=20`, L2=0,
  `early_stopping=False`, CPU 4 threads.
- Feature values đều là `float64`; model feature order được lưu trong
  metadata và phải được kiểm tra lại khi reload.

Ba arm phải được fit và chọn trong cùng các folds. Không được chọn arm từ
outer readout hoặc từ test score. Nếu runner reuse HGB fits của study trước,
chỉ được reuse khi source feature matrix, row keys, fold assignments, model
parameters, input hashes, runtime và cache key khớp; mọi fit của hai arm mới
phải được ghi là fresh fit.

## 4. Selection và promotion gate

Trong mỗi outer fold, inner OOF predictions dùng để chọn warning policy trong
grid threshold `{30,35,40,45,50}`, persistence `{1,3,5}`, lead requirement 30
và target on-time 0,9. Candidate được chọn theo pooled trajectory RMSE thấp
nhất trong các candidate có endpoint mean-engine MAE, endpoint mean-engine
NASA, on-time rate và early-alarm burden không xấu hơn `hgb_baseline169`; mọi
margin bằng 0. Nếu không có arm mới khả thi, fallback là baseline.

Metrics bắt buộc gồm trajectory RMSE, endpoint MAE/NASA tại 50%, 70%, 90%
lifetime, on-time rate, early-alarm burden, warning transitions và warning
retractions. Promotion gate của selected pipeline so với baseline được đóng
băng trước run:

- upper 95% CI của delta trajectory RMSE < 0;
- upper CI của delta endpoint MAE, endpoint NASA và early-alarm burden <= 0;
- lower CI của delta on-time rate >= 0.

CI dùng paired engine bootstrap 10.000 lần, seed 42, giữ multiplicity của
engine; pooled RMSE tính từ tổng SSE và số dòng, không lấy trung bình RMSE
engine. CI là conditional trên fixed predictions, không bao gồm seed/refit
variability và không hiệu chỉnh multiplicity giữa các arm thăm dò.

## 5. Ordering và phạm vi kết luận

Trước khi đọc test labels, runner phải đóng băng final arm, warning policy,
model metadata và test predictions; reload model rồi replay prediction exact.
Test chỉ dùng để reporting sau khi outer decision đã freeze. Folds và test đã
được xem trong lịch sử project, nên đây là follow-up có historical inspection,
không phải blind independent confirmation.

Namespace dự kiến: `outputs/experiments/multiscale_comparison/20261006_study01`.
Smoke phải dùng namespace riêng và không được đánh dấu scientific promotion.
Study không tự thay published model/registry, không commit hoặc push.

Nếu cả hai arm mới không đạt gate, kết luận là biểu diễn multi-scale/profile
chưa chứng minh cải thiện theo protocol này; giữ `hgb_baseline169`. Không chạy
TCN hoặc rerun Linux chỉ để tìm một con số đẹp.

## 6. Module và artifacts

```text
src/cmapss/modeling/multiscale_comparison/
  contracts.py   # arm IDs, feature order, counts và invariants
  config.py      # folds, HGB, warning và promotion controls
  features.py    # xây ba feature panels causal trên common key panel
  runner.py      # nested fitting, selection, replay và test ordering
  analysis.py    # metrics, warning policies và paired CIs
tests/
  test_multiscale_features.py
  test_multiscale_core.py
  test_multiscale_runner.py
```

Mọi notebook hoặc báo cáo phát sinh từ study phải ghi ở phần đầu: **Phan
Nguyễn Mai Phương — MSHV 25002711**.
