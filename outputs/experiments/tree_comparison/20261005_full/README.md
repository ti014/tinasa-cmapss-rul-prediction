**Phan Nguyễn Mai Phương — MSHV 25002711**

# Kết quả so sánh HGB và XGBoost — FD001, 05/10/2026

**Đã hoàn tất 492 lượt fit; XGBoost chưa đáp ứng tiêu chí thay HGB trong grid đã thử.** Quy trình chọn bằng inner CV giữ `hgb_uncapped` ở cả năm outer folds và ở final selection. Promotion gate không đạt vì quy trình này có cùng predictions với đối chứng, nên delta RMSE bằng 0.

HGB uncapped là đối chứng phù hợp hơn cho toàn quỹ đạo trong đợt này. HGB cap125 có điểm test tốt hơn. Kết quả này chưa đủ căn cứ chuyển mô hình đã công bố sang uncapped hoặc thay bằng XGBoost.

## 1. Outer evaluation trên 100 động cơ

Mỗi động cơ thuộc một outer validation fold và được giữ ngoài cả model fitting lẫn chọn cấu hình/policy. Bảng gồm 17.731 cửa sổ trên toàn quỹ đạo; endpoint panel có 300 dòng tại 50/70/90% tuổi thọ. Các vị trí endpoint được xác định hồi cứu, không dùng làm input model.

| Arm | Trajectory RMSE | Endpoint MAE theo engine | Endpoint NASA theo engine | On-time rate | Early alarm burden |
| --- | --- | --- | --- | --- | --- |
| HGB uncapped | 29,807 | 9,891 | 6,884 | 94,0% | 0,05216 |
| HGB cap125 | 36,208 | 9,322 | 3,684 | 94,0% | 0,05500 |
| XGBoost uncapped | 29,975 | 11,626 | 66,306 | 93,0% | 0,05696 |
| XGBoost cap125 | 36,157 | 9,750 | 3,815 | 91,0% | 0,05650 |

`selected_pipeline` trùng hoàn toàn với HGB uncapped trong outer evaluation. Các số NASA ở bảng này là trung bình theo động cơ, với ba endpoint mỗi động cơ; không so trực tiếp với NASA tổng ở bảng test.

XGBoost uncapped có delta trajectory RMSE **0,168** so với HGB uncapped; 95% CI **[-0,540; 0,913]** đi qua 0. Chưa có bằng chứng cải thiện RMSE. Endpoint MAE tăng **1,734** chu kỳ; CI **[0,990; 2,540]** nằm phía tăng sai số.

HGB cap125 giảm endpoint MAE từ 9,891 xuống 9,322, đồng thời tăng trajectory RMSE từ 29,807 lên 36,208. Hai mục tiêu cần được báo cáo riêng.

[Biểu đồ outer metrics](outer_metric_comparison.png) · [Paired confidence intervals](paired_confidence_intervals.png)

## 2. Test reporting-only

Bốn final models, final selection và toàn bộ test predictions được lưu và kiểm checksum trước khi mở nhãn test. Test metrics không tham gia chọn cấu hình hoặc sửa promotion gate. Mỗi động cơ test đóng góp một endpoint cuối cùng.

| Arm | MAE | RMSE | NASA tổng |
| --- | --- | --- | --- |
| HGB uncapped | 13,753 | 20,933 | 3744,351 |
| HGB cap125 | 9,376 | 12,549 | 212,123 |
| XGBoost uncapped | 14,723 | 21,706 | 3481,178 |
| XGBoost cap125 | 9,629 | 12,701 | 208,034 |

Ví dụ: HGB uncapped tốt hơn HGB cap125 về RMSE toàn quỹ đạo ở outer CV, nhưng test MAE lần lượt là 13,753 và 9,376 chu kỳ. Không thể lấy một metric để kết luận uncapped tốt hơn ở mọi tình huống. XGBoost cap125 có NASA tổng thấp hơn HGB cap125 trong bảng test; đây là readout đã đóng băng, không phải căn cứ chọn lại mô hình.

## 3. Cấu hình và final selection

Bốn arm dùng chung ma trận 169 features: 126 rolling30 features, `cycle` và 42 độ lệch sensor last/mean so với trung bình 30 cycles đầu. Evaluation luôn dùng raw RUL. Ví dụ nhãn thật 180 được đổi thành 125 chỉ khi fit arm capped; validation vẫn được chấm với 180.

HGB dùng 200 boosting iterations, mỗi iteration một regression tree; tối đa 31 leaf nodes, tối thiểu 20 mẫu mỗi leaf, learning rate0.05, không giới hạn depth, L2=0, không early stopping. XGBoost dùng 300 trees, `hist`/CPU, depth3/5/7, learning rate0.03/0.1, nominal lambda1/10, nominal min_child_weight20; 12 cấu hình mỗi cap. Các cây được cộng dồn để hiệu chỉnh residual; cấu hình HGB cố định, còn XGBoost được tìm trong grid này.

Mọi engine có tổng sample weight bằng1. Với XGBoost, effective lambda/min_child_weight bằng nominal × số engine fit/số dòng fit. Metadata mỗi fit giữ cả giá trị nominal và actual. Không dùng validation để tính scale.

| Final arm | Depth | Learning rate | Nominal L2 | Endpoint family controls | Threshold/persistence |
| --- | --- | --- | --- | --- | --- |
| HGB cap125 | Không giới hạn | 0.05 | 0.0 | True | 40/1 |
| HGB uncapped | Không giới hạn | 0.05 | 0.0 | True | 40/1 |
| XGBoost cap125 | 3 | 0.1 | 1.0 | True | 45/5 |
| XGBoost uncapped | 5 | 0.1 | 10.0 | False | 40/1 |

`xgb_uncapped` không có cấu hình đáp ứng endpoint family controls trong final inner CV. Cấu hình RMSE tốt nhất được giữ với `numeric_controls_met=false` để đo kết quả. `xgb_cap125` đạt controls so với HGB cùng cap trong final inner CV, nhưng không được chọn bởi quy tắc trajectory và safeguards so với HGB uncapped.

Policy cuối của HGB uncapped: predicted RUL <=40, một quan sát xác nhận. Đây là policy minh họa benchmark, với lead requirement30; chưa phải quy tắc bảo trì thực tế.

## 4. Kiểm chứng và provenance

- Full scientific protocol: outer5 × inner3, seed42, hai process × bốn fitting threads, bootstrap10.000 lần theo engine.
- 468 inner fits +20 outer fits +4 final fits =492; mới hoàn toàn trong namespace này.
- 19 focused tests đạt trong worktree và trong project chính. 14 warnings là deprecation từ Matplotlib/Pyparsing.
- Smoke run hoàn tất; resume giữ nguyên hash của77 artifact gồm36 cache fits, bốn final models và final predictions. Smoke không được dùng làm bằng chứng promotion.
- Verifier đọc lại488 CV cache objects, kiểm disjoint engine IDs, raw truth, keys, feature schema, inner-only choice/policy, selected aliases, outer metrics, paired bootstrap, final model checksum/reload và các cờ test ordering. Không fit lại hoặc mở raw test labels.
- Raw truth, predictions khi reload model và paired bootstrap dùng exact checks. Metric được tính lại độc lập dùng rtol1e-10/atol1e-8 để xét sai khác cộng số thực; không nới tolerance của prediction.
- Full run mất **1137.90 giây (18.96 phút)** trên Windows. Source baseline từ Linux commit `919450a2e773f030f1e943dbe952814b0c337f30` trong managed worktree.
- Source identity: `83a74ec032a6e02e39e4d53739d6ecfc4de179f185ed7df299125b32849484f5`. Runtime: Python3.12.14, sklearn1.9.1, NumPy/pandas2.3.3, XGBoost3.4.1; chi tiết trong [manifest](manifest.json).

[Verification](verification.json) · [Gate trước test](outer_decision_before_test.json) · [Final selection](selection.json) · [Aggregate metrics](aggregate_metrics.csv) · [Paired comparison](paired_comparison.csv) · [Test metrics](test_metrics.csv) · [Engineering preflight](engineering_verification.json)

## 5. Giới hạn và lần thử tiếp theo

Outer folds và test FD001 đã được xem trong nghiên cứu trước. Đây là nghiên cứu tiếp nối, chưa phải xác nhận độc lập trên dữ liệu mới. Bootstrap conditional trên predictions cố định; chưa bao gồm refitting/seed variability hoặc hiệu chỉnh multiplicity trên các so sánh thăm dò. Không suy rộng kết quả của grid này thành kết luận mọi XGBoost đều kém HGB.

Thứ tự đã chốt vẫn là: giữ hai HGB references theo hai mục tiêu, hoàn tất XGBoost (đợt này), rồi chuẩn bị protocol TCN. TCN cần kiểm tra riêng trên cùng engine folds và raw-RUL readouts; kết quả hiện tại chưa cho biết TCN sẽ tốt hơn. Có thể dùng [mã TCN của tác giả nghiên cứu năm 2018](https://github.com/locuslab/TCN) làm nguồn đối chiếu kiến trúc.

Code mới đã được đưa về project chính: package `src/cmapss/modeling/tree_comparison`, audit helper, CLI và tests. Chỉ thêm dependency/entry point XGBoost; giữ các user edits, gồm dev dependency `psutil`. Các artifact/model/registry đã công bố được kiểm hash và giữ nguyên.

Để chạy lại trên Linux sau khi bạn đồng bộ Git, dùng namespace mới:

```bash
uv sync --frozen
uv run --frozen python scripts/run_tree_comparison.py \
  --output-dir outputs/experiments/tree_comparison/20261005_linux_xgb \
  --fit-threads 4 --max-workers 2
uv run --frozen python scripts/verify_tree_comparison.py \
  --output-dir outputs/experiments/tree_comparison/20261005_linux_xgb
```

Run Windows được lưu theo source/input/lock hashes trong worktree. Project chính có dependency lock bao gồm user edits; dùng run ID mới khi chạy từ checkout đó. Resume chỉ dùng cùng identity, không chuyển cache sang source hoặc lock khác. Nội dung đầy đủ về package tree và protocol nằm trong `experiments/TREE_COMPARISON_PROTOCOL.md`.
