# Pilot similarity-based RUL — FD001

**Phan Nguyễn Mai Phương — MSHV 25002711**  
**Ngày:** 07/10/2026

## Phạm vi đóng băng trước chạy

Pilot CPU nhỏ để sàng lọc một cách suy luận mới cho mục tiêu RUL hiện tại.
Không thay published model, registry, report chính hoặc notebook. Không chạy
TCN, không yêu cầu Linux. Không đọc test observations/labels hoặc outer scores
để chọn cấu hình. Pilot không có quyền promotion và không tự chạy full study.

## Giả thuyết và biểu diễn

Mức sensor và hình dạng đoạn suy giảm gần đây có thể giúp tìm những engine
train đang ở giai đoạn tương tự. Chưa giả định phương pháp vượt HGB.

- Dùng canonical FD001 train observations và raw RUL targets.
- Common panel: mọi engine, mọi cycle >=30; cùng 17.731 keys với HGB169.
- Window cố định30, chia6 khối liên tiếp5 chu kỳ; lấy sensor mean mỗi khối.
  Biểu diễn giữ thứ tự6 khối. Không dùng age, target hoặc future query values
  trong khoảng cách; đây là so sánh phương pháp, không cô lập riêng kiến trúc.
- StandardScaler fit trên raw observations của fitting engines, với tổng trọng
  số mỗi engine bằng1. Chỉ giữ sensor có standard deviation >1e-8, học từ fit.
- Thư viện chứa mọi window30 của fitting engines. Mỗi window có raw RUL tại
  endpoint. Lifetime của fitting engine được biết vì train chạy đến hỏng.
- Khoảng cách là mean squared Euclidean trên biểu diễn đã chuẩn hóa. Với mỗi
  query, lấy một window có khoảng cách thấp nhất trên mỗi donor engine; tie
  chọn endpoint sớm nhất. Lấy k donor engine gần nhất, k cố định trong {5,10};
  tie donor theo engine_id. Trọng số 1/max(distance,1e-8), chuẩn hóa tổng1.
- Tính distance theo batch128, float64, tối đa4 BLAS threads; lưu donor traces
  của10 donor tốt nhất để đối chiếu phép tổng hợp và ví dụ.

## Folds và đối chứng

GroupKFold3, shuffle=False, trên đúng common panel; fitting/validation engine
disjoint. Dùng final-inner HGB baseline169 predictions đã lưu của study
20261006_study02 khi fit/validation IDs, keys và targets trùng và checksums
được kiểm tra. Không dùng baseline outer vector (khác training partition).
Nếu reference không đủ bằng chứng, dừng để sửa harness; không ghép sai folds.

HGB reference: raw-RUL, history169, 200 iterations, learning_rate0.05,
max_leaf_nodes31, min_samples_leaf20, L2=0, no early stopping, seed42,
engine-total-one weights. Không refit baseline khi replay đủ bằng chứng.

## Metrics và điểm dừng

Reuse audited trajectory RMSE và endpoint mean-engine MAE/NASA tại fractions
0.5/0.7/0.9. Cùng warning grid thresholds30/35/40/45/50, persistence1/3/5,
lead30, target on-time0.9, calibration trên pooled pilot OOF riêng từng arm.
Warning scores dùng cùng OOF đã calibrate, chỉ là screening, không held-out
warning evidence. Báo pooled metrics, từng fold và RUL regions.

Một arm đáng chạy full khi pooled trajectory RMSE < HGB, giảm RMSE ở ít nhất
2/3 folds, endpoint MAE/NASA không tăng, on-time không giảm, burden không tăng.
Mọi margin0. Nếu nhiều arm đạt, chọn RMSE nhỏ nhất, tie k nhỏ nhất. Nếu không
arm nào đạt thì dừng nhánh pilot; không mở grid sau khi xem kết quả. Dù đạt,
pilot vẫn không promotion; full evaluation giữ gate đã dùng trong các study.

Ngân sách:3 library fits,6 k readouts dùng chung distance search, baseline
fresh fits0, tổng scientific pilot tối đa10 phút; nếu vượt dừng và báo chưa đủ
kết quả, không kết luận mô hình kém. Dry-run nhỏ riêng chỉ kiểm engineering.
Đo wall time, memory working set, batch size và library bytes. Kiểm invariants
và saved-model replay trước scientific run. Snapshot source/protocol/inputs,
runtime versions và reference receipts; run directory mới, không resume.

Ví dụ: một query khớp donor A ở cycle140, donor A hỏng cycle220, thì donor
này góp RUL80. Query chỉ được sử dụng observations đến thời điểm hiện tại.

## Nguồn

- Wang et al. (2008), similarity-based prognostics:
  https://doi.org/10.1109/PHM.2008.4711421. Pilot dùng block means trực tiếp,
  không tuyên bố tái lập đúng Health Indicator model của bài gốc.
- NumPy2.5, sliding_window_view: axis=0; window dimensions appended;
  writeable=False mặc định (đã tra cứu07/10/2026).
- scikit-learn1.9.1 docs, StandardScaler: with_mean/with_std=True;
  fit(...,sample_weight=...) và GroupKFold: shuffle=False, group disjoint.
  Runtime installed version được ghi riêng; không đồng nhất với docs stable.
