# Experiment journal

Học viên: **Phan Nguyễn Mai Phương** — **MSHV 25002711**.

### 2026-09-28 — Rebuild after source line-ending normalization

**Status**: COMPLETED.
**Purpose**: Recreate provenance with current source bytes, retaining the
20260927_full run as historical evidence. No change to the search recipe or
selection rule; expected metrics should match the original study.
**Config**: Same seed 42, 5 outer folds, 3 inner folds, 24 HGB candidates,
windows 10/20/30, 10,000 bootstrap resamples and 4 fit threads.
**Output**: outputs/experiments/rul_improvement/20260928_rebuild.
**Recovery**: Resume checks config, source, inputs, dependencies and baseline
snapshot before reusing cached fits. Record written during recovery; the
search manifest was already created by the original invocation before fitting.
**Actual outcome**: Selected `window=30`, `weight_scheme=uniform`, matching the
original baseline. The verifier passed with 511 fit records, 100 outer engines,
baseline unchanged and 100 saved-model test predictions matching. Ablation,
paired comparison, outer metrics and test comparison match `20260927_full`;
only `fit_seconds` differs.
**Duration**: Resumed run completed after the original process stopped.
**Learning**: Rebuilding with current source bytes restores a passing provenance
check without changing the experiment result.
**Next**: Commit and push the repository changes after the final diff audit.
Full test suite during recovery: 188 passed, 14 dependency deprecation warnings.

### 2026-09-27 — RUL improvement baseline and staged nested GroupKFold

**Status**: PLANNED

**Hypothesis**: A shorter trailing window, tuned HistGradientBoosting parameters,
or extra weight on low-RUL rows may reduce error when RUL is at most 30 cycles
and reduce warnings that arrive after the actual 30-cycle threshold.

**Change**: Add a separate experiment runner. Compare windows 10/20/30, tune 24
HistGradientBoosting configurations, then compare three low-RUL weighting schemes.
Use fixed outer 5-fold and inner 3-fold GroupKFold by engine. Keep the existing
30-cycle model artifacts unchanged until the final comparison is complete.

**Config**:
- window_cycles: 30 baseline; candidates 10, 20, 30
- model: HistGradientBoostingRegressor; existing baseline parameters first
- low-RUL weights: (1, 1, 1), (1, 1.5, 2), (1, 2, 3)
- seed: 42
- test labels: read only after final predictions are written

**Expected outcome**: Produce reproducible outer predictions and an ablation
table. A new configuration is adopted only if low-RUL MAE and on-time warning
rate improve without worsening the control metrics defined in the active plan.

**Baseline**: Existing test mean engine MAE 15.349047 cycles; test low-RUL MAE
4.383042 cycles; train OOF late-or-missed warning count 49/100.

### 2026-09-27 — Kiểm tra runner ban đầu

**Status**: COMPLETED (dry-run, không dùng để kết luận cải thiện)
**Config**: 2 outer folds, 2 inner folds, 3 cấu hình, 20 bootstrap samples.
**Actual outcome**: Luồng hoàn tất trong khoảng 8 phút 18 giây; 100 predictions,
model và final evaluation được ghi. Cấu hình cuối trùng baseline; test MAE
15,349047 cycles. Test đã được đọc ở cuối lượt chạy thử này.
**Learning**: Runner còn thiếu baseline outer, ablation và phục hồi từng fit;
đã bổ sung trước đợt đầy đủ. Kết quả thử không làm thay đổi search space.
**Next**: Dry-run v2 chỉ kiểm tra luồng và resume, không đọc test labels.

### 2026-09-27 — Dry-run v2 và phục hồi

**Status**: PLANNED
**Hypothesis**: Cache theo dữ liệu/cấu hình sẽ cho phép chạy tiếp với cùng
predictions và từ chối khi đầu vào khác. Đây là kiểm tra kỹ thuật.
**Config**: 2 outer, 2 inner folds; 1 HGB candidate; 20 bootstrap; 4 CPU threads;
skip test evaluation; lưu tmp/rul-improvement-dry-run-v2.
**Expected outcome**: Đủ bốn stage outer, paired comparison, model và 100 test
predictions; không có test evaluation. Chạy lại dùng cùng fit cache.

**Dry-run v2 result**: COMPLETED. 30 fit cache files saved; resume took 57.96 s
with all cache hashes and modification times unchanged. Changed config was
rejected. No test labels read. Four outer stage tables and paired CI generated.
54 relevant tests passed; runner-only suite passed again after final evaluation
comparison was added. Hardware: Intel Core Ultra 9 185H, 22 logical CPUs,
25,092,706,304 bytes RAM. Mean fit 6.00 s, maximum 7.33 s in dry-run.

### 2026-09-27 — Full nested RUL improvement study

**Status**: COMPLETED
**Hypothesis**: Same hypothesis and search space frozen above; no changes based
on dry-run scores. Test history includes earlier baseline analyses and dry-run v1.
**Config**: 5 outer folds copied from baseline artifact, 3 inner folds, windows
10/20/30, 24 HGB candidates, 3 weight schemes, 10,000 paired bootstrap resamples,
seed 42, 4 CPU threads. Common 17,731 train keys from cycle 30, 100 engines.
**Output**: outputs/experiments/rul_improvement/20260927_full
**Expected outcome**: Complete predeclared comparison, including unfavorable
results. Search manifest and baseline snapshot written before any fit.

**Full study result**: COMPLETED. 511 distinct cached fits plus final refit;
78.61 minutes. Outer low-RUL MAE 3.6676266 → 3.6729967, all-engine MAE
20.2143407 → 20.2458896, late/missed 49 → 53/100. Burden 0.0129328 → 0.0125798.
Low-RUL MAE delta CI [-0.0506078, 0.0638836]; late-rate delta CI [0.01, 0.08].
Final inner selection retained exact baseline; final test MAE 15.3490474,
RMSE 20.5412667, low-RUL MAE 4.3830424 on 25 engines, all unchanged.
**Learning**: Shorter windows did not win the declared selection. Some candidates
reduce low-RUL error but worsen controls: hgb_08 inner low-RUL MAE 3.262492
vs baseline 3.614783, overall MAE 20.684582 vs 20.361328. Do not interpret
selection scores as independent confirmation. Strong weighting also trades
lower low-RUL error and fewer late warnings for higher overall MAE and burden.
**Decision**: Retain baseline. No post-test tuning. Report all four outer stages,
test endpoint results separately and favorable/unfavorable engine examples.
**Validation**: 175 tests passed; separate artifact verification passed for
engine separation, paired CI recomputation, unchanged baseline and saved-model
prediction equality. Model integration is unnecessary because baseline retained.

### 2026-09-29 — Nhãn capped RUL và lịch sử suy giảm FD001

**Status**: PLANNED, protocol ghi trước khi fit.
**Hypothesis**: Giới hạn nhãn train có thể giảm sai số ở vùng sensor chưa thể hiện
suy giảm; độ lệch so với 30 cycles đầu và slope trên prefix có thể cải thiện
cảnh báo khi tốc độ suy giảm giữa các engine khác nhau.
**Config**: window 30; tám ứng viên là tích của cap {không cắt, 110, 125, 140}
và history {tắt, bật}. History thêm 42 features (21 độ lệch và 21 prefix slopes).
HGB giữ nguyên learning_rate=0.05, max_iter=200, max_leaf_nodes=31,
min_samples_leaf=20, l2_regularization=0, early_stopping=False, seed=42;
tổng sample weight mỗi engine bằng 1. Outer 5 folds lấy từ baseline, inner
3-fold GroupKFold; 10.000 paired engine bootstrap resamples; 4 fit threads.
**Selection**: Inner chọn low-RUL MAE nhỏ nhất trong các ứng viên không tăng
mean-engine MAE, late/missed rate và early-alarm burden so với baseline;
baseline luôn có trong tập ứng viên. Outer chỉ chấm baseline và quy trình chọn.
**Acceptance**: CI 95% của delta low-RUL MAE và late/missed rate đều dưới 0;
overall MAE và burden không tăng. Nếu không đạt, giữ baseline.
**Baseline**: Outer mean-engine MAE 20.2143407; low-RUL MAE 3.6676266;
late/missed 49/100; test MAE đã biết 15.3490474.
**Isolation**: Không cắt nhãn validation/test. Chốt final configuration bằng
inner CV toàn train, lưu model và test predictions trước khi đọc nhãn test.
FD001 test và OOF lịch sử đã được xem; đây là nghiên cứu tiếp nối có tính khám phá.
**Output**: outputs/experiments/followup/20260929/FD001.

### 2026-09-29 — Operating condition trong FD002 và FD004

**Status**: PLANNED, protocol ghi trước khi fit.
**Hypothesis**: Settings và chuẩn hóa sensor theo operating condition làm giảm
sự biến thiên do vận hành, cải thiện RUL trong từng subset nhiều điều kiện.
**Config**: window 10; ba ứng viên cố định: raw sensor baseline; raw sensor +
sáu setting features (last/mean của ba settings); normalized sensor + setting
features. StandardScaler settings, KMeans(k=6,n_init=10,seed=42) và sensor mean/std
theo cluster được fit lại chỉ trên training engines của từng inner/outer fold.
HGB và trọng số bằng FD001 baseline; không cắt nhãn. Outer 5 × inner 3 folds,
10.000 paired bootstrap resamples, 4 threads.
**Selection**: Inner chọn mean-engine MAE thấp nhất, hòa xét low-RUL MAE rồi tên
ứng viên. Outer đánh giá cả baseline và quy trình chọn. So sánh test cùng window
10 sau khi lưu final predictions. Không sử dụng target-test để fit normalization.
**Acceptance**: CI delta outer mean-engine MAE có upper < 0; báo cáo riêng
low-RUL MAE, NASA score, warning timing và burden, kể cả chỉ số xấu đi.
**Baseline**: Test within-subset MAE FD002 22.3555993 và FD004 27.2217781.
**Scope**: Within-subset có train của chính FD002/FD004; không gọi là transfer
không thích nghi từ FD001. Các test scores lịch sử đã được xem.
**Output**: outputs/experiments/followup/20260929/{FD002,FD004}.

### 2026-09-29 — Benchmark end-to-end Pandas và PySpark

**Status**: PLANNED, trước khi đo.
**Hypothesis**: Spark có chi phí khởi tạo cao trên dữ liệu nhỏ; mở rộng số dòng
có thể thay đổi chi phí tương đối. Không đặt trước giả định Spark nhanh hơn.
**Config**: FD001 observations được nhân bản ở scale 1× và 10× với engine IDs
riêng; ba repetitions mỗi engine/scale. Cùng Parquet input, window 30, 126
sensor features. Đo riêng read/materialize, engine summary, rolling features,
write và end-to-end; peak RSS lấy mẫu cho process tree gồm JVM. Spark local[4],
bốn shuffle partitions. Khởi tạo Spark ghi riêng, cache OS không được xóa.
**Validation**: Đối chiếu toàn bộ khóa/cột và sai số numeric Pandas/Spark ở 1×,
rtol=1e-7, atol=1e-7; row count/checksum của đầu vào mọi lượt được ghi.
**Scope**: Local benchmark trên dữ liệu nhân bản, không phải bằng chứng chạy
trên cluster hoặc tăng kích thước mẫu khoa học độc lập. Không chạy đồng thời
benchmark với model fitting. Dữ liệu trung gian lưu dưới tmp/.
**Output**: outputs/benchmarks/20260929.

### Điều chỉnh benchmark trước khi đo chính thức

Smoke test ngày 29/09 phát hiện máy Windows thiếu Hadoop native write support
(`winutils.exe`), nên Spark không ghi Parquet trực tiếp được. Benchmark local
sẽ dùng PyArrow writer trên driver cho cả hai engine. Với Spark, `write_seconds`
bao gồm `toArrow()` và ghi file; đây không phải phép đo distributed Parquet writer.
Giữ nguyên read/summary/rolling, 1x/10x, ba lần lặp và đo RSS cả JVM. Smoke test
không được đưa vào kết quả benchmark. Tham chiếu API:
https://spark.apache.org/docs/4.0.1/api/python/reference/pyspark.sql/api/pyspark.sql.DataFrame.toArrow.html

### Tổ chức thực thi modeling

Ba dataset độc lập được chạy bằng process riêng, mỗi process giữ 4 fit threads
trên máy 16 physical cores/22 logical CPUs. FD001 tiếp tục từ fit cache đã kiểm
checksum; không thay candidate, fold hay tiêu chí. Thời gian modeling chỉ là
wall time của run, không dùng so hiệu năng giữa model. Benchmark bắt đầu sau khi
cả ba process hoàn tất, để không bị tranh tài nguyên với training.

### Điều chỉnh heap trước lượt benchmark hoàn chỉnh

Lượt đầu hoàn tất scale 1x nhưng Spark thất bại với `Java heap space` tại bước
Arrow export của scale 10x, repetition 1. Giữ partial measurements và failure.json
trong `outputs/benchmarks/20260929/failed_default_heap/`. Cấu hình heap ban đầu
là mặc định, chưa ghim rõ. Lượt hoàn chỉnh dùng driver/executor memory 4g ở cả
hai scale, kiểm tra maxMemory thực tế, chạy lại toàn bộ 12 phép đo trong work-dir
mới. Không gộp thời gian từ lượt thiếu heap vào median cuối.


### 2026-09-29 — Kết quả hoàn tất của đợt bổ sung

**Status**: COMPLETED.

| dataset | selected_final | baseline_outer_mae | selected_outer_mae | test_mae | outer_gate_passed |
| --- | --- | --- | --- | --- | --- |
| FD001 | cap_none_history_1 | 20.2143 | 20.2966 | 13.0460 | False |
| FD002 | condition_normalized | 29.2861 | 28.0390 | 19.1797 | True |
| FD004 | condition_normalized | 36.7428 | 35.4491 | 26.6123 | True |

**Learning**: FD001 giảm low-RUL MAE nhưng tăng late warnings và test NASA score; không thay baseline. Kết quả operating condition phải đọc theo within-subset, không suy rộng sang zero-shot transfer. Benchmark đo cả I/O và process-tree RSS trên local, không đại diện cluster.

**Evidence**: [Bảng kết quả, paired CI và giới hạn](FOLLOWUP_20260929.md); validation.json ở từng run.

### 2026-10-05 — Đối chiếu final-inner HGB giữa Windows và Linux

**Status**: COMPLETED.
**Authorization**: Người dùng cho phép chạy lại trên Windows để đối chiếu kết quả đã push từ máy Linux riêng.
**Hypothesis**: Cùng source, dữ liệu, versions, folds và cấu hình sẽ tái tạo metrics đã công bố; mọi sai khác prediction phải được ghi rõ. Giả thuyết được ghi trước khi fit trong journal của worktree commit `919450a2e773f030f1e943dbe952814b0c337f30`.
**Config**: Ba candidates baseline/history/cap125+history; GroupKFold3 trên100 training engines; window30;126/169 features; HGB200,seed42,four threads,engine-total-one weights. Chín fits mới; chỉ final-inner subset, không chạy lại outer/test hoặc promotion gate.
**Preflight**: 12 source files,hai canonical inputs,năm baseline inputs và uv.lock khớp SHA256 Linux sau chuẩn hóa LF trong worktree. Python3.12.14,sklearn1.9.1,numpy/pandas2.3.3,joblib1.6.0 khớp. Workspace chính có thay đổi chưa commit, nên source05/10 được chạy trong managed worktree riêng.
**Actual outcome**: Windows history trajectory RMSE28.835035143912293 so với Linux28.8336868877496; cap125+history36.53125898057416 so với36.53784158065441. Endpoint MAE Windows lần lượt10.550971491579391 và9.758692732845025. Hoàn tất chín fits trong80.17s. Folds,keys,raw truth,candidate specs và feature order khớp chính xác; prediction vectors chưa exact, max delta1.9679165026253145 cycles.
**Diagnostic**: Rolling features lệch tối đa7.105427357601002e-15. Một fresh baseline fit bằng frozen Linux features ở inner fold1 tái tạo chính xác5.878/5.878 predictions, max delta0; mất28.85s. Không suy rộng kiểm tra này thành bằng chứng exact cho mọi fold/candidate.
**Learning**: Kết quả xác nhận cùng thứ tự metrics; không cắt nhãn tốt hơn cho trajectory RMSE, cap125 tốt hơn cho endpoint MAE. Giữ riêng exact vector equality và agreement của metrics. Không nới tolerance, không đọc test labels và không tạo promotion decision.
**Evidence**: [Báo cáo đối chiếu](../outputs/experiments/windows_linux_comparison/20261005/README.md), `comparison.json`, `linux_feature_control.json`; fits/source snapshot tại `outputs/experiments/audit_remediation/20261005_windows_inner_replay`.


### 2026-10-05 — Trajectory-first HGB/XGBoost nested comparison

**Status**: PLANNED, recorded before any new fitting.
**Author**: Phan Nguyễn Mai Phương — MSHV 25002711.
**Authorization**: The user approved implementing and running the comparison.
**Hypothesis**: With identical causal history features and grouped splits, XGBoost may reduce raw whole-trajectory RMSE while preserving endpoint accuracy and calibrated warning quality. Uncapped HGB is the trajectory reference; no superiority is assumed.
**Change**: A factorial comparison of estimator (HGB/XGBoost) and training-target cap (none/125), with inner selection changed to trajectory-first under predeclared controls. This is a new research procedure, not a platform replay of the published endpoint-first study.
**Config**:
- Four arms: `hgb_uncapped`, `hgb_cap125`, `xgb_uncapped`, `xgb_cap125`; all use the same frozen 169 history features (126 rolling30 + age + 42 deviations from first30 sensor means). Features are built causally; no labels enter the feature matrix.
- Canonical FD001 train/test observations from the Linux commit919450a; published engine outer assignments (5 folds); inner GroupKFold3; seed42. Engine IDs never cross fitting/validation groups. Raw labels are used in every evaluation; cap only fitting labels.
- HGB200 iterations, learning_rate0.05, max_leaf_nodes31, min_samples_leaf20, L2=0, squared error, no early stopping.
- XGBoost3.4.1 (actual PyPI release), hist/CPU,300 estimators, max_depth{3,5,7}, learning_rate{0.03,0.1}, nominal reg_lambda{1,10}, nominal min_child_weight20, reg_alpha=gamma=0, subsample=colsample_bytree=1, no early stopping:12 settings per cap.
- All fits use engine-total-one weights. For XGBoost, effective reg_lambda and min_child_weight equal their nominal mean-one values times n_fit_engines/n_fit_rows; that scale uses fitting rows only and is recorded for every fit.
- Two process workers ×4 fitting threads; one frozen feature Parquet per split, source/input/config/lock hashes; compatible checkpoint resume only. Maximum492 fits:468 inner,20 outer,4 final.
- Within each cap, select the minimum trajectory RMSE among XGB settings with endpoint MAE and endpoint mean engine NASA no worse than same-cap HGB. If none are feasible, preserve the best-RMSE setting with a false feasibility flag. Calibrate warning policy for the four selected arms using inner predictions only (thresholds30/35/40/45/50, persistence1/3/5, lead30, target on-time0.9).
- Select an inner procedure by lowest trajectory RMSE subject to endpoint MAE/NASA/on-time/early-alarm-burden controls against uncapped HGB. All margins are0. Fallback is uncapped HGB. Freeze each outer choice before scoring that outer fold.
**Promotion evidence rule (frozen before results)**: Paired engine bootstrap10,000 draws conditional on outer predictions. Selected-pipeline minus uncapped-HGB pooled trajectory RMSE must have upper95% CI<0; endpoint MAE, endpoint mean engine NASA and early alarm burden upperCI<=0; on-time rate lowerCI>=0. Pooled-RMSE bootstrap resamples per-engine SSE and row counts with multiplicity. A separate stricter endpoint/NASA comparison with cap125 HGB is reported. No post-result relaxation or published model replacement.
**Test policy**: Freeze all final specs/policies and write all test predictions before reading labels. Test is reporting-only. The test and original outer folds have already been inspected historically; this is follow-on research, not an independent blind confirmation. The CI does not include repeated training/seed variability or a multiplicity correction across exploratory arms.
**Expected outcome**: Complete dry-run and saved-model reload/resume checks, then the full predeclared study; either a supported candidate under these controls or an explicit failed gate. Endpoint/trajectory tradeoffs remain visible.
**Baseline**: Windows final-inner history HGB trajectory RMSE28.835035143912293, endpoint MAE10.550971491579391; cap125-history trajectory RMSE36.53125898057416, endpoint MAE9.758692732845025. These earlier numbers are contextual, not acceptance thresholds for the new outer evaluation.
**Dry-run**: Separate namespace,2 outer ×2 inner, reduced XGB grid,20 iterations,20 bootstrap draws; validates engineering only and can never pass scientific promotion.
**Grounding**: Official XGBoost stable3.4.2 parameter/API docs and sklearn1.9.1 GroupKFold/HGB/nested-CV docs fetched; installed release3.4.1 is verified directly. Stable documentation is ahead of the currently published PyPI package, so runtime parameter/save/reload checks are required.
**Preservation**: Managed worktree `windows-linux-rul-replay/Final_Project`, branch `codex/xgboost-rul-comparison`; new experiment namespace; no commit/push and no modification of the published registry/models. Existing primary workspace edits remain in place.

### 2026-10-05 — Kết quả hoàn tất HGB/XGBoost

**Status**: COMPLETED; kết quả bổ sung cho protocol PLANNED ở trên, không thay đổi tiêu chí sau khi xem kết quả.
**Author**: Phan Nguyễn Mai Phương — MSHV 25002711.
**Execution**: 492 fresh fits (468 inner, 20 outer, 4 final), 1137,90 giây trên Windows. Source identity `83a74ec032a6e02e39e4d53739d6ecfc4de179f185ed7df299125b32849484f5`.
**Outer outcome**: HGB uncapped trajectory RMSE 29,806846; HGB cap125 36,208014; XGBoost uncapped 29,975343; XGBoost cap125 36,157426. XGBoost uncapped so với HGB uncapped có delta RMSE +0,168497, conditional paired 95% CI [-0,539653; 0,913103]; endpoint MAE tăng 1,734347 chu kỳ, CI [0,990391; 2,540428].
**Decision**: Inner selection giữ HGB uncapped ở cả năm outer folds và final selection. Selected pipeline trùng reference; promotion gate false vì delta trajectory RMSE và CI bằng 0. Các controls còn lại true do predictions trùng reference, không có nghĩa XGBoost đạt gate. Giữ mô hình/registry đã công bố.
**Test reporting-only**: HGB uncapped MAE 13,753450, RMSE 20,933433; HGB cap125 MAE 9,375761, RMSE 12,549297. XGBoost uncapped MAE 14,722642, RMSE 21,706098; XGBoost cap125 MAE 9,628994, RMSE 12,701174. Final selection/models/predictions được đóng băng trước khi mở nhãn; không chọn lại từ test.
**Engineering**: 19 focused tests đạt ở worktree và project chính; verifier đạt 21 checks, gồm 488 CV cache records và exact saved-model replay. Smoke resume giữ nguyên SHA256 của 77 artifacts. Artifacts full/smoke được copy sang project chính và đối chiếu SHA256 từng file.
**Learning**: Grid XGBoost hiện tại chưa cải thiện đối chứng HGB. Uncapped và cap125 có đánh đổi giữa trajectory và endpoint. Folds/test đã được xem trong nghiên cứu trước; CI conditional trên predictions cố định, không phải xác nhận độc lập hoặc đánh giá seed variability.
**Next**: Chuẩn bị protocol TCN theo thứ tự đã chốt; giữ hai HGB references và chốt readouts trước training. Khi chạy lại trên Linux dùng run ID mới vì lock/source của project chính khác worktree.
**Evidence**: [Báo cáo đầy đủ](../outputs/experiments/tree_comparison/20261005_full/README.md), `verification.json`, `engineering_verification.json`, `delivery_preservation.json`; [protocol](TREE_COMPARISON_PROTOCOL.md).

### 2026-10-05 — TCN pilot sàng lọc kiến trúc

**Status**: COMPLETED, non-promotional. Không dùng pilot để thay model hoặc mở published registry.
**Author**: Phan Nguyễn Mai Phương — MSHV 25002711.
**Config**: 21 raw sensors, sequence length30, projection21→32, bốn residual causal Conv1d blocks với dilations1/2/4/8, kernel3, dropout0,1, head32→1; hai arms raw-RUL và cap125; 5 outer folds theo engine; 8 epochs cố định; scaler fit trên fitting engines.
**Outcome**: TCN uncapped RMSE theo fold [41,690; 36,674; 35,580; 36,430; 32,625], MAE [27,405; 24,068; 24,368; 23,695; 23,589]. TCN cap125 RMSE [47,106; 43,099; 44,162; 40,062; 34,731]. TCN uncapped tốt hơn cap125 ở cả năm folds nhưng RMSE trung bình khoảng36,60, cao hơn HGB uncapped outer RMSE29,81.
**Interpretation**: Đây là screening result; 8 epochs có thể underfit. Không đọc test labels, không có inner selection/promotion gate/warning policy, không kết luận TCN kém ở mọi budget.
**Next**: Chỉ chạy full nested 50-epoch protocol nếu cần xác nhận TCN; runtime PyTorch phải được cài phù hợp Linux ARM. Nếu không chạy full, giữ TCN là rejected pilot candidate và chuyển sang phương án khác.
**Evidence**: [TCN pilot](../outputs/experiments/tcn_comparison/20261005_pilot/README.md), `outer_metrics.csv`, `pilot_info.json`; [TCN protocol](TCN_COMPARISON_PROTOCOL.md).

### 2026-10-05 — LightGBM nested comparison

**Status**: PLANNED for the full nested run; the preceding two-setting pilot is recorded separately in `outputs/experiments/lightgbm_comparison/20261005_pilot`.
**Hypothesis**: LightGBM's leaf-wise GBDT may reduce trajectory RMSE on the existing 169-feature matrix, but endpoint NASA can worsen even when RMSE improves.
**Change**: Add LightGBM 4.6.0 CPU arms on the frozen tree-comparison matrix; no feature, fold, weight or raw-RUL evaluation change.
**Config**: Four arms (uncapped/cap125 × base/reg10), 200 trees, learning rate0.05, num_leaves31, min_child_samples20, max_bin255, deterministic CPU, 4 threads, engine-total-one weights, 5 outer ×3 inner grouped folds, raw metrics.
**Baseline**: HGB uncapped pooled outer trajectory RMSE29.806846, endpoint mean-engine MAE9.891336, endpoint mean-engine NASA6.883768.
**Pilot context**: Base LightGBM pooled RMSE29.776109 and endpoint MAE9.823088, but endpoint mean-engine NASA21.804486; this does not pass the zero-margin NASA control and is not a selection result.
**Expected outcome**: A full nested readout either rejects LightGBM on endpoint safeguards or provides a candidate for paired CI review; no published model replacement.

**Status**: COMPLETED. 100% of outer inner selections chose `lgbm_uncapped_base`.
**Actual outcome**: Full nested 5×3 run with 200 trees gave LightGBM uncapped RMSE30.236 as fold mean; pooled paired readout after selecting the same base arm in each fold was RMSE29.842 versus HGB uncapped29.807, delta+0.034, CI[-0.280;0.354]. Endpoint-final MAE was2.435 versus2.417, delta+0.018, CI[-0.095;0.128]. LightGBM cap125 RMSE36.413.
**Learning**: The small pilot gain did not survive the matched 200-tree nested comparison; LightGBM is rejected for this protocol. The earlier pilot NASA deterioration also prevented treating its RMSE as a win.
**Next**: Keep HGB as reference and do not publish LightGBM. Preserve the isolated run for audit; consider a different feature/sequence hypothesis only with a new predeclared protocol.
**Evidence**: [LightGBM full report](../outputs/experiments/lightgbm_comparison/20261005_full/README.md), `outer_metrics.csv`, `inner_selection.csv`.

### 2026-10-05 — Hiệu chỉnh phạm vi TCN và LightGBM thăm dò

**Author**: Phan Nguyễn Mai Phương — MSHV 25002711.
**TCN**: Lượt nested 50 epochs được bắt đầu rồi chủ động dừng trước khi có outer readout, vì rà soát phát hiện runner chưa có endpoint panel 50/70/90%, warning controls và provenance/model-replay receipts. Không có kết quả scientific hoặc căn cứ loại TCN. Pilot 8 epochs chỉ là screening, khác thông tin đầu vào so với HGB history169 và có thể underfit. Status của TCN là chưa xác nhận; không gọi nó rejected model.
**LightGBM exploratory**: Mục COMPLETED ngay trước đây nói về nested exploratory runner, không phải full accepted gate. Summary 30,236/36,413 là trung bình gộp base+reg10 theo fold, không phải selected pipeline; endpoint-final là RUL=0, không phải endpoint panel đã chốt. Cụm rejected trong mục trước vượt phạm vi bằng chứng. Kết luận đúng là chưa có bằng chứng cải thiện; cần study riêng có đầy đủ safeguards.
**Preservation**: Raw predictions/metrics của các lượt thăm dò vẫn được giữ. Báo cáo sửa nhãn để tránh dùng chúng như xác nhận độc lập hoặc promotion evidence.

### 2026-10-05 — HGB/LightGBM rigorous study01

**Status**: PLANNED, ghi trước khi chạy scientific study01.
**Author**: Phan Nguyễn Mai Phương — MSHV 25002711.
**Hypothesis**: LightGBM trên cùng 169 causal history features có thể giảm pooled trajectory RMSE đồng thời giữ endpoint MAE/NASA và warning quality. Không giả định LightGBM thắng.
**Config**: HGB uncapped/cap125 cố định200 iterations, lr0.05,31leaves,20min samples. LightGBM4.6.0 CPU uncapped/cap125 × nominal L2 {0,10},200trees,lr0.05,31leaves,maxdepth-1,min_child_samples20,max_bin255,nominal min_child_weight0.001, no bagging, no early stopping, deterministic/force_col_wise true. Tổng6 candidate specs.
**Weights**: Engine-total-one. Actual reg_lambda và min_child_weight bằng nominal × n_fit_engines/n_fit_rows, scale chỉ tính trên fitting rows; metadata giữ cả nominal/actual.
**Splits/budget**: Published outer5 theo engine; GroupKFold inner3; seed42;2processes×4threads;108inner+20outer+4final=132freshfits. Frozen features/inputs/source/lock, cache chỉ resume khi identity giữ nguyên.
**Selection**: Chọn cấu hình LightGBM thấp nhất trajectory RMSE trong inner CV với endpoint engine MAE/NASA không xấu hơn same-cap HGB; không feasible thì diagnostic best-RMSE được giữ với false flag. Chọn pipeline theo trajectory subject to endpoint/NASA/on-time/burden controls so với HGB uncapped; fallback HGB uncapped. Calibrate warning threshold30/35/40/45/50, persistence1/3/5 từ inner predictions, lead30, target0.9.
**Gate**: Selected pipeline so với HGB uncapped: paired engine bootstrap10.000, pooled-RMSE từ SSE/counts giữ multiplicity, upper95%CI<0; endpoint MAE/NASA/burden upperCI<=0 và on-time lowerCI>=0. Mọi margins0. Không nới gate sau kết quả.
**Test**: Raw-RUL scoring; cap chỉ nhãn fitting. Frozen final specs/policies/models/predictions và saved-model replay receipts trước khi đọc test labels. Test chỉ reporting; historic folds/test đã xem trước đây, không phải blind independent confirmation. CI conditional trên predictions, không refitting/seed variability và không multiplicity-adjusted exploration.
**Expected**: Hoàn tất19focused tests/smoke trước scientific, rồi verifier độc lập, báo kết luận có điều kiện. No published model/registry replacement, no commit/push.
**Namespace**: outputs/experiments/lightgbm_comparison/20261005_study01; Linux dùng run ID mới với uv sync/run --frozen --extra lightgbm.
**Grounding**: LightGBM4.6.0 versioned LGBMRegressor API/Parameters chính thức đã kiểm; runtime4.6.0 wheel Windows đã cài, optional extra pinned trong lock. Code/params/save-reload được kiểm bằng smoke trước science.

### 2026-10-05 — Kết quả hoàn tất HGB/LightGBM study01

**Status**: COMPLETED. Kết luận này dùng study01 với đầy đủ safeguards đã ghi PLANNED, thay thế cách diễn giải kết quả LightGBM thăm dò trước đó.
**Author**: Phan Nguyễn Mai Phương — MSHV 25002711.
**Execution**: 108 inner + 20 outer + 4 final = 132 fresh fits; 473,43 giây trên Windows. Source identity `c10425d965e3f7d0411d14a56d15526ff33373c8e2ee077467b3aa38950ce90a`.
**Outer outcome**: LightGBM uncapped pooled trajectory RMSE29,653172 so với HGB uncapped29,806846; delta−0,153674, paired95%CI[−0,479373;0,185828]. Endpoint mean-engine MAE9,943892 so với9,891336; NASA20,988312 so với6,883768. On-time0,98 so với0,94, early-alarm burden0,055359 so với0,052159. RMSE chưa chứng minh cải thiện; endpoint và burden chưa đạt điều kiện không xấu hơn.
**Cap125 arm**: LightGBM RMSE36,098799 so với HGB36,208014; delta−0,109215, CI[−0,188614;−0,025010]. Endpoint MAE9,333136 so với9,321925, NASA3,859042 so với3,683556; CI của cả hai chênh lệch đều có cận trên dương. Lợi thế RMSE cùng cap không đủ để chọn.
**Selection/gate**: Inner CV chọn HGB uncapped ở cả 5 outer folds và final selection; selected_pipeline trùng HGB uncapped. `promotion_gate_met=false`; không chọn LightGBM để vận hành và không thay published model/registry. Không yêu cầu rerun Linux cho ứng viên chưa đạt.
**Test reporting only**: LightGBM uncapped MAE13,037866/RMSE19,716541/NASA3554,088 so với HGB13,753450/20,933433/3744,351. LightGBM cap125 MAE9,334126/RMSE12,817272/NASA214,655 so với HGB9,375761/12,549297/212,123. Test không được dùng để thay lựa chọn/gate.
**Verification**: 22 focused tests đạt. Read-only verifier xác nhận 132 fits, provenance, engine separation, raw truth, inner selections/policies, paired bootstrap, exact final model replay và test ordering. `verification.json` status=passed; verifier không fit model hoặc đọc raw test labels.
**Limits**: Follow-up trên folds/test đã xem trong lịch sử; endpoint panel50/70/90% được dựng hồi cứu. Bootstrap10.000 theo engine conditional trên fixed predictions, giữ multiplicity; không bao gồm refitting/seed variability hoặc hiệu chỉnh nhiều đối chứng. TCN chỉ có pilot8epochs, chưa có kết quả xác nhận; không loại toàn bộ kiến trúc từ pilot.
**Evidence**: [Báo cáo study01](../outputs/experiments/lightgbm_comparison/20261005_study01/README.md), [verification](../outputs/experiments/lightgbm_comparison/20261005_study01/verification.json), [protocol](LIGHTGBM_COMPARISON_PROTOCOL.md).

### 2026-10-05 — HGB gated blend study01

**Status**: PLANNED, trước khi chạy phép phối hợp trên dữ liệu thực nghiệm.
**Author**: Phan Nguyễn Mai Phương — MSHV 25002711.
**Hypothesis**: Smooth gate theo RUL uncapped dự đoán có thể dùng lợi thế cap125 ở một số vùng gần hỏng, đồng thời tránh giảm dự đoán quá nhiều ở phần còn xa hỏng. Không giả định phương án phối hợp thắng.
**Previous diagnostic**: Trộn cố định cap125 trọng số0,1/0,2/0,3 trên final-inner làm RMSE28,8350 tăng thành28,8709/29,0915/29,4927; endpoint NASA giảm. Đây chỉ là kiểm tra thăm dò trước proposal, không có promotion evidence.
**Change**: Hai HGB cố định, thay duy nhất prediction-combination rule. `r=clip((U-u)/(U-L),0,1)`, `w=M*r²*(3-2*r)`, `p=(1-w)*u+w*c`; gate chỉ dùng raw-scale current predictions, không truth/lifetime/future/true-RUL regions.
**Grid**: lower{60,90} × upper{125,150} × max_capped_weight{0.25,0.5,0.75,1}=16candidates; uncappedfallback; cap125reportingreference. Grid giữ nguyên sau khi nhìn outer readouts.
**Source and budget**: Derived từ verified LightGBM20261005_study01, cùng169causalhistoryfeatures/window30/engine-total-one weights/HGB200iterations. Reuse48HGBfits:30outer-inner+10outer+6final-inner+2finalmodels; freshfits0. Verify source/input/lock/coverage/hash trước dùng. Không tái dùng lựa chọn LightGBM.
**Selection**: 5outer/3inner group theoengine; 16gates mỗiinnerstage, warning threshold30/35/40/45/50 persistence1/3/5 lead30 target0.9. EndpointMAE/NASA/burden<=uncapped, on-time>=uncapped, margins0. MintrajectoryRMSE trong feasible, tie-break endpointMAE/candidateID; chỉ selected nếu RMSEinner thấp hơnuncapped, elsefallback. Lưu diagnostic gated_blend tách selected_pipeline.
**Gate and reporting**: Endpointpanel50/70/90 hồi cứu, rawtruth. Bootstrap10kengine seed42 giữmultiplicity/pooledSSE+counts. Selected vsuncapped: trajectoryupper95%CI<0, endpointMAE/NASA/burdenupper<=0, on-time lower>=0. Samecap comparison reportingonly; no testselection.
**Provenance**: Freeze newsource/importclosure/protocol/config/sourceartifactdigests trướcselection; save2basemodels/spec/policy/finaltestpreds và exactsavedreplay beforetestlabels; outerdecisionfreeze. Historicalfolds/test inspected; conditionalCI excludesrefit/seedvariability/multiple-testing correction. No independentblindclaim.
**Expected**: Focused synthetic tests, engineering smoke, full derived study, independentverifier and resultreport. Candidate fail khôngchọn; no publishedmodel/registrychange/commit/push; khôngyêucầu rerunLinux.
**Namespace**: outputs/experiments/gated_blend_comparison/20261005_study01.
**Protocol**: [GATED_BLEND_PROTOCOL.md](GATED_BLEND_PROTOCOL.md).

**Pre-run correction (trước khi đánh giá grid)**: Rà soát độc lập phát hiện HGB uncapped source outer4 dùng warning threshold40/persistence3. Assumption fixed40/p1 ở mục PLANNED trên là sai cho mọi stage. References mới giữ đúng policy source mỗi stage (và tái tính theo cùng inner calibration rule), để prediction, metrics và warning baseline đồng nhất với study01. Grid16, source HGB, inner split, selection safeguards và paired CI gate không đổi. Correction không dựa kết quả phối hợp mới.

**Engineering smoke interruption**: `20261005_smoke` dừng trong provenance snapshot copy với WinError3 do đường dẫn source_artifact_snapshot lồng quá dài Windows. Chưa đánh giá gate/selection/outer/test. Sửa layout snapshot thành flat short hashed filenames và ghi explicit original-path mapping có kiểm collision; giữ grid16, source48fits và gate. Chạy smoke mới trước science, không resume thư mục dở.

**Engineering full-run interruption (study01)**: Full grid đã hoàn tất 5 outer folds nhưng dừng ở final-inner gate 14 vì `PermissionError [WinError 5]` khi atomic replace `progress.tmp.json` thành `progress.json` trên Windows. Chưa hoàn tất final selection, model replay hoặc test evaluation; chưa mở raw test labels. Giữ partial artifacts; không dùng readout dở để đổi grid hoặc gate. Sửa retry có giới hạn cho PermissionError tạm thời của thao tác replace, kiểm hồi quy, chạy smoke03 rồi full study02; không resume hoặc ghi đè study01. Mọi scientific choices giữ nguyên.

### 2026-10-05 — Kết quả hoàn tất HGB gated blend study02

**Status**: COMPLETED; không đạt promotion gate, không chọn gated blend để vận hành.
**Author**: Phan Nguyễn Mai Phương — MSHV 25002711.
**Execution**: 16 gates ở mỗi trong 6 inner stages, 5 outer/3 inner grouped folds, 10.000 paired engine bootstrap draws, seed42. Reuse48 HGB fits (30 outer-inner +10 outer +6 final-inner +2 final models), fresh fits0; 122,66 giây trên Windows. Source identity `35c5ca253b77062f4f6065d0ef90b3c4a143fa7e5c256e2305ecc4afe7230d77`.
**Selected pipeline outcome**: Outer trajectory RMSE29,802876 so với HGB uncapped29,806846; delta−0,003970, CI[−0,014889;0,005437]. Endpoint mean-engine MAE9,897053/NASA6,897570 so với9,891336/6,883768; delta CI upper lần lượt0,047881/0,057591. On-time0,94 bằng baseline; burden0,052043 so với0,052159 nhưng delta CI upper0,000778. Chỉ on-time control đạt; promotion_gate_met=false.
**Diagnostic outcome**: Gated diagnostic RMSE29,766509, delta−0,040337, CI[−0,078115;−0,007778]; endpoint MAE9,634579/NASA6,771263 cải thiện. Burden tăng thành0,053870, delta CI[−0,000627;0,004159]. Diagnostic arm có 4 folds không feasible, không thay thế selected pipeline từ outer readout.
**Selection**: Outer folds1/2/4/5 fallback HGB uncapped; fold3 chọn L90/U125/M0,25. Final-inner chọn L90/U125/M0,5, threshold40/persistence1; inner RMSE28,788977 so với28,835035. Final-inner đủ điều kiện không thay thế outer promotion decision.
**Test reporting-only**: Final blend MAE13,544074/RMSE20,896506/NASA3738,899 so với HGB uncapped13,753450/20,933433/3744,351. Hai models/spec/policies/final predictions frozen và exact reload replay trước khi mở test labels; không đổi outer gate sau test.
**Engineering**: Smoke02/03 dry probes31/31 checks đạt. Full independent verifier32/32 checks đạt, không fit hoặc đọc raw test labels. 63 focused tests project chính đạt; copied inference tái tạo chính xác4 procedures×100 endpoint rows, max delta0. Source và artifacts copy được đối chiếu SHA256; published model/registry, source runs và dependency files được bảo toàn.
**Technical interruptions**: Smoke đầu bị Windows long path; full study01 bị atomic replace PermissionError khi ghi progress. Giữ artifacts dở, sửa tên snapshot ngắn với mapping và retry có giới hạn, chạy run IDs mới. Scientific grid/controls giữ nguyên sau lỗi; study02 là readout hoàn tất.
**Learning**: Smooth prediction-only gate có tín hiệu RMSE/endpoint nhỏ ở diagnostic, nhưng warning burden và khả năng chọn ổn định giữa folds chưa đạt mục tiêu. Không chọn hoặc yêu cầu Linux rerun ứng viên chưa đạt. Folds/test đã xem trước đây; CI conditional trên fixed predictions, không refit/seed variability hay multiple-testing correction.
**Evidence**: [Báo cáo study02](../outputs/experiments/gated_blend_comparison/20261005_study02/README.md), [verification](../outputs/experiments/gated_blend_comparison/20261005_study02/verification.json), [delivery](../outputs/experiments/gated_blend_comparison/20261005_study02/delivery.json), [protocol](GATED_BLEND_PROTOCOL.md).

### 2026-10-06 — HGB multi-scale và initial-profile

**Status**: PLANNED, trước khi chạy smoke và scientific study mới.
**Author**: Phan Nguyễn Mai Phương — MSHV 25002711.
**History review**: Window10 riêng lẻ, low-RUL weighting, target cap, history, XGBoost, LightGBM và prediction-only blend đã được kiểm tra; chưa có đúng tổ hợp window10+window30 hoặc initial30 profile42 trên HGB169 và cùng nested protocol.
**Hypothesis**: Thêm trailing10 statistics hoặc mean/std sensor của 30cycles đầu có thể giảm trajectory RMSE ở RUL cao mà giữ endpoint/warning safeguards. Không giả định có cải thiện.
**Change**: Ba arm cố định: hgb_baseline169 (169features), hgb_multiscale10_30 (295=baseline+126window10), hgb_initial_profile42 (211=baseline+42initial30mean/sample-std). Arm initial là ablation riêng, không kết hợp cùngwindow10. 21window10last trùngwindow30last; báo rõ 105cột còn lại chứa thêm statistics.
**Config**: Raw labels, cùngcycle>=30 keypanel, engine-total-one weights, HGB200/lr0.05/leaves31/minleaf20/L2=0/noearlystop, 5outer×3inner theoengine, seed42. Warningthreshold30/35/40/45/50,persistence1/3/5,lead30,target0.9. Mọimargins0.
**Selection**: Min pooledtrajectoryRMSE trong inner arms đủ endpointMAE/NASA/on-time/burden safeguards; fallbackbaseline169. Outer readout không chọn lại.
**Gate**: Pairedenginebootstrap10.000 giữmultiplicity/pooledSSE+counts; selectedpipeline vsbaseline: trajectoryupper95%CI<0,endpointMAE/NASA/burdenupper<=0,on-timelower>=0. Finalmodels/specs/policies/testpredictions vàexactreloadreplay trướcđọctestlabels; testreportingonly.
**Baseline**: HGB169 outertrajectoryRMSE29.806846,endpointmean-engineMAE9.891336,NASA6.883768,on-time0.94,burden0.052159. Sốlịch sử này làcontext, khôngdùngthaynewouterreadout.
**Limits**: Historicallyinspectedfolds/test; retrospectiveendpoints; conditionalCI excludesrefitting/seedvariability/multiplicityadjustment. Khôngtựthaypublishedmodel/registry, khôngcommit/push, khôngchạyTCN/Linux.
**Expected**: Focusedleakage/contracts/tests,smoke,fullstudy,independentverification vàbáocáokếtluận. Nếukhôngđạtgate thìgiữbaseline; khôngnớigate.
**Namespace**: outputs/experiments/multiscale_comparison/20261006_study01.
**Protocol**: [MULTISCALE_FEATURE_PROTOCOL.md](MULTISCALE_FEATURE_PROTOCOL.md).

### 2026-10-06 — Kết quả HGB multi-scale study02

**Status**: COMPLETED; full scientific run đã hoàn tất, promotion gate không đạt.
**Author**: Phan Nguyễn Mai Phương — MSHV 25002711.
**Execution**: 69 CV cache records + 3 final models = 72 fits; HGB200, 5 outer × 3 inner grouped engine folds, seed42, paired bootstrap10.000. Study01 dừng sớm ở một vài cache fits và không dùng để báo cáo; study02 là run hoàn chỉnh.
**Outer outcome**: Baseline169 trajectory RMSE29,806846. Multi-scale10+30 RMSE29,480160; delta−0,326685, CI[−0,688575;0,059603]. Endpoint MAE9,987586 so với9,891336, NASA7,566429 so với6,883768, on-time0,91 so với0,94, burden0,052726 so với0,052159. Initial-profile42 RMSE33,128001, endpoint MAE13,172342, NASA41,096410, burden0,072444. Outer selected pipeline fallback baseline ở cả5 folds; promotion_gate_met=false.
**Final-inner/test reporting**: Final inner chọn multi-scale10+30 với trajectory RMSE28,491274 so với baseline28,835035 và policy threshold45/persistence5. Test reporting-only: multi-scale RMSE19,916635, MAE13,504225, NASA2321,634; baseline RMSE20,933433, MAE13,753450, NASA3744,351. Test không dùng để mở gate hoặc đổi published pipeline.
**Verification**: Independent verifier `verification.json` status=passed; 41/41 checks đạt, gồm parity 23 baseline CV vectors và final model gốc, feature causality/common panel, engine separation, cache/source hashes, policy selection, pooled bootstrap, model replay và test ordering. Không thay published model/registry, không commit/push.
**Learning**: Multi-scale có tín hiệu giảm trajectory RMSE điểm trên outer, nhưng chưa giữ được endpoint và on-time controls với CI yêu cầu. Initial-profile42 không phù hợp trong protocol này. Giữ HGB baseline169 làm pipeline được promote; ghi multi-scale như hướng thăm dò có tín hiệu nhưng chưa đủ bằng chứng.
**Evidence**: [Báo cáo study02](../outputs/experiments/multiscale_comparison/20261006_study02/RESULTS.md), [verification](../outputs/experiments/multiscale_comparison/20261006_study02/verification.json), [protocol](MULTISCALE_FEATURE_PROTOCOL.md).

### 2026-10-07 — Kết quả similarity-based RUL pilot FD001

**Status**: COMPLETED, screening-only; nhánh không đạt điểm dừng và không mở full study.
**Author**: Phan Nguyễn Mai Phương — MSHV 25002711.
**Execution**: 3 GroupKFold inner folds theo engine, 17.731 common rows, 100 engines; library rows lần lượt 11.853/11.755/11.854; khoảng cách block-mean 30 cycles, batch128, 4 threads; 19,40 giây trên Windows. Không đọc test labels.
**Baseline reference**: HGB169 final-inner cached predictions được kiểm tra exact keys/raw y_true theo từng fold; pooled trajectory RMSE28,835035, endpoint mean-engine MAE10,550971, NASA20,516150, on-time0,94, burden0,055746.
**Outcome**: Similarity k=5 pooled trajectory RMSE37,172962, endpoint MAE20,889846, NASA770,941591, on-time0,91, burden0,073286; k=10 RMSE36,543007, endpoint MAE20,843944, NASA103,859812, on-time0,92, burden0,099723. Cả hai arm thua HGB ở cả3 folds và không đạt bất kỳ control nào trong panel.
**Decision**: `pilot_continue_candidate=false` cho k=5 và k=10; dừng nhánh similarity. Không chạy full nested study, không đọc test, không thay published model/registry.
**Learning**: Với biểu diễn block-mean sensor và nearest donor engine cố định này, similarity retrieval làm mất thông tin RUL cần thiết trên FD001; không tiếp tục bằng cách mở thêm k, DTW, PCA hoặc tuning sau khi xem kết quả.
**Evidence**: [RESULTS.md](../outputs/experiments/similarity_pilot/20261007_pilot02/RESULTS.md), `pooled_metrics.csv`, `fold_metrics.csv`, `decision.csv`; [protocol](SIMILARITY_PILOT_PROTOCOL.md). Pilot01 was the same scientific result before adding the final cache checksum assertion; pilot02 is the canonical receipt.

### 2026-10-07 — Similarity-based RUL pilot FD001

**Status**: PLANNED, trước mọi code chạy/smoke mới.
**Author**: Phan Nguyễn Mai Phương — MSHV 25002711.
**Hypothesis**: Đối chiếu mức sensor và hình dạng đoạn suy giảm30cycles với fitting engines có thể cải thiện RUL so với HGB169. Khác các biến thể tree/features/blend đã thử; chưa có bảo đảm cải thiện.
**Change**: StandardScaler fit-only engine-total-one; loại sensor std<=1e-8; window30 chia6blockmeans5cycles; mean-square distance; một best window trên mỗi donor engine; k={5,10}, inverse-distance weights. Không PCA/HI/DTW/deep learning hoặc mở thêm grid.
**Config**: Common17731trainkeys cycle>=30; GroupKFold3 shuffleFalse giống final-inner reference; reuse HGB169 matched cached vectors với hash/IDs/keys/rawtargets checks. Batch128 float64,4BLASthreads;3library fits,6readouts,0freshHGBfits;10phút scientificbudget.
**Screen**: PooledtrajectoryRMSE giảm, ít nhất2/3folds giảm, endpointMAE/NASA và calibratedOOF on-time/burden không xấu đi, margins0. Khôngđạt thìdừngnhánh; đạt chỉhỗtrợfullstudy, khôngpromotion. Warning OOF calibration/scoring cùngpanel khôngphảibằngchứngheldout. Khôngchạyfulltựđộng.
**Baseline**: Final-inner HGB169 trajectoryRMSE28.835035, endpointmean-engineMAE10.550971/NASA20.516150; chỉcontext, runner tínhlạitrênmatchedpanel.
**Expected outcome**: Hoàn tất pilotCPU nhỏ, kếtluận tiếp tục haydừng từmetrics; khôngthaymodel/registry hoặc mởrộng project. Testkhôngđọc; khôngcommit/push.
**Protocol**: [SIMILARITY_PILOT_PROTOCOL.md](SIMILARITY_PILOT_PROTOCOL.md).
