# Experiment journal

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
