**Phan Nguyễn Mai Phương — MSHV 25002711**

# LightGBM nested pilot

Bốn predeclared arms dùng 169 frozen features, cùng 5 outer × 3 inner engine folds và raw RUL. Chưa đọc test labels và chưa có promotion gate/warning panel.

```text
       family  outer_rmse  outer_mae  outer_endpoint_mae
  lgbm_cap125   36.413178  21.439044            4.753449
lgbm_uncapped   30.235973  18.721949            5.523066
```

## Đối chiếu với HGB uncapped

Trong cả năm outer folds, inner selection chọn `lgbm_uncapped_base`. Khi ghép predictions theo engine/cycle với HGB uncapped, pooled trajectory RMSE là **29,842** cho LightGBM và **29,807** cho HGB; delta **+0,034**, paired engine bootstrap 95% CI **[-0,280; 0,354]**. Endpoint cuối của toàn trajectory có MAE **2,435** so với **2,417**; delta **+0,018**, CI **[-0,095; 0,128]**.

Kết quả này không cho thấy cải thiện có ý nghĩa trong protocol hiện tại. Pilot 300 trees từng có RMSE thấp hơn một chút, nhưng đó là readout thăm dò khác budget; full nested run đã đóng băng 200 trees để đối chiếu với HGB.

LightGBM cap125 có RMSE **36,413**, cao hơn LightGBM uncapped **30,236**. Chưa đọc test labels, không chọn model từ test và không thay published registry.

[Outer metrics](outer_metrics.csv) · [Inner selection](inner_selection.csv) · [Summary](summary.csv)
