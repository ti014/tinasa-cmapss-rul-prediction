**Phan Nguyễn Mai Phương — MSHV 25002711**

# LightGBM pilot — FD001

Pilot non-promotional trên frozen 169 causal features, cùng outer engine folds với HGB/XGBoost. Dùng raw RUL và engine-total-one weights; không đọc test labels, không thay registry.

```text
 procedure  fold_rmse_mean  fold_mae_mean
 lgbm_base       29.737241      17.909662
lgbm_reg10       30.479791      19.184178
```

Đây là screening result; full nested selection và paired promotion gate chưa chạy.
