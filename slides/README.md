# Slide trình bày FD001

Deck chính: [`cmapss_fd001_presentation.pptx`](cmapss_fd001_presentation.pptx)
Kịch bản demo dashboard: [`handoff/DEMO_SCRIPT.md`](../handoff/DEMO_SCRIPT.md).

Deck gồm 10 slide, bao phủ phạm vi dữ liệu, pipeline artifact, EDA và thống kê,
rolling features, kiểm soát data leakage, model selection, final evaluation,
PCA--ARIMA, reproducibility và giới hạn bàn giao.

Nguồn số liệu lấy từ các artifact đã kiểm chứng trong `outputs/`, đặc biệt:

- `outputs/modeling/tables/final_metrics.json`
- `outputs/modeling/tables/model_metrics.csv`
- `outputs/timeseries/run_info.json`
- `tmp/e2e-repro-final2/validation.json`

Deck đã được export bằng Artifact Tool, finalizer kiểm tra package integrity,
layout, font, import và 10 slide; toàn bộ slide đã được render để kiểm tra
clipping và overflow. Biên bản finalizer nằm trong thư mục tạm
`tmp/slides-finalizer/` và được tóm tắt trong `handoff/FINAL_HANDOFF.md`.
