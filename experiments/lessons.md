# Bài học từ thực nghiệm RUL

- 27/09/2026: Giảm MAE RUL thấp có thể làm tăng MAE toàn bộ. Ví dụ hgb_08
  giảm khoảng 9,75% low-RUL MAE trong inner CV, nhưng tăng overall MAE 0,3233.
  Nguồn: `20260928_rebuild/final_selection_metrics.csv`.
- Window ngắn không tự động tốt hơn. Window 30 được chọn ở cả năm outer
  runs và final selection trong đợt đã xét; không suy rộng ngoài search này.
- Quy trình chọn cấu hình thắng trong inner CV vẫn có thể kém hơn baseline
  ở outer evaluation. Cảnh báo trễ tăng từ 49 lên 53/100 engines trong đợt này.
- Khi muốn chấp nhận đánh đổi, chốt tiêu chí mới trước đợt tiếp theo. Không
  đổi luật sau khi xem outer/test rồi gọi đó là xác nhận độc lập.
- Cache cần gắn với config, split, input và source. Dry-run đã chứng minh
  có thể dùng lại 30 fits và từ chối cấu hình thay đổi.
