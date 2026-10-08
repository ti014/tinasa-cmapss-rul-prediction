# Kết quả thử nghiệm mở rộng

**Học viên:** Phan Nguyễn Mai Phương · **MSHV:** 25002711

Thư mục này chứa kết quả các thử nghiệm cải thiện mô hình. Không thử nghiệm nào thay
đổi mô hình chính hay chính sách cảnh báo (xem Mục 6.7 báo cáo).

Repository chỉ theo dõi hai lần chạy còn được sử dụng:

| Lần chạy | Được đọc bởi |
|---|---|
| `rul_improvement/20260928_rebuild/` | Dashboard (tab Thử nghiệm cải thiện), notebook 05, `slides/extract_slide_data.py` |
| `followup/20260929/` | Dashboard (bảng thử nghiệm bổ sung), `slides/extract_slide_data.py` |

Các lần chạy khác (tree comparison, LightGBM, TCN, gated blend, multiscale, similarity,
audit, so sánh Windows/Linux, `rul_improvement/20260927_full`) được bỏ khỏi git từ ngày
08/10/2026 để repository gọn hơn. Chúng vẫn có trong lịch sử git; khôi phục bằng:

```powershell
git checkout 1e73c96 -- outputs/experiments/<tên thư mục>
```

Protocol và kết luận của từng thử nghiệm nằm trong [`experiments/`](../../experiments/README.md).
