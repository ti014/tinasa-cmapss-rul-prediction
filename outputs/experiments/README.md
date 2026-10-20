# Kết quả thử nghiệm mở rộng

**Học viên:** Phan Nguyễn Mai Phương · **MSHV:** 25002711

Thư mục này chứa kết quả các thử nghiệm cải thiện mô hình. Không thử nghiệm nào thay
đổi mô hình chính hay chính sách cảnh báo (xem Mục 6.7 báo cáo).

Hai lần chạy sau được dashboard, notebook hoặc slide dùng trực tiếp:

| Lần chạy | Được đọc bởi |
|---|---|
| `rul_improvement/20260928_rebuild/` | Dashboard (tab Thử nghiệm cải thiện), notebook 05, `slides/extract_slide_data.py` |
| `followup/20260929/` | Dashboard (bảng thử nghiệm bổ sung), `slides/extract_slide_data.py` |

Các run khác vẫn được theo dõi khi chúng là bằng chứng được trích dẫn trong
`experiments/journal.md`, protocol, verifier hoặc được workflow khác sử dụng.
Không bỏ tracking một run chỉ vì dashboard không đọc nó.

`similarity_pilot/20261007_pilot01/` là bản lặp có cùng kết quả khoa học với
`similarity_pilot/20261007_pilot02/`; run thứ hai là receipt chuẩn và được journal
tham chiếu. Pilot01 được bỏ tracking để tránh giữ hai bản kết quả trùng nhau, nhưng
các file local hiện có không bị xóa. Có thể khôi phục từ commit lưu trữ:

```powershell
git restore --source=1e73c96 -- outputs/experiments/similarity_pilot/20261007_pilot01
```

Protocol và kết luận của từng thử nghiệm nằm trong [`experiments/`](../../experiments/README.md).
Quy tắc trong `.gitignore` chỉ áp dụng cho file chưa được theo dõi; muốn bỏ tracking
artifact khác, cần kiểm tra references trước rồi gỡ file khỏi index một cách có chủ ý.
