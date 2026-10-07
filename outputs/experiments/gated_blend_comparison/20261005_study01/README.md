**Phan Nguyễn Mai Phương — MSHV 25002711**

# Lượt đầy đủ bị dừng do thao tác ghi file trên Windows

Ngày 05/10/2026, runner hoàn tất năm outer folds và ghi outer decision, nhưng dừng trong final-inner gate evaluation với `PermissionError [WinError 5]` khi thay `progress.tmp.json` thành `progress.json`. Final selection, saved-model replay và test evaluation chưa hoàn tất. Thư mục này không có kết quả hoàn tất hoặc verification receipt của full protocol.

Giữ artifacts để đối chiếu sự cố. Sửa thao tác atomic replace bằng retry có giới hạn cho lỗi chia sẻ file tạm thời trên Windows; không thay công thức, grid, dữ liệu, selection rule hoặc promotion gate. Chạy smoke mới và full run với ID mới; không resume hoặc ghi đè thư mục này. Không dùng outer readout của lượt dở để thay đổi lựa chọn thí nghiệm.
