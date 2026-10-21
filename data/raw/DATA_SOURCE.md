# NASA C-MAPSS FD001--FD004

Kho dữ liệu raw gồm đủ bốn subset NASA C-MAPSS. FD001 vẫn là bộ dữ liệu chính
của báo cáo cuối kỳ; FD002--FD004 được lưu để phục vụ các thực nghiệm chuyển miền
ở giai đoạn tiếp theo.

## Nguồn

- NASA: https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data
- Gói đã tải: https://data.nasa.gov/docs/legacy/CMAPSSData.zip
- Bản công bố có checksum: https://zenodo.org/records/15346912
- MD5 của gói ZIP đã kiểm tra: `79a22f36e80606c69d0e9e4da5bb2b7a`.

## Cách tải lại

```powershell
uv run --frozen python -m cmapss.data.acquire --dest data/raw
```

Lệnh tải gói ZIP, kiểm tra MD5 của gói và SHA-256 của 12 file dữ liệu, rồi ghi
`outputs/acquisition/manifest.json`. Đây là hồ sơ của lớp Bronze. File đã có trong
`data/raw` chỉ được giữ khi trùng nội dung; file khác nội dung làm lệnh dừng chứ không
bị ghi đè. Ví dụ: lần chạy ngày 09/10/2026 (giờ Việt Nam) ghi cả 12 file ở trạng thái `unchanged`.
Có thể truyền `--archive <đường dẫn ZIP>` để dùng gói đã tải sẵn.

Các file dữ liệu giữ nguyên nội dung từ nguồn. Kết quả, artifact và data contract
hiện tại vẫn chỉ áp dụng cho FD001; việc có mặt của FD002--FD004 không có nghĩa
pipeline đã được đánh giá trên các subset này.

## Tệp và cách hiểu

- `train_FD001.txt`: lịch sử vận hành đến khi hỏng của 100 động cơ.
- `test_FD001.txt`: lịch sử của 100 động cơ khác, bị cắt trước khi hỏng.
- `RUL_FD001.txt`: 100 nhãn RUL tại chu kỳ cuối của mỗi động cơ test;
  dòng thứ i tương ứng động cơ test có ID i.
- `train_FD002.txt`, `test_FD002.txt`, `RUL_FD002.txt`: subset sáu điều kiện vận
  hành và một fault mode.
- `train_FD003.txt`, `test_FD003.txt`, `RUL_FD003.txt`: subset một điều kiện vận
  hành và hai fault modes.
- `train_FD004.txt`, `test_FD004.txt`, `RUL_FD004.txt`: subset sáu điều kiện vận
  hành và hai fault modes.
- `readme.txt`: mô tả gốc của toàn bộ gói C-MAPSS.
- Bài báo gốc "Damage Propagation Modeling" (Saxena và cs., PHM08) không được lưu trong repository; tải theo liên kết ở phần Nguồn của bản ghi NASA.

## Giấy phép

Bản ghi NASA ghi "License not specified". Dữ liệu được phân phối lại nguyên vẹn, kèm ghi nhận nguồn, và không thuộc giấy phép MIT của mã nguồn. Cần đối chiếu lại với bản ghi nguồn trước khi dùng cho mục đích khác

Theo các file raw đã tải, FD004 có 249 train engines, 248 test engines và 248
test labels. Con số này khác dòng mô tả FD004 trong `readme.txt`; profile của
repository dùng số engine quan sát được và số dòng nhãn để bảo đảm mapping test
engine hợp lệ.

Các file train và test không có header; các giá trị được phân tách bằng
whitespace. Mỗi dòng có 26 cột: engine ID, cycle, 3 operational settings và 21
sensor readings. README gốc ghi nhầm mục cuối là "sensor measurement 26"; cột
26 thực tế chứa sensor measurement 21.

ID động cơ chỉ có ý nghĩa trong từng split. Không nối lịch sử train và test có
cùng ID thành một động cơ. RUL được đo bằng cycle, không phải giờ hoặc ngày.

Trong tập train, RUL tại cycle `t` bằng cycle cuối của động cơ trừ `t`. Ví dụ,
nếu động cơ hỏng ở cycle 200 thì RUL tại cycle 170 là 30 cycle. Không áp dụng
công thức này cho test vì mỗi chuỗi test kết thúc trước khi hỏng; việc tính RUL
phải sử dụng nhãn đi kèm và độ lệch tại điểm quan sát cuối.

