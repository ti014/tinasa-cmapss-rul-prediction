# NASA C-MAPSS FD001

FD001 là bộ dữ liệu chính của báo cáo cuối kỳ. Bộ dữ liệu mô phỏng một điều kiện
vận hành (Sea Level) và một dạng suy giảm (High-Pressure Compressor degradation).

## Nguồn

- NASA: https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data
- Gói đã tải: https://data.nasa.gov/docs/legacy/CMAPSSData.zip
- Bản công bố có checksum: https://zenodo.org/records/15346912
- MD5 của gói ZIP đã kiểm tra: `79a22f36e80606c69d0e9e4da5bb2b7a`.

Repository chỉ sử dụng FD001 và tài liệu gốc trong gói ZIP; không ghép
FD002–FD004. Các file dữ liệu giữ nguyên nội dung từ nguồn.

## Tệp và cách hiểu

- `train_FD001.txt`: lịch sử vận hành đến khi hỏng của 100 động cơ.
- `test_FD001.txt`: lịch sử của 100 động cơ khác, bị cắt trước khi hỏng.
- `RUL_FD001.txt`: 100 nhãn RUL tại chu kỳ cuối của mỗi động cơ test;
  dòng thứ i tương ứng động cơ test có ID i.
- `readme.txt`: mô tả gốc của toàn bộ gói C-MAPSS.
- `Damage Propagation Modeling.pdf`: bài báo gốc đi kèm.

Các file train và test không có header; các giá trị được phân tách bằng
whitespace. Mỗi dòng có 26 cột: engine ID, cycle, 3 operational settings và 21
sensor readings. README gốc ghi nhầm mục cuối là "sensor measurement 26"; cột
26 thực tế chứa sensor measurement 21.

ID động cơ chỉ có ý nghĩa trong từng split. Không nối lịch sử train và test có
cùng ID thành một động cơ. RUL được đo bằng cycle, không phải giờ hoặc ngày.

Trong tập train, RUL tại cycle `t` bằng cycle cuối của động cơ trừ `t`. Ví dụ,
nếu động cơ hỏng ở cycle 200 thì RUL tại cycle 170 là 30 cycles. Không áp dụng
công thức này cho test vì mỗi chuỗi test kết thúc trước khi hỏng; việc tính RUL
phải sử dụng nhãn đi kèm và độ lệch tại điểm quan sát cuối.

Bộ dữ liệu AI4I trước đây được lưu tại
`../archive/ai4i/predictive_maintenance.csv`; bộ dữ liệu này không thuộc phạm vi
phân tích chính.
