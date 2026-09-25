# Kịch bản Thuyết trình 10 phút — Chi tiết Word-for-Word

> **Hướng dẫn sử dụng:** File này viết sẵn từng câu nói, kèm chỉ dẫn chuyển
> slide/dashboard ở dòng in nghiêng. Bạn có thể in ra giấy A4 để cầm đọc, hoặc
> học thuộc ý chính rồi diễn đạt tự nhiên. Tổng thời lượng: ~10 phút.

---

## Phần 1 — Giới thiệu và Bối cảnh (Slide 1–2, khoảng 1 phút)

*(Slide 1 — Bìa đề tài)*

"Kính chào thầy/cô và các bạn. Em tên là Phan Nguyễn Mai Phương. Hôm nay em
xin trình bày báo cáo cuối kỳ môn Phân tích dữ liệu lớn với đề tài: **Pipeline
dự đoán Remaining Useful Life trên tập dữ liệu NASA C-MAPSS FD001**.

Remaining Useful Life, viết tắt là RUL, là số chu kỳ vận hành còn lại trước khi
một động cơ hỏng. Bài toán này thuộc lĩnh vực bảo trì dự đoán (predictive
maintenance), và cách tiếp cận của em là supervised regression, tức hồi quy có
giám sát, vì biến mục tiêu là một con số liên tục."

*(Chuyển sang Slide 2 — Phạm vi và dữ liệu)*

"Dữ liệu em sử dụng là bộ C-MAPSS FD001, do NASA công bố. Bộ dữ liệu này mô
phỏng quá trình vận hành đến khi hỏng của động cơ turbofan, gồm:

- Tập train: 100 động cơ, tổng cộng 20.631 dòng quan sát, chạy hoàn chỉnh đến
  khi hỏng.
- Tập test: 100 động cơ khác, tổng cộng 13.096 dòng, bị cắt trước khi hỏng.
- File nhãn RUL_FD001.txt: chứa đúng 100 giá trị RUL thật tại thời điểm quan
  sát cuối của từng test engine.

Mỗi dòng quan sát có 26 trường: mã động cơ, cycle vận hành, 3 thông số vận hành
và 21 giá trị cảm biến. Toàn bộ dữ liệu không có giá trị thiếu và cycle luôn
liên tục từ 1 đến T_i.

Công thức xác định nhãn RUL trên tập train rất trực quan: nếu động cơ i hỏng
tại cycle T_i, thì tại cycle t bất kỳ, RUL bằng T_i trừ t. Ví dụ, động cơ kết
thúc tại cycle 200 thì tại cycle 150, RUL bằng 50. Công thức này chỉ áp dụng
cho train vì train có toàn bộ lịch sử; tập test bị cắt cụt nên nhãn thật được
giữ tách biệt trong file riêng.

Một điểm quan trọng cần lưu ý: FD001 chỉ đại diện cho một điều kiện vận hành và
một dạng lỗi máy nén áp suất cao, nên kết quả chỉ có giá trị trong phạm vi
FD001, chưa thể khái quát hóa cho động cơ ngoài thực địa."

---

## Phần 2 — Luồng xử lý dữ liệu và Big Data (Slide 3, khoảng 1 phút 30 giây)

*(Chuyển sang Slide 3 — Pipeline dữ liệu)*

"Toàn bộ pipeline dữ liệu của em tuân theo kiến trúc Medallion gồm ba tầng:
Bronze, Silver và Gold.

- Tầng Bronze giữ nguyên file thô bất biến của NASA, không chỉnh sửa.
- Tầng Silver chuyển dữ liệu sang định dạng Parquet có kiểu dữ liệu rõ ràng.
  Tầng này ghi ra hai file: observations.parquet chứa 33.727 dòng quan sát từ cả
  train và test, và train_targets.parquet chứa 20.631 dòng nhãn RUL tương ứng.
- Tầng Gold tổng hợp thông tin bậc cao: engine summary, rolling features dùng
  cho modeling, và bảng SQLite phục vụ truy vấn nhanh.

Về thành phần Big Data, em triển khai song song pipeline ETL bằng cả Pandas
(làm reference) và PySpark (làm validation). Sau đó, em chạy Parity Checker để
đối chiếu schema, khóa chính, số dòng và giá trị số học giữa hai hệ thống. Kết
quả: sai khác tuyệt đối lớn nhất giữa Pandas và PySpark trên cả 3 bảng
Silver/Gold đều bằng 0.0, nghĩa là hai hệ thống cho kết quả hoàn toàn tương
đương.

Ngoài ra, em cũng đo benchmark khả năng mở rộng trên Pandas: cùng hàm biến đổi
build_engine_summary, khi nhân dữ liệu lên 10 lần (337 nghìn dòng), thời gian
tăng tuyến tính lên khoảng 0.15 giây; khi nhân 100 lần (3.3 triệu dòng), thời
gian tăng lên khoảng 1.67 giây. Tuy nhiên, cần nhấn mạnh rằng phép đo này chỉ
phản ánh thời gian tính toán in-memory, không đại diện cho hiệu năng trên cụm
phân tán thực tế."

---

## Phần 3 — Phân tích Khám phá và Thống kê (Slide 4, khoảng 1 phút)

*(Chuyển sang Slide 4 — EDA và Thống kê)*

"Trước khi modeling, em thực hiện phân tích khám phá trên tập train. Kết quả cho
thấy tuổi thọ 100 động cơ train dao động từ 128 đến 362 cycles, trung vị là 199
cycles, trung bình 206 cycles. Trung bình lớn hơn trung vị cho thấy phân phối
hơi lệch về phía các chuỗi sống lâu.

Về chất lượng cảm biến, em phát hiện có 6 trong 21 cảm biến hoàn toàn hằng số
trên toàn bộ tập train, ví dụ sensor_1 luôn bằng 518.67, sensor_5 luôn bằng
14.62. Những sensor này không mang thông tin suy giảm nhưng em vẫn giữ nguyên
trong bảng dữ liệu để tuân thủ data contract, không loại bỏ tùy tiện trước khi
modeling.

Để kiểm tra xem sensor nào thực sự thể hiện sự suy giảm, em áp dụng suy luận
thống kê theo protocol đã chốt trước: so sánh giá trị trung bình của 30 cycles
đầu tiên với 30 cycles cuối cùng trong từng động cơ, rồi tính chênh lệch cặp.
Trên 100 động cơ nhân 21 sensor, em có tổng cộng 2.100 cặp quan sát.

Khoảng tin cậy 95% được tính bằng bootstrap 10.000 lần, lấy mẫu lại theo đơn vị
động cơ. Kiểm định Wilcoxon signed-rank với 100.000 hoán vị, sau đó hiệu chỉnh
đa kiểm định bằng Benjamini-Hochberg FDR ở mức 5%. Kết quả: 15 trong 21 sensor
bác bỏ giả thuyết không, cho thấy có sự thay đổi có ý nghĩa thống kê giữa giai
đoạn đầu và cuối vòng đời.

Một ví dụ đáng chú ý là sensor_6: p-value rất nhỏ (khoảng 0.000028) nhưng chênh
lệch trung bình chỉ là 0.00045 đơn vị. Đây là minh chứng rõ ràng rằng p-value
nhỏ không đồng nghĩa với kích thước hiệu ứng lớn."

---

## Phần 4 — Trích xuất Đặc trưng và Chống rò rỉ Dữ liệu (Slide 5, khoảng 1 phút 30 giây)

*(Chuyển sang Slide 5 — Rolling Features)*

"Bước tiếp theo là trích xuất đặc trưng. Em sử dụng cửa sổ trượt quá khứ
(trailing window) có độ dài cố định w bằng 30 cycles. Tại mỗi cycle t, cửa sổ
chỉ chứa 30 quan sát từ cycle t trừ 29 đến cycle t của cùng một động cơ. Không
bao giờ sử dụng quan sát tương lai.

Từ mỗi cửa sổ, với mỗi sensor trong 21 sensors, em tính 6 đặc trưng thống kê:

1. Giá trị mới nhất (last): chính là giá trị tại cycle t.
2. Trung bình (mean) trong 30 cycles.
3. Độ lệch chuẩn mẫu (std) với ddof bằng 1.
4. Giá trị nhỏ nhất (min).
5. Giá trị lớn nhất (max).
6. Hệ số góc xu hướng tuyến tính (slope): tính bằng hồi quy OLS theo vị trí k
   từ 1 đến 30 trong cửa sổ.

Tổng cộng: 21 sensor nhân 6 đặc trưng bằng 126 cột feature. Tập Gold rolling
features có 27.927 dòng, gồm 17.731 train windows và 10.196 test windows.

Về phòng chống rò rỉ dữ liệu, em đặt ra ba ranh giới an toàn:

- Thứ nhất, window không bao giờ trượt qua ranh giới giữa hai động cơ.
- Thứ hai, window không sử dụng bất kỳ quan sát nào ở tương lai.
- Thứ ba, file nhãn test RUL_FD001.txt tuyệt đối không được đọc trong bước
  này. Động cơ nào có dưới 30 cycles sẽ bị loại thay vì padding hoặc rút ngắn."

---

## Phần 5 — Lựa chọn Mô hình (Slide 6, khoảng 1 phút 30 giây)

*(Chuyển sang Slide 6 — Model Selection)*

"Sang phần lựa chọn mô hình. Em áp dụng 5-fold GroupKFold trên 100 động cơ
train, trong đó engine_id là group key. Nghĩa là mọi window thuộc cùng một
động cơ chỉ nằm trong đúng một fold. Mỗi fold chứa 20 động cơ.

Tại sao phải chia theo engine? Vì các window trong cùng một động cơ có phụ thuộc
theo thời gian; nếu chia ngẫu nhiên theo dòng, window ở cycle 100 của cùng engine
có thể rơi vào train trong khi window ở cycle 101 rơi vào validation, tạo ra rò
rỉ dữ liệu nghiêm trọng và khiến metric lạc quan giả tạo.

Ngoài ra, em gán trọng số mẫu cân bằng theo engine: mỗi engine nhận tổng trọng
số bằng 1, tức w_i_t bằng 1 chia cho n_i (số window của engine đó). Kỹ thuật
này ngăn các động cơ sống lâu, có nhiều window hơn, chi phối hàm mất mát.

Ba mô hình ứng viên:

1. Median Baseline: dự đoán giá trị trung vị có trọng số từ training fold, đạt
   mean engine MAE là 46.95 cycles.
2. Ridge Regression (alpha bằng 1): hồi quy tuyến tính có phạt L2, với
   StandardScaler chỉ fit trên training fold, đạt MAE 23.99 cycles.
3. HistGradientBoostingRegressor: mô hình phi tuyến với 200 iterations, learning
   rate 0.05, tối đa 31 leaf nodes và tối thiểu 20 samples mỗi leaf. Đạt MAE
   20.21 cycles, thấp nhất.

Tiêu chí chọn mô hình là mean engine-level MAE thấp nhất. Do đó,
HistGradientBoosting được chọn làm mô hình chính thức."

---

## Phần 6 — Đánh giá Cuối cùng trên Test (Slide 7, khoảng 1 phút 30 giây)

*(Chuyển sang Slide 7 — Final Evaluation, hoặc mở Dashboard trang Đánh giá RUL)*

"Sau khi chọn mô hình, em refit HistGradientBoosting trên toàn bộ 17.731 train
windows, rồi tạo đúng 100 dự đoán RUL, mỗi dự đoán ứng với cycle quan sát cuối
cùng của từng test engine. File final_predictions.parquet được ghi xong trước
khi đọc nhãn test. Đây là nguyên tắc bắt buộc: model không thể nhìn trộm đáp
án trước khi xuất dự đoán.

Các metric đánh giá cuối trên 100 test engines:

- Mean engine MAE: 15.35 cycles, với khoảng tin cậy bootstrap 95% là từ 12.73
  đến 18.09.
- RMSE: 20.54 cycles, khoảng tin cậy từ 17.34 đến 23.59.
- NASA Asymmetric Score: 1513.26.

Về công thức NASA score, hàm này phạt bất đối xứng: khi model dự đoán trễ hơn
thực tế (tức engine sắp hỏng mà model nói vẫn còn lâu), hệ số phạt chia cho 10,
rất nặng. Khi model dự đoán sớm hơn (tức engine còn khỏe mà model nói sắp hỏng),
hệ số chia cho 13, nhẹ hơn. Điều này phản ánh thực tế: dự đoán trễ gây nguy
hiểm hơn dự đoán sớm.

*(Chọn Engine 23 trên Dashboard)*

Để minh bạch, em chỉ ra trường hợp sai số lớn nhất: Engine 23 có RUL thật là
113 nhưng mô hình dự đoán 170, sai tới 57 cycles. Đây là trường hợp dự đoán quá
lạc quan (overestimation). Cả 5 ca sai số lớn nhất đều thuộc dạng này, chủ yếu
rơi vào nhóm engine có RUL lớn (trên 90 cycles). Ngược lại, nhóm engine gần hỏng
(RUL từ 0 đến 30 cycles), mô hình dự đoán rất chính xác với MAE chỉ 4.38
cycles."

---

## Phần 7 — Cảnh báo sớm (vẫn trên Slide 7 hoặc chuyển Dashboard, khoảng 45 giây)

*(Mở trang Cảnh báo sớm trên Dashboard)*

"Em cũng xây dựng phân tích cảnh báo sớm minh họa. Ngưỡng cảnh báo được chốt
trước ở mức RUL nhỏ hơn hoặc bằng 30 cycles. Khi dự đoán out-of-fold lần đầu
rơi xuống dưới ngưỡng này, hệ thống ghi nhận sự kiện cảnh báo.

Kết quả trên 100 động cơ train: 100% engine đều nhận được cảnh báo trước khi
hỏng, không có trường hợp bỏ sót. Cụ thể:

- 44 engine được cảnh báo trước ngưỡng thực tế.
- 7 engine được cảnh báo đúng thời điểm chạm ngưỡng.
- 49 engine được cảnh báo sau ngưỡng nhưng vẫn trước khi hỏng.

Thời gian cảnh báo sớm (lead time) có trung vị 30 cycles, nhỏ nhất là 19 và lớn
nhất là 55.

Cần lưu ý: đây là phân tích mô tả nội bộ trên tập out-of-fold vốn đã được dùng
để chọn mô hình, chưa phải kiểm định chính sách bảo trì độc lập."

---

## Phần 8 — Phân tích PCA–ARIMA Health Index (Slide 8, khoảng 1 phút)

*(Chuyển sang Slide 8 — Time-Series Analysis, hoặc mở Dashboard trang Health Index)*

"Bên cạnh dự đoán RUL trực tiếp, em thực hiện một phân tích chuỗi thời gian bổ
trợ bằng cách xây dựng chỉ số sức khỏe động cơ, gọi là Health Index.

Cách làm: em chọn trước 10 động cơ validation đại diện từ 5 fold. Sau đó, fit
StandardScaler và PCA 1 thành phần (PC1) chỉ trên 90 động cơ train còn lại,
rồi transform 10 engine được chọn. Điểm PC1 được định hướng lại sao cho Health
Index giảm dần theo tuổi thọ, phản ánh sự thoái hóa.

Trên mỗi engine, em chia chuỗi Health Index thành 70% đầu để fit mô hình ARIMA
và 30% cuối làm holdout đánh giá. Tổng cộng có 704 dòng dự báo. Bậc ARIMA(p,d,q)
được chọn bằng grid search AIC trên đoạn fit.

Kết quả: cả 10 engine đều hội tụ. MAE trung bình của ARIMA là 0.41 trên thang
chuẩn hóa, vượt trội so với baseline Drift (0.75) và Last-value (0.95).

*(Chọn Engine 70 trên Dashboard)*

Tuy nhiên, ở Engine 70, tỉ lệ bao phủ khoảng dự đoán 95% chỉ đạt 21.43%, rất
thấp. Em giữ nguyên con số này trong báo cáo vì nó phản ánh trung thực tính bất
định của chuỗi khi ARIMA chưa nắm bắt đầy đủ cấu trúc dữ liệu.

Kiểm định ADF trên các đoạn fit đều cho p-value rất cao (0.86 đến 0.999), không
bác bỏ tính không dừng. Ljung-Box điều chỉnh bậc tự do cho thấy Engine 69 và 70
vẫn còn tự tương quan trong phần dư.

Điểm quan trọng: Health Index forecast không phải RUL forecast. ARIMA được trình
bày như phân tích phụ, không thay thế mô hình RUL chính."

---

## Phần 9 — Tái lập và Giới hạn (Slide 9–10, khoảng 1 phút)

*(Chuyển sang Slide 9 — Reproducibility)*

"Về khả năng tái lập, toàn bộ pipeline có thể được tái tạo từ đầu bằng một lệnh
duy nhất. Lệnh này dựng lại output trong một thư mục hoàn toàn mới, rồi đối
chiếu 18 bảng số liệu với tolerance nhỏ hơn 10 mũ trừ 9, và kiểm tra 5 biểu đồ
khớp byte-identical. Bộ test tự động gồm 151 test case, đạt 100%.

Môi trường được khóa phiên bản bằng uv.lock: Python 3.12, Pandas 2.3, scikit-learn
1.9, statsmodels 0.14. Dashboard Streamlit hoạt động ở chế độ read-only, chỉ đọc
artifact đã sinh, không fit lại model và không truy cập file nhãn thô.

*(Chuyển sang Slide 10 — Giới hạn)*

Cuối cùng, em xin thẳng thắn nêu các giới hạn:

- FD001 là dữ liệu mô phỏng, chỉ đại diện 1 điều kiện vận hành và 1 dạng lỗi.
  Kết quả chưa chứng minh khả năng khái quát hóa sang động cơ thực tế.
- Cycle vận hành không tương đương với giờ bay hoặc thời gian lịch.
- Dự đoán RUL không tự động xác định lịch bảo trì tối ưu vì FD001 không cung
  cấp chi phí bảo trì hay downtime.
- Hướng mở rộng trong tương lai là kiểm tra độ nhạy với các kích thước window
  khác nhau (10, 20, 30 cycles) và thử nghiệm trên các tập FD002 đến FD004 để
  đánh giá tổng quát hóa."

---

## Phần 10 — Kết thúc (khoảng 15 giây)

"Dạ, đó là toàn bộ nội dung báo cáo cuối kỳ của em. Em xin cảm ơn thầy/cô và
các bạn đã chú ý lắng nghe. Em xin sẵn sàng nhận câu hỏi phản biện ạ."

---

# Phụ lục — Câu hỏi phản biện thường gặp và câu trả lời gợi ý

### 1. "Tại sao phải chia GroupKFold theo engine thay vì chia ngẫu nhiên?"

"Dạ, vì các window trong cùng một động cơ có phụ thuộc thời gian mạnh. Nếu chia
ngẫu nhiên theo dòng, window cycle 100 của engine A có thể rơi vào train trong
khi window cycle 101 rơi vào validation, gây rò rỉ dữ liệu và khiến metric
validation lạc quan giả tạo. GroupKFold đảm bảo mọi window của cùng engine chỉ
nằm trong một fold duy nhất."

### 2. "Có bị data leakage không?"

"Dạ, em đã thiết lập nhiều tầng phòng chống: scaler và PCA chỉ fit trên training
fold, window không trượt qua ranh giới engine, không sử dụng quan sát tương lai,
và nhãn test chỉ được đọc sau khi file final_predictions.parquet đã ghi xong.
Trạng thái này được ghi nhận trong file run_info.json với giá trị
test_label_join.status bằng completed_after_final_predictions."

### 3. "Tại sao chọn HistGradientBoosting mà không dùng Deep Learning?"

"Dạ, trong phạm vi báo cáo cuối kỳ này, em ưu tiên protocol đánh giá chặt chẽ
và khả năng tái lập hơn là đạt điểm số cao nhất. HistGradientBoosting đã vượt
trội so với Ridge và Median Baseline trong GroupKFold cross-validation, đạt MAE
20.21 cycles. Deep Learning (như CNN hoặc LSTM) là hướng mở rộng hợp lệ nhưng
đòi hỏi thêm hyperparameter tuning và thời gian huấn luyện đáng kể, nằm ngoài
phạm vi hiện tại."

### 4. "ARIMA có dự báo RUL không?"

"Dạ không. ARIMA chỉ dự báo quỹ đạo Health Index, là chỉ số sức khỏe tổng hợp
từ PCA. Health Index forecast không phải RUL countdown. Em trình bày ARIMA như
phân tích chuỗi thời gian bổ trợ để quan sát xu hướng thoái hóa, không phải mô
hình RUL thay thế."

### 5. "Kết quả có áp dụng được cho động cơ thực tế không?"

"Dạ, chưa thể kết luận được. FD001 là dữ liệu mô phỏng với 1 điều kiện vận
hành và 1 dạng lỗi. Để áp dụng thực tế, cần đánh giá trên dữ liệu thật với
nhiều điều kiện vận hành, nhiều dạng lỗi, và kết hợp chi phí bảo trì kinh tế
để thiết lập ngưỡng cảnh báo phù hợp."

### 6. "PySpark có thay thế Pandas trong modeling không?"

"Dạ không. PySpark được dùng để kiểm tra tính tương đương (parity) với Pandas ở
tầng ETL, nhằm chứng minh pipeline có thể mở rộng sang hệ thống phân tán mà
không thay đổi kết quả. Phần modeling vẫn chạy bằng Pandas và scikit-learn vì
quy mô FD001 chỉ vài MB, không cần cụm phân tán."

### 7. "Tái lập được bằng chứng nào?"

"Dạ, em có lệnh reproducibility chạy lại toàn bộ pipeline vào thư mục mới, so
sánh 18 bảng số liệu với tolerance 10 mũ trừ 9, kiểm tra 5 hình ảnh khớp
byte-identical, chạy dashboard loader, và bộ test 151 case đạt 100%. Môi trường
được khóa phiên bản bằng uv.lock."

### 8. "Tại sao coverage khoảng dự đoán ARIMA lại thấp?"

"Dạ, khoảng dự đoán 95% của ARIMA dựa trên giả định phần dư có phân phối chuẩn
và không còn tự tương quan. Tuy nhiên, kiểm định ADF cho thấy chuỗi không dừng
mạnh, và Ljung-Box cho thấy một số engine còn tự tương quan trong phần dư. Khi
giả định bị vi phạm, khoảng dự đoán có thể quá hẹp, dẫn đến coverage thấp. Em
giữ nguyên kết quả này để phản ánh trung thực giới hạn của phương pháp."

### 9. "Trọng số mẫu (sample weight) hoạt động thế nào?"

"Dạ, trong mỗi training fold, em gán cho mỗi động cơ tổng trọng số bằng 1, tức
mỗi window của engine i có trọng số bằng 1 chia cho số window n_i của engine đó.
Ví dụ, engine có 200 window thì mỗi window có trọng số 1/200, engine có 100
window thì mỗi window có trọng số 1/100. Nhờ vậy, engine sống lâu (nhiều window)
không chiếm ưu thế trong hàm mất mát so với engine sống ngắn."

### 10. "NASA score tính thế nào?"

"Dạ, hàm NASA score cộng dồn trên 100 test engines. Với mỗi engine, sai số có
dấu e bằng dự đoán trừ thực tế. Nếu e âm (dự đoán sớm), phạt bằng exp trừ e
chia 13 trừ 1, nhẹ hơn. Nếu e dương hoặc bằng 0 (dự đoán trễ hoặc đúng), phạt
bằng exp e chia 10 trừ 1, nặng hơn nhiều. Hàm này phạt bất đối xứng vì trong
thực tế, dự đoán trễ (không kịp bảo trì trước khi hỏng) nguy hiểm hơn dự đoán
sớm (bảo trì quá sớm, chỉ tốn chi phí)."
