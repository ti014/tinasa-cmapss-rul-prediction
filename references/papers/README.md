# Tài liệu PDF để đọc offline

Cập nhật ngày 20/09/2026: **đã có 6/8 bài**, gồm 5 PDF tải từ nguồn công khai và 1 PDF sao chép từ bộ dữ liệu có sẵn. Tổng dung lượng PDF khoảng 5,36 MB.

## File và nguồn tải

| Mã | File trong repository / trạng thái | Số trang | Nguồn |
| --- | --- | --- | --- |
| R1 | Saxena 2008 — Damage Propagation (không lưu trong repository) | 9 | Có trong gói [NASA CMAPSSData.zip](https://data.nasa.gov/docs/legacy/CMAPSSData.zip) |
| R2 | **Chưa tải được** — Lei 2018, Machinery health prognostics | — | [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0888327017305988) yêu cầu quyền truy cập; chưa tìm thấy bản toàn văn công khai từ tác giả hoặc trường |
| R3 | Ramasso & Saxena 2014 — C-MAPSS Review | 11 | [PHM Society PDF](https://papers.phmsociety.org/index.php/phmconf/article/download/2512/1473) |
| R4 | Li, Ding & Sun 2018 — CNN RUL | 29 | [UC Merced / eScholarship PDF](https://escholarship.org/content/qt5ns8r3fs/qt5ns8r3fs.pdf) |
| R5 | Saxena 2010 — Prognostic Metrics | 20 | [PHM Society PDF](https://papers.phmsociety.org/index.php/ijphm/article/download/1336/324) |
| R6 | **Chưa tải được** — Zheng 2017, LSTM for RUL | — | [IEEE DOI](https://doi.org/10.1109/ICPHM.2017.7998311); chưa tìm thấy PDF công khai từ tác giả/đơn vị nghiên cứu |
| R7 | Akidau 2015 — Dataflow Model | 12 | [VLDB PDF](https://www.vldb.org/pvldb/vol8/p1792-Akidau.pdf) |
| R8 | Zaharia 2012 — Resilient Distributed Datasets | 14 | [USENIX PDF](https://www.usenix.org/system/files/conference/nsdi12/nsdi12-final138.pdf) |

R4 là author manuscript (bản thảo của tác giả) kèm trang bìa của repository, nên số trang PDF khác phân trang 1–11 của bản journal. Khi trích dẫn vẫn dùng metadata trong [bibliography.bib](../bibliography.bib).

## Kiểm tra file

- Sáu file có định dạng PDF hợp lệ; công cụ đọc PDF nhận diện được toàn bộ các trang và trích xuất được văn bản.
- Đã đối chiếu tiêu đề và tác giả với danh mục. R1 có SHA-256 trùng với bản trong `data/raw/`.
- [checksums.sha256](checksums.sha256) ghi các giá trị băm dùng để kiểm tra tính toàn vẹn của file trong repository; đây là hash được tính sau khi tải, không phải chữ ký xác thực của nhà xuất bản.
- Việc tải và kiểm tra file không có nghĩa là đã đọc toàn bộ nội dung; mức độ đọc được ghi riêng trong [reading-guide.md](../reading-guide.md).

R2 và R6 chưa có file PDF trong repository. Có thể bổ sung khi có bản từ tài khoản thư viện hoặc author manuscript công khai. Không lưu trang HTML hoặc abstract dưới đuôi PDF.

Ví dụ, để đọc phần time window cho FD001, mở R4 — Li, Ding & Sun (2018), rồi xem Sections 3.1 và 3.2 theo hướng dẫn đọc.
