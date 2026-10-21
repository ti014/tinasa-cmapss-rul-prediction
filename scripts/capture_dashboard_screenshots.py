"""Chụp ảnh dashboard cho sổ tay thuyết trình (handoff/assets/dashboard/).

Chạy dashboard ở cổng 8599 trước, rồi dùng Edge đã cài sẵn qua Playwright:
uv run --extra dashboard streamlit run app.py --server.headless true --server.port 8599
uv run --no-project --with playwright python scripts/capture_dashboard_screenshots.py handoff/assets/dashboard
"""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = "http://localhost:8599"
OUT = Path(sys.argv[1])
# (tên file, mục điều hướng, chiều cao ảnh cũ)
PAGES = [
    ("overview.png", "Tổng quan", 887, None),
    ("denoise.png", "Khử nhiễu", 930, "Khử nhiễu tín hiệu cảm biến bằng EWMA"),
    ("anomaly.png", "Xu hướng & bất thường", 1345, "Xu hướng suy giảm và dấu hiệu bất thường"),
    ("warning.png", "Cảnh báo sớm", 1163, "Cảnh báo sớm: RUL dự đoán"),
    ("maintenance_top.png", "Lịch bảo trì", 880, "Báo cáo dự đoán thời điểm cần bảo trì"),
]


def open_page(page, label: str, height: int, heading: str | None = None) -> None:
    page.set_viewport_size({"width": 1440, "height": height})
    radio = page.locator('[data-testid="stSidebar"] [role="radiogroup"] label').filter(has_text=label)
    for _ in range(4):  # Streamlit có thể bỏ qua click khi trang trước còn đang chạy lại
        radio.click()
        if not heading:
            break
        try:
            page.get_by_text(heading).first.wait_for(timeout=15000)
            break
        except Exception:
            page.wait_for_timeout(2000)
    else:
        raise RuntimeError(f"không mở được trang {label}")
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(4000)  # chờ biểu đồ Vega vẽ xong


with sync_playwright() as p:
    browser = p.chromium.launch(channel="msedge")
    page = browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=1)
    page.goto(URL)
    page.get_by_text("Điều hướng").wait_for(timeout=60000)
    for name, label, height, heading in PAGES:
        open_page(page, label, height, heading)
        page.screenshot(path=str(OUT / name))
        print("wrote", name)

    # Ảnh chi tiết động cơ 090: chọn động cơ rồi cắt vùng nội dung chính.
    open_page(page, "Lịch bảo trì", 2600, "Báo cáo dự đoán thời điểm cần bảo trì")
    box = page.get_by_text("Xem chi tiết động cơ test").locator("xpath=following::div[@data-baseweb='select'][1]")
    box.click()
    page.keyboard.type("090")
    page.keyboard.press("Enter")
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(4000)
    top = page.get_by_text("Kế hoạch bảo trì theo thời gian").bounding_box()["y"] - 20
    page.screenshot(path=str(OUT / "maintenance_engine090.png"),
                    clip={"x": 300, "y": top, "width": 1140, "height": 907})
    print("wrote maintenance_engine090.png")
    browser.close()
