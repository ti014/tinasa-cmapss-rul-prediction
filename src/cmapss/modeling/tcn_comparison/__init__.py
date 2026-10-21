"""
So sánh TCN trên chuỗi sensor FD001 gốc; chỉ import PyTorch khi cần để nhánh cây chạy
được khi thiếu torch.
"""

from __future__ import annotations


from .contracts import TCNConfig, TCNSpec, specs

__all__ = ["TCNConfig", "TCNSpec", "specs"]
