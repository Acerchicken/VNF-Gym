"""File chuyển tiếp: tham số đã chuyển sang configs/config.py (thư mục configs/ ở gốc repo).

Giữ file này để mọi chỗ đang `from nfv_sim.config import ...` / `from .config import ...` vẫn chạy.
Đừng sửa tham số ở đây — mở configs/config.py.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:      # để import được configs/ dù chạy script từ thư mục nào
    sys.path.insert(0, _ROOT)

from configs.config import *  # noqa: E402,F401,F403
