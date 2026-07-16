"""
Tiện ích console nhỏ: ép stdout/stderr sang UTF-8.

Console mặc định của Windows (cp1252) không encode được tiếng Việt có dấu,
làm crash mọi lệnh print() có dấu. Gọi force_utf8_stdout() ở đầu mỗi script
chạy trực tiếp (if __name__ == "__main__") để tránh lỗi này.
"""
import sys


def force_utf8_stdout():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except Exception:
                pass
