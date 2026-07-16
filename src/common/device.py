"""Xác định device thực tế sẽ dùng — dùng 1 lần ở đầu mỗi pipeline, rồi truyền
đi xuyên suốt (tránh mỗi hàm tự âm thầm đổi device gây lệch nhau)."""
import torch


def resolve_device(requested: str) -> str:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print(f"[device] Không có GPU, dùng CPU thay cho '{requested}'.")
        return "cpu"
    return requested
