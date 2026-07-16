"""
Vendored từ CLIP-LoRA (Zanella & Ben Ayed, CVPRW 2024) — loralib/utils.py

Chỉ giữ 3 hàm generic (thuần lọc theo chuỗi 'lora_' trong tên tham số, không
phụ thuộc kiến trúc CLIP) — dùng trực tiếp được cho MedCLIP đã gắn LinearLoRA.
Bỏ `apply_lora`, `save_lora`, `load_lora`, `INDEX_POSITIONS_*` vì các hàm đó
hardcode cấu trúc nn.MultiheadAttention/PlainMultiheadAttentionLoRA của CLIP
gốc — không áp dụng được cho Swin/BERT bên trong MedCLIP.
Giữ nguyên logic gốc, không sửa.
"""
from typing import Dict

import torch
import torch.nn as nn

from lora.layers import LoRALayer


def mark_only_lora_as_trainable(model: nn.Module, bias: str = "none") -> None:
    """Đóng băng mọi tham số, chỉ để ngỏ (requires_grad=True) tham số LoRA."""
    for n, p in model.named_parameters():
        if "lora_" not in n:
            p.requires_grad = False
    if bias == "none":
        return
    elif bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "lora_only":
        for m in model.modules():
            if isinstance(m, LoRALayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


def lora_state_dict(model: nn.Module, bias: str = "none") -> Dict[str, torch.Tensor]:
    """Chỉ giữ lại phần state_dict của tham số LoRA — checkpoint rất nhẹ
    (vài trăm KB - vài MB) thay vì lưu lại toàn bộ MedCLIP."""
    my_state_dict = model.state_dict()
    if bias == "none":
        return {k: my_state_dict[k] for k in my_state_dict if "lora_" in k}
    elif bias == "all":
        return {k: my_state_dict[k] for k in my_state_dict if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        for k in my_state_dict:
            if "lora_" in k:
                to_return[k] = my_state_dict[k]
                bias_name = k.split("lora_")[0] + "bias"
                if bias_name in my_state_dict:
                    to_return[bias_name] = my_state_dict[bias_name]
        return to_return
    else:
        raise NotImplementedError


def get_lora_parameters(model: nn.Module, bias: str = "none"):
    """Trả về list tham số LoRA để đưa vào optimizer (chỉ phần này được học)."""
    params = []
    for name, param in model.named_parameters():
        if bias == "none":
            if "lora_" in name:
                params.append(param)
        elif bias == "all":
            if "lora_" in name or "bias" in name:
                params.append(param)
        elif bias == "lora_only":
            if "lora_" in name:
                params.append(param)
                bias_name = name.split("lora_")[0] + "bias"
                if bias_name in model.state_dict():
                    bias_param = dict(model.named_parameters())[bias_name]
                    params.append(bias_param)
        else:
            raise NotImplementedError
    return params
