"""
Bọc MedCLIPModel gốc để dùng device-agnostic (CPU hoặc GPU đều chạy được).

Lý do cần bọc riêng: `MedCLIPModel.encode_image` / `.encode_text` gốc
(medclip/modeling_medclip.py) hardcode `.cuda()` bên trong, sẽ crash ngay nếu
máy không có GPU. Ở đây gọi thẳng `vision_model` / `text_model` (bỏ qua lớp
wrapper hardcode-cuda đó) rồi tự chuyển device + tự L2-normalize — logic giống
hệt bản gốc, chỉ khác chỗ device.
"""
import os

import torch
from medclip import MedCLIPModel, MedCLIPVisionModelViT

from configs import config as cfg


def load_medclip(device: str = cfg.DEVICE) -> MedCLIPModel:
    """Load MedCLIP backbone Swin-tiny (vision) + BioClinicalBERT (text), tải
    checkpoint pretrain gốc của tác giả MedCLIP (cache về MEDCLIP_PRETRAINED_DIR).

    Lưu ý: `device` phải là giá trị ĐÃ resolve qua common.device.resolve_device()
    trước khi gọi hàm này — hàm này không tự đổi device, để tránh lệch device
    giữa model và các tensor được các hàm encode_*_safe() chuyển sau đó.
    """
    model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
    model.from_pretrained(input_dir=cfg.MEDCLIP_PRETRAINED_DIR)
    model.to(device)
    model.eval()
    return model


def encode_image_safe(model: MedCLIPModel, pixel_values: torch.Tensor, device: str) -> torch.Tensor:
    """Ảnh -> embedding đã L2-normalize (512 chiều). Device-agnostic."""
    pixel_values = pixel_values.to(device)
    embeds = model.vision_model(pixel_values, project=True)
    return embeds / embeds.norm(dim=-1, keepdim=True)


def encode_text_safe(model: MedCLIPModel, input_ids: torch.Tensor,
                      attention_mask: torch.Tensor, device: str) -> torch.Tensor:
    """Text (đã tokenize) -> embedding đã L2-normalize (512 chiều). Device-agnostic."""
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    embeds = model.text_model(input_ids, attention_mask)
    return embeds / embeds.norm(dim=-1, keepdim=True)
