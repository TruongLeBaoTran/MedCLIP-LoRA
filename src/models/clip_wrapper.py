"""
Bọc CLIP gốc (nhiệm vụ 4 — đối chứng cho MedCLIP + LoRA).

Dùng package `clip` cục bộ nằm sẵn trong CLIP-LoRA/clip/ (không pip install
openai-clip riêng — tái dùng đúng bản CLIP-LoRA đã vendor). CLIP gốc KHÔNG có
bug hardcode .cuda() như MedCLIP (clip.load() đã tự map_location đúng), nên
wrapper này đơn giản hơn medclip_wrapper.py nhiều.
"""
import os
import sys

import torch

from configs import config as cfg

if cfg.CLIP_LORA_REPO_DIR not in sys.path:
    # append (KHÔNG insert(0)) — CLIP-LoRA/lora.py (file) sẽ đè lên package
    # src/lora/ của mình nếu chèn lên đầu sys.path, vì trùng tên "lora".
    sys.path.append(cfg.CLIP_LORA_REPO_DIR)
import clip  # noqa: E402  (import sau khi chỉnh sys.path)

from prompts_bone import BONE_PROMPTS  # noqa: E402


def load_clip(device: str, backbone: str = None):
    """Trả về (model, clip_module) — clip_module để gọi clip.tokenize() ở nơi khác."""
    backbone = backbone or cfg.CLIP_BACKBONE
    model, _preprocess = clip.load(backbone, device=device, jit=False)
    model.eval()
    return model, clip


def encode_image_clip(model, pixel_values: torch.Tensor, device: str) -> torch.Tensor:
    pixel_values = pixel_values.to(device)
    embeds = model.encode_image(pixel_values)
    return embeds / embeds.norm(dim=-1, keepdim=True)


def encode_text_clip(model, tokenized_text: torch.Tensor, device: str) -> torch.Tensor:
    tokenized_text = tokenized_text.to(device)
    embeds = model.encode_text(tokenized_text)
    return embeds / embeds.norm(dim=-1, keepdim=True)


def build_text_features_clip(model, device: str, requires_grad: bool = False) -> torch.Tensor:
    """Trả về tensor [NUM_CLASSES, 512] đã L2-normalize — dùng LẠI đúng
    BONE_PROMPTS đã viết cho MedCLIP (prompts_bone.py) để công bằng khi so
    sánh: cùng nội dung mô tả, chỉ khác encoder xử lý.

    requires_grad=False (mặc định, dùng cho eval): tính dưới torch.no_grad().
    requires_grad=True (dùng khi train LoRA target="text"/"both"): giữ đồ thị
    đạo hàm để gradient chảy vào LoRA của text encoder — xem giải thích tương
    tự ở prompts_bone.py::build_text_features (nhánh MedCLIP).
    """
    class_embeds = []
    for cls_name in cfg.CLASS_NAMES:
        prompts = BONE_PROMPTS[cls_name]
        tokens = clip.tokenize(prompts)
        if requires_grad:
            embeds = encode_text_clip(model, tokens, device)
        else:
            with torch.no_grad():
                embeds = encode_text_clip(model, tokens, device)
        mean_embed = embeds.mean(dim=0)
        mean_embed = mean_embed / mean_embed.norm()
        class_embeds.append(mean_embed)
    return torch.stack(class_embeds, dim=0)
