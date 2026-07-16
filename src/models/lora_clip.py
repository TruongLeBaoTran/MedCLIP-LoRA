"""
Gắn LoRA vào CLIP gốc (nhiệm vụ 4 — đối chứng cho MedCLIP + LoRA).

Khác với apply_lora_medclip() (target Linear tên query/key/value tách rời),
CLIP gốc dùng nn.MultiheadAttention (Q/K/V gộp trong 1 tensor in_proj_weight)
— nên cần PlainMultiheadAttentionLoRA (vendored từ CLIP-LoRA, xem lora/layers.py)
thay vì LinearLoRA trực tiếp.

Đơn giản hoá so với apply_lora() gốc của CLIP-LoRA: luôn gắn LoRA cho TẤT CẢ
layer của encoder được chọn (không có INDEX_POSITIONS_TEXT/VISION chia theo
vị trí như bản gốc — không cần thiết cho mục đích đối chứng "toàn bộ encoder
LoRA" y hệt cách apply_lora_medclip() làm).
"""
import torch.nn as nn

from lora.layers import PlainMultiheadAttentionLoRA


def apply_lora_clip(
    clip_model,
    r: int = 2,
    alpha: int = 1,
    dropout: float = 0.25,
    target: str = "both",  # "vision" | "text" | "both"
    enable_lora=("q", "k", "v"),
) -> list:
    """Duyệt các ResidualAttentionBlock trong clip_model.transformer (text) và
    clip_model.visual.transformer (vision), thay nn.MultiheadAttention bằng
    PlainMultiheadAttentionLoRA. Trả về list tên layer đã chèn.
    """
    roots = []
    if target in ("text", "both"):
        roots.append(("text", clip_model.transformer.resblocks))
    if target in ("vision", "both"):
        roots.append(("vision", clip_model.visual.transformer.resblocks))

    inserted = []
    for root_name, resblocks in roots:
        for i, block in enumerate(resblocks):
            for child_name, submodule in list(block.named_children()):
                if isinstance(submodule, nn.MultiheadAttention):
                    new_layer = PlainMultiheadAttentionLoRA(
                        submodule, enable_lora=list(enable_lora), r=r, lora_alpha=alpha, dropout_rate=dropout
                    )
                    setattr(block, child_name, new_layer)
                    inserted.append(f"{root_name}.resblocks.{i}.{child_name}")
    return inserted
