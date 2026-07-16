"""
Gắn LoRA vào MedCLIP — phần kỹ thuật cốt lõi của nhiệm vụ 3.

Vì sao cần viết mới (không dùng được `apply_lora()` gốc của CLIP-LoRA):
`apply_lora()` gốc chỉ nhận diện `nn.MultiheadAttention` (kiến trúc attention
của CLIP gốc). MedCLIP không dùng kiến trúc đó — vision encoder là HuggingFace
SwinModel, text encoder là HuggingFace BertModel, cả hai đều có Q/K/V là 3
`nn.Linear` TÁCH RỜI (attribute tên `query`/`key`/`value`), không phải 1
`nn.MultiheadAttention` gộp chung. Nhờ 2 nhánh có cùng kiểu đặt tên, chỉ cần
1 hàm generic duyệt theo TÊN attribute, không cần viết riêng cho vision/text.
"""
import torch.nn as nn

from lora.layers import LinearLoRA


def apply_lora_medclip(
    medclip_model,
    r: int = 2,
    alpha: int = 1,
    dropout: float = 0.25,
    target: str = "both",  # "vision" | "text" | "both"
    target_modules=("query", "key", "value"),
) -> list:
    """Duyệt vision_model/text_model của MedCLIP, thay các nn.Linear có tên
    nằm trong target_modules (mặc định Q/K/V của attention) bằng bản có LoRA.

    Trả về list tên đầy đủ các layer đã bị thay (để log/kiểm tra số lượng).
    """
    roots = []
    if target in ("vision", "both"):
        roots.append(("vision_model", medclip_model.vision_model))
    if target in ("text", "both"):
        roots.append(("text_model", medclip_model.text_model))

    inserted = []
    for root_name, root_module in roots:
        for module_name, module in root_module.named_modules():
            for child_name, child in list(module.named_children()):
                if child_name in target_modules and isinstance(child, nn.Linear):
                    new_layer = LinearLoRA(child, r=r, lora_alpha=alpha, dropout_rate=dropout)
                    setattr(module, child_name, new_layer)
                    inserted.append(f"{root_name}.{module_name}.{child_name}")
    return inserted
