"""
Prompt văn bản cho 3 lớp X-quang xương, dùng cho zero-shot (task 2) và làm
"classifier" (so cosine similarity) cho cả few-shot LoRA (task 3).

Viết bằng tiếng Anh vì BioClinicalBERT (text encoder của MedCLIP) được train
trên văn bản y khoa tiếng Anh (MIMIC-CXR reports) — prompt tiếng Việt sẽ nằm
ngoài phân bố huấn luyện của model.
"""
import torch
from medclip import constants
from transformers import AutoTokenizer

from configs import config as cfg
from models.medclip_wrapper import encode_text_safe

BONE_PROMPTS = {
    "Normal": [
        "a bone x-ray with no abnormality",
        "a normal bone radiograph without any tumor or lesion",
        "an x-ray showing normal bone structure and density",
        "a plain radiograph of a healthy bone with no mass",
    ],
    "Benign": [
        "a bone x-ray showing a benign bone tumor",
        "a radiograph showing a benign bone lesion with well-defined margins",
        "an x-ray of a benign osteochondroma or simple bone cyst",
        "a bone radiograph with a slow-growing benign neoplasm",
    ],
    "Malignant": [
        "a bone x-ray showing a malignant bone tumor",
        "a radiograph showing an aggressive malignant bone lesion with irregular margins",
        "an x-ray of osteosarcoma or other malignant bone neoplasm",
        "a bone radiograph with signs of malignant bone destruction",
    ],
}

_tokenizer = None


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(constants.BERT_TYPE)
    return _tokenizer


def build_text_features(model, device: str, requires_grad: bool = False) -> torch.Tensor:
    """Trả về tensor [NUM_CLASSES, 512] đã L2-normalize — embedding trung bình
    (ensemble) của các prompt mỗi lớp, theo đúng thứ tự cfg.CLASS_NAMES.

    requires_grad=False (mặc định, dùng cho zero-shot/eval): tính dưới
    torch.no_grad(), không giữ đồ thị tính đạo hàm — tiết kiệm bộ nhớ.
    requires_grad=True (dùng khi train LoRA với target="text"/"both"): PHẢI
    giữ đồ thị tính đạo hàm để gradient chảy được vào LoRA của text encoder,
    vì lúc đó text embedding cũng phụ thuộc tham số đang học.
    """
    tokenizer = _get_tokenizer()
    class_embeds = []
    for cls_name in cfg.CLASS_NAMES:
        prompts = BONE_PROMPTS[cls_name]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=77)
        if requires_grad:
            embeds = encode_text_safe(model, inputs["input_ids"], inputs["attention_mask"], device)
        else:
            with torch.no_grad():
                embeds = encode_text_safe(model, inputs["input_ids"], inputs["attention_mask"], device)
        mean_embed = embeds.mean(dim=0)
        mean_embed = mean_embed / mean_embed.norm()
        class_embeds.append(mean_embed)
    return torch.stack(class_embeds, dim=0)
