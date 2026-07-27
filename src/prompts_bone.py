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


def build_text_features_per_prompt(model, device: str, requires_grad: bool = False) -> list:
    """Trả về list gồm NUM_CLASSES tensor, phần tử thứ k có shape [N_k, 512],
    mỗi dòng đã L2-normalize RIÊNG (không trung bình) — dùng cho PromptClassifier
    kiểu MedCLIP gốc (MedCLIP/medclip/modeling_medclip.py::PromptClassifier):
    so cosine từng prompt riêng lẻ trước, gộp N điểm/lớp bằng max/mean SAU đó
    ở nơi gọi (khác build_text_features() bên dưới, vốn gộp bằng cách trung
    bình EMBEDDING trước khi so cosine — dùng cho Task 3/predict.py, không đổi).

    Không giả định N giống nhau giữa các lớp (list of ragged tensors, không
    phải tensor [K,N,512] cố định) — an toàn nếu BONE_PROMPTS có số prompt
    khác nhau/lớp.
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
        class_embeds.append(embeds)  # [N_k, 512], mỗi dòng đã L2-normalize (encode_text_safe tự làm)
    return class_embeds


def build_text_features(model, device: str, requires_grad: bool = False) -> torch.Tensor:
    """Trả về tensor [NUM_CLASSES, 512] — 1 vector "đại diện"/lớp, dùng cho
    Task 3 (LoRA few-shot) và predict.py.

    Về mặt toán, đây CHÍNH XÁC là cơ chế "mean"/ensemble=True của
    PromptClassifier gốc MedCLIP (MedCLIP/medclip/modeling_medclip.py:273-274:
    `cls_sim = torch.mean(logits, 1)`), không phải 1 lựa chọn tự đặt: vì mỗi
    embedding prompt đã L2-normalize (encode_text_safe), "cosine similarity"
    giữa ảnh và 1 prompt chỉ là tích vô hướng — mà tích vô hướng tuyến tính,
    nên mean(ảnh · prompt_n) = ảnh · mean(prompt_n) với mọi n. Tức là "trung
    bình N điểm cosine rồi mới xét" (cách MedCLIP làm) và "trung bình N
    embedding rồi so cosine 1 lần" (cách hàm này làm) cho ĐÚNG 1 con số như
    nhau — KHÔNG được chuẩn hoá lại (renormalize) vector trung bình, vì
    PromptClassifier gốc cũng không làm việc đó (chỉ trung bình các điểm số
    logits thô, không đụng lại vào embedding). Nếu renormalize thêm ở đây sẽ
    tạo ra 1 hệ số phóng đại khác nhau tuỳ độ "phân tán" của các prompt mỗi
    lớp — lệch khỏi cơ chế gốc.

    Vì sao Task 3 dùng "mean" chứ không phải "max" của PromptClassifier gốc:
    "max" chỉ lan truyền gradient tới 1/N prompt "thắng" mỗi bước (không khả
    vi mượt, "người thắng" đổi liên tục khi LoRA đang cập nhật text encoder)
    -- không phù hợp để train. "mean" lan truyền gradient đều tới cả N prompt
    mỗi bước, mượt và ổn định trong suốt quá trình train LoRA, đồng thời vẫn
    bám đúng 1 trong 2 cơ chế PromptClassifier gốc chấp nhận (xem figures/
    MedCLIP-LoRA_offline.tex, Step 3).

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
        mean_embed = embeds.mean(dim=0)  # KHÔNG renormalize -- xem docstring
        class_embeds.append(mean_embed)
    return torch.stack(class_embeds, dim=0)
