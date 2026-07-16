"""
Demo dự đoán 1 ảnh X-quang bất kỳ — không phải nhiệm vụ chính của đề tài, chỉ
để minh hoạ trực quan model đang hoạt động thế nào trên 1 ảnh cụ thể.

Ưu tiên dùng MedCLIP + LoRA đã train (outputs/checkpoints/lora_medclip.pt) nếu
có, tự động rơi về MedCLIP zero-shot nếu chưa train checkpoint nào.
"""
import os
import sys

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.console import force_utf8_stdout
from common.device import resolve_device
from common.transforms import get_medclip_transform
from configs import config as cfg
from models.lora_medclip import apply_lora_medclip
from models.medclip_wrapper import encode_image_safe, load_medclip
from prompts_bone import build_text_features


def load_predictor(device: str = None, use_lora: bool = True):
    """Load model + text_features 1 lần — tái dùng cho nhiều lần gọi predict_image()
    thay vì load lại model mỗi lần (rất chậm).

    use_lora=True (mặc định): nạp checkpoint LoRA đã train nếu tồn tại; nếu
    chưa train checkpoint nào thì tự động rơi về zero-shot (có in cảnh báo).
    """
    device = resolve_device(device or cfg.DEVICE)
    model = load_medclip(device=device)

    ckpt_path = os.path.join(cfg.CHECKPOINTS_DIR, "lora_medclip.pt")
    if use_lora and os.path.exists(ckpt_path):
        apply_lora_medclip(model, r=cfg.LORA_R, alpha=cfg.LORA_ALPHA,
                            dropout=cfg.LORA_DROPOUT, target=cfg.LORA_TARGET)
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state, strict=False)
        print(f"[predict] Dùng MedCLIP + LoRA đã train ({ckpt_path})")
    else:
        if use_lora:
            print(f"[predict] Chưa có checkpoint LoRA ({ckpt_path}) — dùng MedCLIP zero-shot. "
                  f"Chạy pipelines.lora_train trước để có checkpoint.")
        else:
            print("[predict] Dùng MedCLIP zero-shot.")

    model.eval()
    text_features = build_text_features(model, device)
    return model, text_features, device


def predict_image(image_path: str, predictor=None) -> dict:
    """Dự đoán 1 ảnh X-quang bất kỳ (đường dẫn file).

    predictor: kết quả trả về từ load_predictor(), truyền vào để tái dùng khi
    predict nhiều ảnh liên tiếp; để None sẽ tự load 1 lần (zero-shot mặc định).

    Trả về {"pred_class": str, "probs": {class_name: xác_suất}}.
    """
    if predictor is None:
        predictor = load_predictor()
    model, text_features, device = predictor

    image = Image.open(image_path)
    pixel_values = get_medclip_transform(train=False)(image).unsqueeze(0)

    with torch.no_grad():
        image_features = encode_image_safe(model, pixel_values, device)
        logits = cfg.LOGIT_SCALE * image_features @ text_features.T
        probs = torch.softmax(logits, dim=1)[0]

    pred_idx = int(probs.argmax().item())
    return {
        "pred_class": cfg.CLASS_NAMES[pred_idx],
        "probs": {cfg.CLASS_NAMES[i]: round(probs[i].item(), 4) for i in range(cfg.NUM_CLASSES)},
    }


if __name__ == "__main__":
    force_utf8_stdout()
    import argparse

    parser = argparse.ArgumentParser(description="Dự đoán 1 ảnh X-quang xương bất kỳ.")
    parser.add_argument("image_path", help="Đường dẫn tới file ảnh")
    parser.add_argument("--no-lora", action="store_true", help="Ép dùng zero-shot, bỏ qua checkpoint LoRA")
    args = parser.parse_args()

    predictor = load_predictor(use_lora=not args.no_lora)
    result = predict_image(args.image_path, predictor)
    print(f"\nDự đoán: {result['pred_class']}")
    print(f"Xác suất từng lớp: {result['probs']}")
