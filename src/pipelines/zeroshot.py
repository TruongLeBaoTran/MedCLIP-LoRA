"""
Nhiệm vụ 2 — MedCLIP zero-shot.

Không train. Load MedCLIP pretrain gốc, so cosine similarity giữa embedding
ảnh và embedding prompt từng lớp (đã build sẵn), argmax ra nhãn dự đoán.
"""
import json
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.console import force_utf8_stdout
from common.device import resolve_device
from common.metrics import compute_metrics, plot_confusion_matrix, save_metrics_json
from common.seed import set_seed
from common.transforms import get_medclip_transform
from configs import config as cfg
from data.dataset import BoneXrayDataset
from models.medclip_wrapper import encode_image_safe, load_medclip
from prompts_bone import build_text_features


def run_zeroshot(device: str = None, batch_size: int = 32) -> dict:
    device = resolve_device(device or cfg.DEVICE)
    set_seed(cfg.SEED)

    model = load_medclip(device=device)
    text_features = build_text_features(model, device)  # [NUM_CLASSES, 512], đã normalize

    test_ds = BoneXrayDataset(cfg.TEST_CSV, transform=get_medclip_transform(train=False))
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            image_features = encode_image_safe(model, images, device)  # [B, 512]
            logits = cfg.LOGIT_SCALE * image_features @ text_features.T  # [B, NUM_CLASSES]
            preds = logits.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    metrics = compute_metrics(all_labels, all_preds)
    plot_confusion_matrix(
        all_labels, all_preds,
        save_path=os.path.join(cfg.RESULTS_DIR, "zeroshot_confusion_matrix.png"),
        title="MedCLIP Zero-shot",
    )
    save_metrics_json(metrics, "zeroshot")
    return metrics


if __name__ == "__main__":
    force_utf8_stdout()
    result = run_zeroshot()
    print(json.dumps(result, ensure_ascii=False, indent=2))
