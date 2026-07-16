"""
CLIP gốc zero-shot — mốc 0-shot riêng cho đường cong nhiệm vụ 4 (CLIP + LoRA
đối chứng). Không dùng chung mốc 0-shot với MedCLIP zero-shot vì đó là 2 model
khác nhau — nếu dùng lẫn sẽ hiểu sai đường cong CLIP đang bắt đầu từ đâu.
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
from common.transforms import get_clip_transform
from configs import config as cfg
from data.dataset import BoneXrayDataset
from models.clip_wrapper import build_text_features_clip, encode_image_clip, load_clip


def run_clip_zeroshot(device: str = None, batch_size: int = 32, backbone: str = None) -> dict:
    device = resolve_device(device or cfg.DEVICE)
    set_seed(cfg.SEED)
    backbone = backbone or cfg.CLIP_BACKBONE

    model, _clip = load_clip(device=device, backbone=backbone)
    text_features = build_text_features_clip(model, device)

    test_ds = BoneXrayDataset(cfg.TEST_CSV, transform=get_clip_transform(train=False))
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            image_features = encode_image_clip(model, images, device)
            logits = cfg.LOGIT_SCALE * image_features @ text_features.T
            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.tolist())

    metrics = compute_metrics(all_labels, all_preds)
    plot_confusion_matrix(
        all_labels, all_preds,
        save_path=os.path.join(cfg.RESULTS_DIR, "clip_zeroshot_confusion_matrix.png"),
        title=f"CLIP gốc ({backbone}) Zero-shot",
    )
    save_metrics_json(metrics, "clip_zeroshot")
    return metrics


if __name__ == "__main__":
    force_utf8_stdout()
    result = run_clip_zeroshot()
    print(json.dumps(result, ensure_ascii=False, indent=2))
