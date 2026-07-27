"""
Nhiệm vụ 2 — MedCLIP zero-shot.

Không train. Load MedCLIP pretrain gốc, so cosine similarity giữa embedding
ảnh và embedding TỪNG prompt của mỗi lớp riêng lẻ, rồi gộp N điểm/lớp bằng
max (mặc định) hoặc mean (ensemble) — đúng cơ chế PromptClassifier gốc của
MedCLIP (medclip/modeling_medclip.py:247-285), không trung bình embedding
trước khi so cosine (khác Task 3, xem prompts_bone.py::build_text_features_per_prompt
để biết lý do 2 task dùng 2 cách gộp khác nhau).
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
from prompts_bone import build_text_features_per_prompt


def run_zeroshot(device: str = None, batch_size: int = 32, aggregation: str = None) -> dict:
    device = resolve_device(device or cfg.DEVICE)
    set_seed(cfg.SEED)

    aggregation = cfg.ZEROSHOT_PROMPT_AGGREGATION if aggregation is None else aggregation
    if aggregation not in ("max", "mean"):
        raise ValueError(f"aggregation phải là 'max' hoặc 'mean', nhận '{aggregation}'")

    model = load_medclip(device=device)
    text_features_per_class = build_text_features_per_prompt(model, device)  # list[K], mỗi phần tử [N_k, 512]
    n_prompts = [t.shape[0] for t in text_features_per_class]
    print(f"[zeroshot] prompt aggregation = '{aggregation}' "
          f"(số prompt/lớp: {dict(zip(cfg.CLASS_NAMES, n_prompts))}) — khớp PromptClassifier gốc MedCLIP.")

    test_ds = BoneXrayDataset(cfg.TEST_CSV, transform=get_medclip_transform(train=False))
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            image_features = encode_image_safe(model, images, device)  # [B, 512]
            class_scores = []
            for text_feats_k in text_features_per_class:
                sims = image_features @ text_feats_k.T  # [B, N_k] cosine similarity từng prompt riêng lẻ
                score_k = sims.max(dim=1).values if aggregation == "max" else sims.mean(dim=1)
                class_scores.append(score_k)
            logits = cfg.LOGIT_SCALE * torch.stack(class_scores, dim=1)  # [B, NUM_CLASSES]
            preds = logits.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    metrics = compute_metrics(all_labels, all_preds)
    metrics["prompt_aggregation"] = aggregation  # traceability trong zeroshot_metrics.json
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
