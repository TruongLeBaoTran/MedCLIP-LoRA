"""
Nhiệm vụ 1 — Baseline CNN/ViT.

Fine-tune từ ImageNet-pretrained trên toàn bộ train split BTXRD (3 lớp).
Độc lập hoàn toàn với MedCLIP — dùng để so sánh với zero-shot/LoRA.
"""
import copy
import json
import os
import sys

import pandas as pd
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.console import force_utf8_stdout
from common.device import resolve_device
from common.metrics import compute_metrics, plot_confusion_matrix, save_metrics_json
from common.seed import set_seed
from common.transforms import get_baseline_transform
from configs import config as cfg
from data.dataset import BoneXrayDataset


def build_baseline_model(backbone: str, num_classes: int, device: str) -> nn.Module:
    if backbone == "resnet50":
        model = torchvision.models.resnet50(weights="IMAGENET1K_V2")
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif backbone == "vit_b16":
        import timm
        model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=num_classes)
    else:
        raise ValueError(f"backbone không hỗ trợ: {backbone}")
    return model.to(device)


def _compute_class_weights(train_csv: str, num_classes: int, device: str) -> torch.Tensor:
    """Trọng số nghịch đảo tần suất lớp — giảm ảnh hưởng mất cân bằng lớp
    (Malignant chỉ ~9% dữ liệu) lên hàm loss."""
    train_df = pd.read_csv(train_csv)
    counts = train_df["label"].value_counts().reindex(range(num_classes), fill_value=0)
    weights = 1.0 / counts.clip(lower=1)
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights.values, dtype=torch.float32, device=device)


def _predict(model: nn.Module, loader: DataLoader, device: str):
    model.eval()
    preds, labels_all = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            preds.extend(logits.argmax(dim=1).cpu().tolist())
            labels_all.extend(labels.tolist())
    return preds, labels_all


def run_baseline(device: str = None, epochs: int = None, batch_size: int = None, lr: float = None) -> dict:
    device = resolve_device(device or cfg.DEVICE)
    set_seed(cfg.SEED)
    epochs = epochs or cfg.BASELINE_EPOCHS
    batch_size = batch_size or cfg.BASELINE_BATCH_SIZE
    lr = lr or cfg.BASELINE_LR

    train_ds = BoneXrayDataset(cfg.TRAIN_CSV, transform=get_baseline_transform(train=True))
    val_ds = BoneXrayDataset(cfg.VAL_CSV, transform=get_baseline_transform(train=False))
    test_ds = BoneXrayDataset(cfg.TEST_CSV, transform=get_baseline_transform(train=False))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = build_baseline_model(cfg.BASELINE_BACKBONE, cfg.NUM_CLASSES, device)
    class_weights = _compute_class_weights(cfg.TRAIN_CSV, cfg.NUM_CLASSES, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_f1, best_state = -1.0, None
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
        avg_loss = total_loss / len(train_ds)

        val_preds, val_labels = _predict(model, val_loader, device)
        val_metrics = compute_metrics(val_labels, val_preds)
        print(f"[baseline] epoch {epoch + 1}/{epochs} loss={avg_loss:.4f} "
              f"val_acc={val_metrics['accuracy']:.4f} val_macro_f1={val_metrics['macro_f1']:.4f}")

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    os.makedirs(cfg.CHECKPOINTS_DIR, exist_ok=True)
    torch.save(best_state, os.path.join(cfg.CHECKPOINTS_DIR, "baseline_best.pt"))

    test_preds, test_labels = _predict(model, test_loader, device)
    metrics = compute_metrics(test_labels, test_preds)
    plot_confusion_matrix(
        test_labels, test_preds,
        save_path=os.path.join(cfg.RESULTS_DIR, "baseline_confusion_matrix.png"),
        title=f"Baseline {cfg.BASELINE_BACKBONE}",
    )
    save_metrics_json(metrics, "baseline")
    return metrics


if __name__ == "__main__":
    force_utf8_stdout()
    result = run_baseline()
    print(json.dumps(result, ensure_ascii=False, indent=2))
