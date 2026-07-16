"""
Đánh giá kết quả — DÙNG CHUNG cho cả 3 nhiệm vụ (baseline / zero-shot / lora)
để đảm bảo so sánh công bằng: cùng công thức, cùng định dạng output.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")  # không cần hiển thị màn hình, chỉ lưu file ảnh
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from configs import config as cfg


def compute_metrics(y_true, y_pred, class_names=None) -> dict:
    """Tính accuracy, macro-F1, và precision/recall/F1 từng lớp.

    Trả về dict phẳng, dễ ghi ra JSON và gộp bảng so sánh sau này.
    """
    class_names = class_names or cfg.CLASS_NAMES
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )

    metrics = {
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "per_class": {
            name: {
                "precision": report[name]["precision"],
                "recall": report[name]["recall"],
                "f1": report[name]["f1-score"],
                "support": report[name]["support"],
            }
            for name in class_names
        },
        "n_samples": len(y_true),
    }
    return metrics


def plot_confusion_matrix(y_true, y_pred, class_names=None, save_path: str = None, title: str = ""):
    """Vẽ confusion matrix (số lượng thật, không chuẩn hoá) và lưu ra file ảnh."""
    class_names = class_names or cfg.CLASS_NAMES
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    ax.set_title(title or "Confusion matrix")

    thresh = cm.max() / 2 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return cm


def save_metrics_json(metrics: dict, task_name: str) -> str:
    """Lưu metrics ra outputs/results/{task_name}_metrics.json, dùng cho compare_results.py."""
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    path = os.path.join(cfg.RESULTS_DIR, f"{task_name}_metrics.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return path
