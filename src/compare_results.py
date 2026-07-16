"""
Gộp kết quả 3 nhiệm vụ (baseline / zero-shot / lora_fewshot) thành 1 bảng
so sánh — đây là kết quả cốt lõi cuối cùng của cả đề tài.

Chạy sau khi đã chạy xong pipelines/baseline.py, pipelines/zeroshot.py,
pipelines/lora_train.py (mỗi file tự lưu outputs/results/{task}_metrics.json).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.console import force_utf8_stdout
from configs import config as cfg

_HEADLINE_SHOTS = max(cfg.FEWSHOT_SWEEP_SHOTS)  # mức shot lớn nhất trong sweep, đại diện cho LoRA trong bảng

TASK_LABELS = {
    "baseline": f"Baseline {cfg.BASELINE_BACKBONE}",
    "zeroshot": "MedCLIP zero-shot",
    f"lora_fewshot_{_HEADLINE_SHOTS}shot": f"MedCLIP + LoRA ({_HEADLINE_SHOTS}-shot)",
    # Nhiệm vụ 4 (đối chứng) — chỉ xuất hiện trong bảng nếu đã chạy
    # pipelines.clip_lora_train.run_clip_lora_shots_sweep()
    f"clip_lora_fewshot_{_HEADLINE_SHOTS}shot": f"CLIP gốc ({cfg.CLIP_BACKBONE}) + LoRA ({_HEADLINE_SHOTS}-shot)",
}


def load_task_metrics(task_name: str):
    path = os.path.join(cfg.RESULTS_DIR, f"{task_name}_metrics.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_comparison_table() -> list:
    rows = []
    for task_key, label in TASK_LABELS.items():
        metrics = load_task_metrics(task_key)
        if metrics is None:
            print(f"[compare_results] Chưa có kết quả cho '{task_key}' "
                  f"(outputs/results/{task_key}_metrics.json không tồn tại), bỏ qua.")
            continue
        row = {
            "Phương pháp": label,
            "Accuracy": round(metrics["accuracy"], 4),
            "Macro-F1": round(metrics["macro_f1"], 4),
        }
        for cls_name in cfg.CLASS_NAMES:
            cls_metrics = metrics["per_class"][cls_name]
            row[f"{cls_name} Precision"] = round(cls_metrics["precision"], 4)
            row[f"{cls_name} Recall"] = round(cls_metrics["recall"], 4)
            row[f"{cls_name} F1"] = round(cls_metrics["f1"], 4)
        rows.append(row)
    return rows


def print_markdown_table(rows: list) -> None:
    if not rows:
        print("Chưa có kết quả nào để so sánh.")
        return
    headers = list(rows[0].keys())
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        print("| " + " | ".join(str(row[h]) for h in headers) + " |")


def save_comparison_csv(rows: list, path: str = None) -> None:
    import pandas as pd

    if not rows:
        return
    path = path or os.path.join(cfg.RESULTS_DIR, "comparison_table.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"\nĐã lưu bảng so sánh vào {path}")


def load_shots_curve(shots_list: list = None, prefix: str = "lora_fewshot", zeroshot_task: str = "zeroshot") -> dict:
    """Đọc kết quả LoRA few-shot ở nhiều mức shot + zero-shot làm mốc 0-shot.
    Trả về dict {n_shots: metrics}, sắp theo thứ tự tăng dần số shot.

    prefix: "lora_fewshot" (mặc định, MedCLIP) hoặc "clip_lora_fewshot" (CLIP
    gốc, nhiệm vụ 4) — đổi để đọc kết quả nhánh CLIP đối chứng.
    """
    shots_list = shots_list or cfg.FEWSHOT_SWEEP_SHOTS
    curve = {}

    zeroshot_metrics = load_task_metrics(zeroshot_task)
    if zeroshot_metrics is not None:
        curve[0] = zeroshot_metrics

    for n_shots in shots_list:
        metrics = load_task_metrics(f"{prefix}_{n_shots}shot")
        if metrics is not None:
            curve[n_shots] = metrics
        else:
            print(f"[compare_results] Chưa có kết quả {prefix}_{n_shots}shot.")
    return dict(sorted(curve.items()))


def print_shots_curve_table(curve: dict) -> None:
    """In bảng accuracy/macro-F1 theo số shot. Nếu kết quả đến từ sweep nhiều
    seed (có khoá *_std), in dạng mean±std thay vì 1 con số đơn."""
    if not curve:
        print("Chưa có kết quả few-shot nào để vẽ đường cong.")
        return

    def fmt(s, key):
        val = curve[s][key]
        std = curve[s].get(f"{key}_std")
        return f"{val:.4f}±{std:.4f}" if std is not None else f"{val:.4f}"

    shots = list(curve.keys())
    headers = ["Số shot"] + [str(s) for s in shots]
    acc_row = ["Accuracy"] + [fmt(s, "accuracy") for s in shots]
    f1_row = ["Macro-F1"] + [fmt(s, "macro_f1") for s in shots]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    print("| " + " | ".join(acc_row) + " |")
    print("| " + " | ".join(f1_row) + " |")


def plot_shots_curve(curve: dict, baseline_metrics: dict = None, save_path: str = None) -> None:
    """Vẽ accuracy/macro-F1 theo số shot (0 = zero-shot), có đường ngang tham
    chiếu Baseline full-data (nếu có) — đúng dạng biểu đồ chuẩn few-shot VLM
    (xem CLIP-LoRA/few_shot.png). Nếu kết quả có độ lệch chuẩn (*_std, từ chạy
    nhiều seed) thì vẽ kèm error bar."""
    if not curve:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shots = list(curve.keys())
    acc = [curve[s]["accuracy"] for s in shots]
    f1 = [curve[s]["macro_f1"] for s in shots]
    acc_err = [curve[s].get("accuracy_std", 0.0) for s in shots]
    f1_err = [curve[s].get("macro_f1_std", 0.0) for s in shots]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.errorbar(shots, acc, yerr=acc_err, marker="o", capsize=4, label="Accuracy")
    ax.errorbar(shots, f1, yerr=f1_err, marker="s", capsize=4, label="Macro-F1")
    if baseline_metrics is not None:
        ax.axhline(baseline_metrics["accuracy"], color="gray", linestyle="--",
                    label=f"Baseline accuracy (full-data, {cfg.BASELINE_BACKBONE})")
    ax.set_xlabel("Số shot / lớp (0 = zero-shot)")
    ax.set_ylabel("Score")
    ax.set_xticks(shots)
    ax.set_ylim(0, 1)
    ax.set_title("MedCLIP + LoRA — accuracy/macro-F1 theo số shot")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_backbone_comparison_curve(
    medclip_curve: dict, clip_curve: dict, baseline_metrics: dict = None, save_path: str = None
) -> None:
    """So sánh trực tiếp MedCLIP + LoRA vs CLIP gốc + LoRA trên cùng 1 biểu đồ
    — cùng dataset/split/few-shot/siêu tham số LoRA, chỉ khác backbone. Đây là
    biểu đồ trả lời câu hỏi trung tâm của nhiệm vụ 4: domain pretrain y khoa
    (MedCLIP, pretrain trên X-quang NGỰC) có thật sự giúp ích trên ảnh X-quang
    XƯƠNG hay không, so với CLIP gốc (pretrain ảnh tổng quát)."""
    if not medclip_curve and not clip_curve:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 5))
    if medclip_curve:
        shots = list(medclip_curve.keys())
        acc = [medclip_curve[s]["accuracy"] for s in shots]
        err = [medclip_curve[s].get("accuracy_std", 0.0) for s in shots]
        ax.errorbar(shots, acc, yerr=err, marker="o", capsize=4, label="MedCLIP + LoRA")
    if clip_curve:
        shots = list(clip_curve.keys())
        acc = [clip_curve[s]["accuracy"] for s in shots]
        err = [clip_curve[s].get("accuracy_std", 0.0) for s in shots]
        ax.errorbar(shots, acc, yerr=err, marker="^", capsize=4, label=f"CLIP gốc ({cfg.CLIP_BACKBONE}) + LoRA")
    if baseline_metrics is not None:
        ax.axhline(baseline_metrics["accuracy"], color="gray", linestyle="--",
                    label=f"Baseline accuracy (full-data, {cfg.BASELINE_BACKBONE})")

    all_shots = sorted(set(list(medclip_curve.keys()) + list(clip_curve.keys())))
    ax.set_xlabel("Số shot / lớp (0 = zero-shot)")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(all_shots)
    ax.set_ylim(0, 1)
    ax.set_title("MedCLIP + LoRA vs CLIP gốc + LoRA (cùng dataset/split/few-shot)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    force_utf8_stdout()
    table_rows = build_comparison_table()
    print_markdown_table(table_rows)
    save_comparison_csv(table_rows)

    print()
    shots_curve = load_shots_curve(prefix="lora_fewshot")
    print_shots_curve_table(shots_curve)
    plot_shots_curve(
        shots_curve,
        baseline_metrics=load_task_metrics("baseline"),
        save_path=os.path.join(cfg.RESULTS_DIR, "lora_shots_curve.png"),
    )
    if shots_curve:
        print(f"\nĐã lưu biểu đồ đường cong vào {os.path.join(cfg.RESULTS_DIR, 'lora_shots_curve.png')}")

    print()
    clip_shots_curve = load_shots_curve(prefix="clip_lora_fewshot", zeroshot_task="clip_zeroshot")
    if clip_shots_curve:
        print("Đường cong CLIP gốc + LoRA:")
        print_shots_curve_table(clip_shots_curve)
        plot_backbone_comparison_curve(
            shots_curve, clip_shots_curve,
            baseline_metrics=load_task_metrics("baseline"),
            save_path=os.path.join(cfg.RESULTS_DIR, "backbone_comparison_curve.png"),
        )
        print(f"\nĐã lưu biểu đồ so sánh backbone vào "
              f"{os.path.join(cfg.RESULTS_DIR, 'backbone_comparison_curve.png')}")
    else:
        print("[compare_results] Chưa chạy nhiệm vụ 4 (CLIP + LoRA đối chứng), bỏ qua biểu đồ so sánh backbone.")
