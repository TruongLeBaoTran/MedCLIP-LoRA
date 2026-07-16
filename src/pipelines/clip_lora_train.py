"""
Nhiệm vụ 4 (đối chứng) — CLIP gốc + LoRA, cùng dataset/split/few-shot/siêu
tham số LoRA với nhiệm vụ 3 (MedCLIP + LoRA), chỉ đổi backbone. Trả lời câu
hỏi: MedCLIP (pretrain trên X-quang ngực) có thật sự tốt hơn CLIP gốc
(pretrain ảnh tổng quát) trên ảnh xương, hay domain y khoa không giúp ích gì
nếu domain con (xương) khác domain pretrain (ngực)?

Cấu trúc lặp lại gần như y hệt pipelines/lora_train.py — chỉ khác phần load
model (clip.load thay vì load_medclip) và cách gắn LoRA (apply_lora_clip
dùng PlainMultiheadAttentionLoRA, thay vì apply_lora_medclip dùng LinearLoRA
trực tiếp lên Q/K/V tách rời).

Tham số use_val_checkpoint giống hệt pipelines/lora_train.py — xem chú thích
ở đó và README mục 6.7 (đánh đổi giữa chọn checkpoint theo val vs bám đúng
protocol CLIP-LoRA gốc theo tranh luận "true few-shot learning").
"""
import copy
import json
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.console import force_utf8_stdout
from common.device import resolve_device
from common.metrics import compute_metrics, plot_confusion_matrix, save_metrics_json
from common.seed import set_seed
from common.transforms import get_clip_transform
from configs import config as cfg
from data.dataset import BoneXrayDataset, sample_few_shot
from lora.utils import get_lora_parameters, lora_state_dict, mark_only_lora_as_trainable
from models.clip_wrapper import build_text_features_clip, encode_image_clip, load_clip
from models.lora_clip import apply_lora_clip


def _evaluate(model, text_features, device, csv_path, batch_size=32):
    model.eval()
    ds = BoneXrayDataset(csv_path, transform=get_clip_transform(train=False))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            image_features = encode_image_clip(model, images, device)
            logits = cfg.LOGIT_SCALE * image_features @ text_features.T
            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.tolist())
    return all_preds, all_labels


def run_clip_lora_train(
    device: str = None,
    n_shots: int = None,
    r: int = None,
    alpha: int = None,
    dropout: float = None,
    target: str = None,
    lr: float = None,
    n_iters: int = None,
    batch_size: int = None,
    val_every: int = 20,
    task_name: str = None,
    seed: int = None,
    backbone: str = None,
    use_val_checkpoint: bool = None,
) -> dict:
    device = resolve_device(device or cfg.DEVICE)
    seed = cfg.SEED if seed is None else seed
    set_seed(seed)
    n_shots = n_shots or cfg.FEWSHOT_N_SHOTS
    task_name = task_name or f"clip_lora_fewshot_{n_shots}shot"
    # Dùng CHUNG r/alpha/dropout/n_iters/lr/batch_size với nhiệm vụ 3 (MedCLIP)
    # theo mặc định — để so sánh có kiểm soát, chỉ đổi backbone.
    r = r or cfg.LORA_R
    alpha = alpha or cfg.LORA_ALPHA
    dropout = cfg.LORA_DROPOUT if dropout is None else dropout
    target = target or cfg.LORA_TARGET
    lr = lr or cfg.LORA_LR
    n_iters = n_iters or cfg.LORA_N_ITERS
    # Khớp CLIP-LoRA gốc (lora.py:73) — xem chú thích trong lora_train.py
    total_iters = n_iters * n_shots
    batch_size = batch_size or cfg.LORA_BATCH_SIZE
    backbone = backbone or cfg.CLIP_BACKBONE
    use_val_checkpoint = (
        cfg.LORA_USE_VAL_CHECKPOINT_SELECTION if use_val_checkpoint is None else use_val_checkpoint
    )

    model, _clip = load_clip(device=device, backbone=backbone)
    inserted = apply_lora_clip(model, r=r, alpha=alpha, dropout=dropout, target=target,
                                enable_lora=cfg.CLIP_LORA_ENABLE)
    print(f"[clip_lora_train] đã gắn LoRA vào {len(inserted)} layer "
          f"(backbone={backbone}, target={target}, r={r}, alpha={alpha})")
    mark_only_lora_as_trainable(model)
    n_trainable = sum(p.numel() for p in get_lora_parameters(model))
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[clip_lora_train] tham số trainable: {n_trainable}/{n_total} ({100 * n_trainable / n_total:.3f}%)")

    fewshot_df = sample_few_shot(cfg.TRAIN_CSV, n_shots=n_shots, seed=seed)
    print(f"[clip_lora_train] few-shot subset: {len(fewshot_df)} ảnh "
          f"({fewshot_df['label'].value_counts().sort_index().to_dict()})")
    train_ds = BoneXrayDataset(fewshot_df, transform=get_clip_transform(train=True))
    train_loader = DataLoader(train_ds, batch_size=min(batch_size, len(train_ds)), shuffle=True)

    optimizer = torch.optim.AdamW(get_lora_parameters(model), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iters, eta_min=1e-6)

    fixed_text_features = None
    if target == "vision":
        with torch.no_grad():
            fixed_text_features = build_text_features_clip(model, device)

    best_val_f1 = -1.0
    best_lora_state = None

    model.train()
    count_iters = 0
    while count_iters < total_iters:
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            text_features = fixed_text_features if target == "vision" else build_text_features_clip(
                model, device, requires_grad=True
            )
            image_features = encode_image_clip(model, images, device)
            logits = cfg.LOGIT_SCALE * image_features @ text_features.T
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            count_iters += 1
            is_last = count_iters == total_iters
            if count_iters % val_every == 0 or is_last:
                batch_acc = (logits.argmax(dim=1) == labels).float().mean().item()

                if use_val_checkpoint:
                    with torch.no_grad():
                        val_text_features = build_text_features_clip(model, device)
                    val_preds, val_labels = _evaluate(model, val_text_features, device, cfg.VAL_CSV)
                    val_metrics = compute_metrics(val_labels, val_preds)
                    model.train()

                    print(f"[clip_lora_train] iter {count_iters}/{total_iters} loss={loss.item():.4f} "
                          f"batch_acc={batch_acc:.4f} val_acc={val_metrics['accuracy']:.4f} "
                          f"val_macro_f1={val_metrics['macro_f1']:.4f}")

                    if val_metrics["macro_f1"] > best_val_f1:
                        best_val_f1 = val_metrics["macro_f1"]
                        best_lora_state = copy.deepcopy(lora_state_dict(model))
                else:
                    # Không đụng đến val trong lúc train — đúng protocol CLIP-LoRA
                    # gốc (lora.py: VALIDATION=False). Chỉ in tiến độ để theo dõi.
                    print(f"[clip_lora_train] iter {count_iters}/{total_iters} loss={loss.item():.4f} "
                          f"batch_acc={batch_acc:.4f} (không dùng val)")

            if count_iters >= total_iters:
                break

    if use_val_checkpoint and best_lora_state is not None:
        model.load_state_dict(best_lora_state, strict=False)
        print(f"[clip_lora_train] dùng checkpoint LoRA tốt nhất theo val (val_macro_f1={best_val_f1:.4f})")
    elif not use_val_checkpoint:
        print("[clip_lora_train] dùng checkpoint LoRA ở bước train cuối cùng (không chọn qua val, "
              "đúng protocol CLIP-LoRA gốc)")

    os.makedirs(cfg.CHECKPOINTS_DIR, exist_ok=True)
    ckpt_path = os.path.join(cfg.CHECKPOINTS_DIR, "clip_lora.pt")
    torch.save(lora_state_dict(model), ckpt_path)
    print(f"[clip_lora_train] đã lưu LoRA checkpoint (nhẹ) vào {ckpt_path}")

    with torch.no_grad():
        final_text_features = build_text_features_clip(model, device)
    test_preds, test_labels = _evaluate(model, final_text_features, device, cfg.TEST_CSV)

    metrics = compute_metrics(test_labels, test_preds)
    metrics["checkpoint_selection"] = "val_best" if use_val_checkpoint else "final_iter"
    plot_confusion_matrix(
        test_labels, test_preds,
        save_path=os.path.join(cfg.RESULTS_DIR, f"{task_name}_confusion_matrix.png"),
        title=f"CLIP ({backbone}) + LoRA ({n_shots}-shot)",
    )
    save_metrics_json(metrics, task_name)
    return metrics


def _aggregate_seed_runs(run_metrics: list) -> dict:
    """Giống hệt logic gộp trong pipelines/lora_train.py — xem chú thích ở đó."""
    import numpy as np

    accs = [m["accuracy"] for m in run_metrics]
    f1s = [m["macro_f1"] for m in run_metrics]
    agg = {
        "accuracy": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "macro_f1": float(np.mean(f1s)),
        "macro_f1_std": float(np.std(f1s)),
        "macro_precision": float(np.mean([m["macro_precision"] for m in run_metrics])),
        "macro_recall": float(np.mean([m["macro_recall"] for m in run_metrics])),
        "per_class": {},
        "n_seeds": len(run_metrics),
        "n_samples": run_metrics[0]["n_samples"],
        "checkpoint_selection": run_metrics[0].get("checkpoint_selection"),
    }
    for cls_name in cfg.CLASS_NAMES:
        agg["per_class"][cls_name] = {
            metric: float(np.mean([m["per_class"][cls_name][metric] for m in run_metrics]))
            for metric in ("precision", "recall", "f1")
        }
        agg["per_class"][cls_name]["support"] = run_metrics[0]["per_class"][cls_name]["support"]
    return agg


def run_clip_lora_shots_sweep(shots_list: list = None, seeds: list = None, **kwargs) -> dict:
    """Tương đương run_lora_shots_sweep() của MedCLIP, cho nhánh CLIP gốc.
    Lưu:
    - Từng lần chạy riêng theo seed: clip_lora_fewshot_{n}shot_seed{s}_metrics.json
    - Gộp theo mức shot (mean±std): clip_lora_fewshot_{n}shot_metrics.json
    """
    shots_list = shots_list or cfg.FEWSHOT_SWEEP_SHOTS
    seeds = seeds or cfg.FEWSHOT_SWEEP_SEEDS
    summary = {}
    for n_shots in shots_list:
        run_metrics = []
        for seed in seeds:
            print(f"\n{'=' * 60}\n[clip_lora_sweep] {n_shots}-shot, seed={seed}\n{'=' * 60}")
            metrics = run_clip_lora_train(
                n_shots=n_shots, seed=seed,
                task_name=f"clip_lora_fewshot_{n_shots}shot_seed{seed}",
                **kwargs,
            )
            run_metrics.append(metrics)

        agg = _aggregate_seed_runs(run_metrics)
        save_metrics_json(agg, f"clip_lora_fewshot_{n_shots}shot")
        summary[n_shots] = agg
        print(f"[clip_lora_sweep] {n_shots}-shot (qua {len(seeds)} seed): "
              f"accuracy={agg['accuracy']:.4f}±{agg['accuracy_std']:.4f}, "
              f"macro_f1={agg['macro_f1']:.4f}±{agg['macro_f1_std']:.4f}")
    return summary


if __name__ == "__main__":
    force_utf8_stdout()
    result = run_clip_lora_train()
    print(json.dumps(result, ensure_ascii=False, indent=2))
