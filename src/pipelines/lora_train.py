"""
Nhiệm vụ 3 — MedCLIP + LoRA few-shot.

Đóng băng toàn bộ MedCLIP gốc, chỉ train các ma trận LoRA (rank thấp) chèn
vào Q/K/V của attention. Dùng vài ảnh/lớp (few-shot) lấy từ train split.
Loss = cross-entropy trên logits giữa ảnh và text prompt từng lớp — không
thêm linear head, giữ đúng "bản chất VLM" (xem hoi-dap.md). text_features
(prompts_bone.py::build_text_features) là trung bình N điểm cosine/lớp
(KHÔNG renormalize sau khi trung bình embedding) — đúng cơ chế
ensemble=True của PromptClassifier gốc MedCLIP, không còn là cosine
similarity thuần với 1 vector đã unit-norm nữa (xem docstring của
build_text_features() để biết vì sao 2 cách tương đương về mặt toán).

Có 2 cách chạy:
- run_lora_train(): 1 lần chạy, 1 mức shot, 1 seed cố định — dùng để debug nhanh.
- run_lora_shots_sweep(): chạy đầy đủ nhiều mức shot × nhiều seed, gộp mean±std
  — đây là cách nên dùng để lấy kết quả báo cáo cuối cùng (chuẩn CoOp/CLIP-LoRA).

Tham số use_val_checkpoint (mặc định lấy từ config.LORA_USE_VAL_CHECKPOINT_SELECTION):
chọn checkpoint theo val (True) hay lấy checkpoint bước train cuối cùng, đúng
protocol CLIP-LoRA gốc (False) — xem README mục 6.7 để biết đánh đổi giữa 2 lựa
chọn theo tranh luận "true few-shot learning" (Perez et al., NeurIPS 2021).
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
from common.transforms import get_medclip_transform
from configs import config as cfg
from data.dataset import BoneXrayDataset, sample_few_shot
from lora.utils import get_lora_parameters, lora_state_dict, mark_only_lora_as_trainable
from models.lora_medclip import apply_lora_medclip
from models.medclip_wrapper import encode_image_safe, load_medclip
from prompts_bone import build_text_features


def _evaluate(model, text_features, device, csv_path, batch_size=32):
    model.eval()
    ds = BoneXrayDataset(csv_path, transform=get_medclip_transform(train=False))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            image_features = encode_image_safe(model, images, device)
            logits = cfg.LOGIT_SCALE * image_features @ text_features.T
            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.tolist())
    return all_preds, all_labels


def run_lora_train(
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
    use_val_checkpoint: bool = None,
) -> dict:
    device = resolve_device(device or cfg.DEVICE)
    # seed quyết định: (1) khởi tạo random của torch/numpy, (2) ẢNH NÀO cụ thể
    # được bốc vào few-shot subset (xem sample_few_shot bên dưới). Đổi seed khi
    # muốn kiểm tra kết quả có ổn định hay chỉ do may rủi bốc trúng ảnh dễ/khó
    # — quan trọng nhất ở mức shot thấp (1-shot, 2-shot).
    seed = cfg.SEED if seed is None else seed
    set_seed(seed)
    n_shots = n_shots or cfg.FEWSHOT_N_SHOTS
    # Tên để lưu kết quả — mặc định gắn kèm số shot (lora_fewshot_4shot,
    # lora_fewshot_8shot...) để chạy nhiều mức shot không ghi đè lẫn nhau,
    # phục vụ vẽ đường cong accuracy theo số shot (chuẩn báo cáo few-shot VLM).
    task_name = task_name or f"lora_fewshot_{n_shots}shot"
    r = r or cfg.LORA_R
    alpha = alpha or cfg.LORA_ALPHA
    dropout = cfg.LORA_DROPOUT if dropout is None else dropout
    target = target or cfg.LORA_TARGET
    lr = lr or cfg.LORA_LR
    n_iters = n_iters or cfg.LORA_N_ITERS
    # Khớp CLIP-LoRA gốc (lora.py:73): tổng số bước train tỉ lệ thuận với số
    # ảnh few-shot, giữ số epoch (số lần duyệt hết tập few-shot) xấp xỉ ổn
    # định giữa các mức shot, thay vì để 1-shot và 16-shot train cùng số bước.
    total_iters = n_iters * n_shots
    batch_size = batch_size or cfg.LORA_BATCH_SIZE
    # True: chọn checkpoint theo val (ổn định hơn, nhưng dùng thêm nhãn ngoài
    # K-shot). False: lấy checkpoint bước cuối, đúng protocol CLIP-LoRA gốc
    # (không dùng val cho việc gì cả) -> "true few-shot" chặt hơn. Xem
    # config.py::LORA_USE_VAL_CHECKPOINT_SELECTION và README mục 6.7.
    use_val_checkpoint = (
        cfg.LORA_USE_VAL_CHECKPOINT_SELECTION if use_val_checkpoint is None else use_val_checkpoint
    )

    model = load_medclip(device=device)
    inserted = apply_lora_medclip(model, r=r, alpha=alpha, dropout=dropout, target=target)
    print(f"[lora_train] đã gắn LoRA vào {len(inserted)} layer (target={target}, r={r}, alpha={alpha})")
    mark_only_lora_as_trainable(model)
    n_trainable = sum(p.numel() for p in get_lora_parameters(model))
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[lora_train] tham số trainable: {n_trainable}/{n_total} ({100 * n_trainable / n_total:.3f}%)")

    # Few-shot subset lấy từ train split (không phải toàn bộ train — đúng tinh thần few-shot)
    fewshot_df = sample_few_shot(cfg.TRAIN_CSV, n_shots=n_shots, seed=seed)
    print(f"[lora_train] few-shot subset: {len(fewshot_df)} ảnh "
          f"({fewshot_df['label'].value_counts().sort_index().to_dict()})")
    train_ds = BoneXrayDataset(fewshot_df, transform=get_medclip_transform(train=True))
    train_loader = DataLoader(train_ds, batch_size=min(batch_size, len(train_ds)), shuffle=True)

    optimizer = torch.optim.AdamW(get_lora_parameters(model), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iters, eta_min=1e-6)

    # Nếu chỉ LoRA nhánh vision, text encoder đứng yên -> tính text_features 1 lần
    # là đủ. Nếu có LoRA nhánh text (target="text"/"both"), text_features đổi theo
    # từng bước train nên phải tính lại mỗi iteration, giữ đồ thị đạo hàm.
    fixed_text_features = None
    if target == "vision":
        with torch.no_grad():
            fixed_text_features = build_text_features(model, device)

    # Theo dõi val để chọn checkpoint LoRA tốt nhất (giống baseline.py) — tránh
    # lấy nguyên trạng thái ở bước cuối cùng, có thể đã overfit lên vài ảnh few-shot.
    best_val_f1 = -1.0
    best_lora_state = None

    model.train()
    count_iters = 0
    while count_iters < total_iters:
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            text_features = fixed_text_features if target == "vision" else build_text_features(
                model, device, requires_grad=True
            )
            image_features = encode_image_safe(model, images, device)
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
                        val_text_features = build_text_features(model, device)
                    val_preds, val_labels = _evaluate(model, val_text_features, device, cfg.VAL_CSV)
                    val_metrics = compute_metrics(val_labels, val_preds)
                    model.train()  # _evaluate() chuyển model sang eval(), quay lại train() để train tiếp

                    print(f"[lora_train] iter {count_iters}/{total_iters} loss={loss.item():.4f} "
                          f"batch_acc={batch_acc:.4f} val_acc={val_metrics['accuracy']:.4f} "
                          f"val_macro_f1={val_metrics['macro_f1']:.4f}")

                    if val_metrics["macro_f1"] > best_val_f1:
                        best_val_f1 = val_metrics["macro_f1"]
                        best_lora_state = copy.deepcopy(lora_state_dict(model))
                else:
                    # Không đụng đến val trong lúc train — đúng protocol CLIP-LoRA
                    # gốc (lora.py: VALIDATION=False). Chỉ in tiến độ để theo dõi.
                    print(f"[lora_train] iter {count_iters}/{total_iters} loss={loss.item():.4f} "
                          f"batch_acc={batch_acc:.4f} (không dùng val)")

            if count_iters >= total_iters:
                break

    if use_val_checkpoint and best_lora_state is not None:
        model.load_state_dict(best_lora_state, strict=False)
        print(f"[lora_train] dùng checkpoint LoRA tốt nhất theo val (val_macro_f1={best_val_f1:.4f})")
    elif not use_val_checkpoint:
        print("[lora_train] dùng checkpoint LoRA ở bước train cuối cùng (không chọn qua val, "
              "đúng protocol CLIP-LoRA gốc)")

    os.makedirs(cfg.CHECKPOINTS_DIR, exist_ok=True)
    ckpt_path = os.path.join(cfg.CHECKPOINTS_DIR, "lora_medclip.pt")
    torch.save(lora_state_dict(model), ckpt_path)
    print(f"[lora_train] đã lưu LoRA checkpoint (nhẹ) vào {ckpt_path}")

    with torch.no_grad():
        final_text_features = build_text_features(model, device)
    test_preds, test_labels = _evaluate(model, final_text_features, device, cfg.TEST_CSV)

    metrics = compute_metrics(test_labels, test_preds)
    # Ghi lại chế độ chọn checkpoint đã dùng để tự tài liệu hoá file kết quả
    # (val_best = chọn theo val; final_iter = bước cuối, đúng protocol CLIP-LoRA gốc)
    metrics["checkpoint_selection"] = "val_best" if use_val_checkpoint else "final_iter"
    plot_confusion_matrix(
        test_labels, test_preds,
        save_path=os.path.join(cfg.RESULTS_DIR, f"{task_name}_confusion_matrix.png"),
        title=f"MedCLIP + LoRA few-shot ({n_shots}-shot)",
    )
    save_metrics_json(metrics, task_name)
    return metrics


def _aggregate_seed_runs(run_metrics: list) -> dict:
    """Gộp nhiều lần chạy (mỗi lần 1 seed khác nhau, cùng 1 mức shot) thành
    mean ± std. Giữ đúng schema của compute_metrics() (accuracy, macro_f1,
    per_class...) để compare_results.py đọc được như bình thường, chỉ thêm
    2 khoá *_std để vẽ error bar.
    """
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


def run_lora_shots_sweep(shots_list: list = None, seeds: list = None, **kwargs) -> dict:
    """Chạy run_lora_train() qua nhiều mức shot, MỖI mức lặp lại qua nhiều seed
    (đổi ảnh few-shot được bốc) rồi gộp mean±std — đúng chuẩn báo cáo few-shot
    VLM (CoOp, CLIP-LoRA: 3 seed/mức shot). Không báo cáo 1 con số, mà báo cáo
    đường cong accuracy/macro-F1 theo số shot (xem CLIP-LoRA/few_shot.png), có
    độ lệch chuẩn để biết kết quả ổn định hay chỉ do may rủi bốc trúng ảnh
    dễ/khó — quan trọng nhất ở 1-shot/2-shot.

    Lưu:
    - Từng lần chạy riêng theo seed: lora_fewshot_{n}shot_seed{s}_metrics.json
    - Gộp theo mức shot (mean±std qua các seed): lora_fewshot_{n}shot_metrics.json
      — đây là file compare_results.py đọc để vẽ đường cong.

    Trả về dict {n_shots: metrics_đã_gộp}.
    """
    shots_list = shots_list or cfg.FEWSHOT_SWEEP_SHOTS
    seeds = seeds or cfg.FEWSHOT_SWEEP_SEEDS
    summary = {}
    for n_shots in shots_list:
        run_metrics = []
        for seed in seeds:
            print(f"\n{'=' * 60}\n[lora_sweep] {n_shots}-shot, seed={seed}\n{'=' * 60}")
            metrics = run_lora_train(
                n_shots=n_shots, seed=seed,
                task_name=f"lora_fewshot_{n_shots}shot_seed{seed}",
                **kwargs,
            )
            run_metrics.append(metrics)

        agg = _aggregate_seed_runs(run_metrics)
        save_metrics_json(agg, f"lora_fewshot_{n_shots}shot")
        summary[n_shots] = agg
        print(f"[lora_sweep] {n_shots}-shot (qua {len(seeds)} seed): "
              f"accuracy={agg['accuracy']:.4f}±{agg['accuracy_std']:.4f}, "
              f"macro_f1={agg['macro_f1']:.4f}±{agg['macro_f1_std']:.4f}")
    return summary


if __name__ == "__main__":
    force_utf8_stdout()
    result = run_lora_train()
    print(json.dumps(result, ensure_ascii=False, indent=2))
