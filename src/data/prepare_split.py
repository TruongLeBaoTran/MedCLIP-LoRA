"""
Script tiền xử lý: chia BTXRD thành train/val/test 1 lần duy nhất, lưu ra CSV
cố định để mọi pipeline (baseline / zero-shot / lora) đọc lại đúng cùng 1 split.

Cách chia:
1. Đọc nhãn 3 lớp (Normal/Benign/Malignant) từ cột `neoplasm` trong classification.xlsx.
2. Gom nhóm các ảnh nghi cùng 1 ca bệnh (ví dụ nhiều góc chụp của cùng 1 tổn thương)
   bằng proxy (center, age, gender, bones_type, tumor_type) — dataset không có patient ID
   thật, đây là cách xấp xỉ tốt nhất để tránh rò rỉ dữ liệu giữa train/val/test.
3. Với mỗi nhóm, gán 1 nhãn đại diện (nhãn xuất hiện nhiều nhất trong nhóm).
4. Chia theo tỉ lệ SPLIT_RATIO, đảm bảo cả nhóm luôn nằm trọn trong 1 tập
   (không bao giờ 1 nhóm bị xé lẻ giữa train/val/test).

Chạy: python -m data.prepare_split (từ thư mục src/)
"""
import json
import os
import random
import sys
from collections import Counter, defaultdict

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs import config as cfg
from common.console import force_utf8_stdout


def _build_group_key(row) -> str:
    """Ghép các cột proxy thành 1 chuỗi định danh nhóm. fillna trước khi ghép
    vì tumor_type là NaN với toàn bộ ảnh Normal (không có tumor)."""
    parts = []
    for col in cfg.GROUP_PROXY_COLUMNS:
        value = row[col]
        parts.append("NA" if pd.isna(value) else str(value))
    return "|".join(parts)


def _split_groups_by_ratio(groups_of_class: list, seed: int):
    """Chia danh sách nhóm (đã random shuffle) thành 3 phần theo SPLIT_RATIO,
    tính theo TỔNG SỐ ẢNH (không phải số nhóm) để tỉ lệ ảnh cuối cùng sát mục tiêu.
    Trả về 3 list nhóm: (train_groups, val_groups, test_groups).
    """
    rng = random.Random(seed)
    groups_of_class = groups_of_class[:]
    rng.shuffle(groups_of_class)

    total_images = sum(len(g["rows"]) for g in groups_of_class)
    train_ratio, val_ratio, _test_ratio = cfg.SPLIT_RATIO
    train_quota = total_images * train_ratio
    val_quota = total_images * val_ratio

    train_groups, val_groups, test_groups = [], [], []
    train_count = val_count = 0
    for g in groups_of_class:
        n = len(g["rows"])
        if train_count < train_quota:
            train_groups.append(g)
            train_count += n
        elif val_count < val_quota:
            val_groups.append(g)
            val_count += n
        else:
            test_groups.append(g)
    return train_groups, val_groups, test_groups


def prepare_split():
    df = pd.read_excel(cfg.LABELS_XLSX)

    # Nhãn 3 lớp
    df["label"] = df["neoplasm"].map(cfg.NEOPLASM_TO_LABEL)
    assert df["label"].isna().sum() == 0, "Có giá trị neoplasm lạ chưa map được sang label"

    # Kiểm tra file ảnh thật sự tồn tại (dùng đúng cột image_filename, không tự ghép đuôi)
    missing = [f for f in df["image_filename"] if not os.path.exists(os.path.join(cfg.IMAGES_DIR, f))]
    if missing:
        print(f"[CẢNH BÁO] {len(missing)} ảnh trong classification.xlsx không tìm thấy file, sẽ bỏ qua.")
        df = df[~df["image_filename"].isin(missing)].reset_index(drop=True)

    # Gom nhóm theo proxy ca bệnh
    df["group_key"] = df.apply(_build_group_key, axis=1)
    groups_raw = defaultdict(list)
    for _, row in df.iterrows():
        groups_raw[row["group_key"]].append(row)

    # Với mỗi nhóm: nhãn đại diện = nhãn xuất hiện nhiều nhất trong nhóm
    mixed_label_groups = 0
    groups = []
    for key, rows in groups_raw.items():
        labels_in_group = [r["label"] for r in rows]
        label_counts = Counter(labels_in_group)
        majority_label, _ = label_counts.most_common(1)[0]
        if len(label_counts) > 1:
            mixed_label_groups += 1
        groups.append({"key": key, "label": majority_label, "rows": rows})

    if mixed_label_groups:
        print(f"[CẢNH BÁO] {mixed_label_groups} nhóm proxy có lẫn nhãn khác nhau "
              f"(nhiều khả năng trùng proxy ngẫu nhiên, không cùng ca bệnh thật) "
              f"— đã gán theo nhãn đa số trong nhóm.")

    # Chia riêng theo từng lớp để giữ đúng tỉ lệ 3 lớp ở mỗi tập (stratified)
    groups_by_class = defaultdict(list)
    for g in groups:
        groups_by_class[g["label"]].append(g)

    split_rows = {"train": [], "val": [], "test": []}
    for label, class_groups in groups_by_class.items():
        train_g, val_g, test_g = _split_groups_by_ratio(class_groups, seed=cfg.SEED + label)
        for split_name, group_list in [("train", train_g), ("val", val_g), ("test", test_g)]:
            for g in group_list:
                for row in g["rows"]:
                    split_rows[split_name].append({
                        "image_filename": row["image_filename"],
                        "label": int(row["label"]),
                        "group_key": row["group_key"],
                    })

    os.makedirs(cfg.SPLITS_DIR, exist_ok=True)
    out_paths = {"train": cfg.TRAIN_CSV, "val": cfg.VAL_CSV, "test": cfg.TEST_CSV}
    for split_name, rows in split_rows.items():
        pd.DataFrame(rows)[["image_filename", "label"]].to_csv(out_paths[split_name], index=False)

    # --- Kiểm chứng: không nhóm nào bị xé lẻ giữa các split ---
    group_to_splits = defaultdict(set)
    for split_name, rows in split_rows.items():
        for row in rows:
            group_to_splits[row["group_key"]].add(split_name)
    torn_groups = sum(1 for splits in group_to_splits.values() if len(splits) > 1)

    stats = {
        "total_images": len(df),
        "total_groups": len(groups),
        "torn_groups": torn_groups,  # kỳ vọng 0
        "per_split": {},
    }
    for split_name, rows in split_rows.items():
        label_counts = Counter(r["label"] for r in rows)
        stats["per_split"][split_name] = {
            "total": len(rows),
            **{cfg.CLASS_NAMES[k]: label_counts.get(k, 0) for k in range(cfg.NUM_CLASSES)},
        }

    with open(cfg.SPLIT_STATS_JSON, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    assert torn_groups == 0, "Có nhóm bị xé lẻ giữa các split — kiểm tra lại thuật toán chia!"
    print(f"\nĐã lưu split vào {cfg.SPLITS_DIR}")


if __name__ == "__main__":
    force_utf8_stdout()
    prepare_split()
