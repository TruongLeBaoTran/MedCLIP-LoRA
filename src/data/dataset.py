"""
Dataset đọc ảnh BTXRD từ file split CSV (sinh bởi prepare_split.py) và
hàm lấy few-shot subset dùng cho nhiệm vụ 3 (MedCLIP + LoRA).
"""
import os

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from configs import config as cfg


class BoneXrayDataset(Dataset):
    """data: đường dẫn tới file CSV split, HOẶC 1 DataFrame có sẵn
    (dùng khi cần bọc 1 subset, ví dụ few-shot subset từ sample_few_shot)."""

    def __init__(self, data, images_dir: str = cfg.IMAGES_DIR, transform=None):
        self.df = pd.read_csv(data) if isinstance(data, str) else data.reset_index(drop=True)
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.images_dir, row["image_filename"])
        image = Image.open(img_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, int(row["label"])


def sample_few_shot(csv_path: str, n_shots: int, seed: int = cfg.SEED) -> pd.DataFrame:
    """Lấy ngẫu nhiên n_shots ảnh/lớp từ 1 split (thường dùng trên train.csv) để
    huấn luyện LoRA few-shot. Nếu 1 lớp có ít ảnh hơn n_shots, lấy hết lớp đó.
    """
    df = pd.read_csv(csv_path)
    parts = []
    for label, group in df.groupby("label"):
        n = min(n_shots, len(group))
        parts.append(group.sample(n=n, random_state=seed))
    return pd.concat(parts).reset_index(drop=True)
