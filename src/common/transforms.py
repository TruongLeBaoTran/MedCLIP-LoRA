"""
Tiền xử lý ảnh dùng chung.

BTXRD có kích thước/tỉ lệ ảnh rất không đồng nhất (từ ~400x400 đến hơn
2500x3000, vừa dọc vừa ngang) — nếu resize thẳng về 224x224 sẽ làm méo
hình xương. Nên PAD VỀ HÌNH VUÔNG trước, rồi mới resize (đúng cách paper
MedCLIP xử lý ảnh gốc).

Có 2 bộ transform riêng vì 2 nhánh model cần chuẩn hoá khác nhau:
- MedCLIP: ảnh xám 1 kênh, normalize theo mean/std riêng của MedCLIP
  (vision_model tự lặp lại thành 3 kênh bên trong nếu cần).
- Baseline CNN/ViT: ảnh RGB 3 kênh, normalize theo chuẩn ImageNet.
"""
from PIL import Image, ImageOps
import torchvision.transforms as T

from configs import config as cfg


class PadToSquare:
    """Pad ảnh về hình vuông (giữ nguyên tỉ lệ gốc, không crop/méo hình)."""

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        size = max(w, h)
        return ImageOps.pad(img, (size, size), color=0, centering=(0.5, 0.5))


def get_medclip_transform(train: bool = False) -> T.Compose:
    """Transform theo đúng chuẩn tiền xử lý gốc của MedCLIP (ảnh xám 1 kênh)."""
    ops = [PadToSquare(), T.Grayscale(num_output_channels=1)]
    if train:
        ops.append(T.RandomHorizontalFlip(p=0.5))
    ops += [
        T.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[cfg.MEDCLIP_IMG_MEAN], std=[cfg.MEDCLIP_IMG_STD]),
    ]
    return T.Compose(ops)


def get_baseline_transform(train: bool = False) -> T.Compose:
    """Transform cho baseline CNN/ViT (ảnh RGB, chuẩn ImageNet)."""
    ops = [PadToSquare(), T.Lambda(lambda img: img.convert("RGB"))]
    if train:
        ops += [T.RandomHorizontalFlip(p=0.5), T.RandomRotation(10)]
    ops += [
        T.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=cfg.IMAGENET_MEAN, std=cfg.IMAGENET_STD),
    ]
    return T.Compose(ops)


def get_clip_transform(train: bool = False) -> T.Compose:
    """Transform cho CLIP gốc (nhiệm vụ 4, đối chứng). Vẫn pad-to-square trước
    (nhất quán với các nhánh khác, tránh méo/cắt mất vùng xương quan trọng) —
    KHÔNG dùng `preprocess` mặc định của clip.load() vì nó CenterCrop, rủi ro
    cắt mất vùng xương với ảnh X-quang tỉ lệ dài/hẹp bất thường."""
    ops = [PadToSquare(), T.Lambda(lambda img: img.convert("RGB"))]
    if train:
        ops.append(T.RandomHorizontalFlip(p=0.5))
    ops += [
        T.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=cfg.CLIP_IMG_MEAN, std=cfg.CLIP_IMG_STD),
    ]
    return T.Compose(ops)
