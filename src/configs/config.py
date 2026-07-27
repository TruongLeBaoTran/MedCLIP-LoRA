"""
Cấu hình tập trung cho toàn bộ dự án.

Mọi hằng số (đường dẫn, seed, tỉ lệ split, siêu tham số LoRA...) đặt ở đây
để các pipeline (baseline / zero-shot / lora) đọc chung, tránh hardcode rải rác.
"""
import os

# ----------------------------------------------------------------------
# Đường dẫn
# ----------------------------------------------------------------------
# Thư mục gốc của cả dự án (chứa MedCLIP/, CLIP-LoRA/, Dataset/, src/, notebooks/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATASET_DIR = os.path.join(PROJECT_ROOT, "Dataset")
IMAGES_DIR = os.path.join(DATASET_DIR, "BTXRD", "images")
LABELS_XLSX = os.path.join(DATASET_DIR, "classification.xlsx")

SPLITS_DIR = os.path.join(DATASET_DIR, "splits")
TRAIN_CSV = os.path.join(SPLITS_DIR, "train.csv")
VAL_CSV = os.path.join(SPLITS_DIR, "val.csv")
TEST_CSV = os.path.join(SPLITS_DIR, "test.csv")
SPLIT_STATS_JSON = os.path.join(SPLITS_DIR, "split_stats.json")

OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
CHECKPOINTS_DIR = os.path.join(OUTPUTS_DIR, "checkpoints")
RESULTS_DIR = os.path.join(OUTPUTS_DIR, "results")

# Nơi cache checkpoint MedCLIP pretrain tải về (cố định, không phụ thuộc cwd lúc chạy)
MEDCLIP_PRETRAINED_DIR = os.path.join(PROJECT_ROOT, "pretrained", "medclip-vit")

# Thư mục CLIP-LoRA gốc — dùng để import package `clip` cục bộ của họ (không pip
# install openai-clip riêng, tái dùng đúng bản họ đã vendor sẵn)
CLIP_LORA_REPO_DIR = os.path.join(PROJECT_ROOT, "CLIP-LoRA")

# ----------------------------------------------------------------------
# Nhãn 3 lớp — thứ tự cố định, dùng xuyên suốt (index = label id)
# ----------------------------------------------------------------------
CLASS_NAMES = ["Normal", "Benign", "Malignant"]
NUM_CLASSES = len(CLASS_NAMES)

# Giá trị gốc của cột `neoplasm` trong classification.xlsx -> label id
NEOPLASM_TO_LABEL = {
    "no tumor": 0,   # Normal
    "benign": 1,     # Benign
    "malignant": 2,  # Malignant
}

# ----------------------------------------------------------------------
# Chia dữ liệu
# ----------------------------------------------------------------------
SEED = 42

# Tỉ lệ train/val/test — xem lý do chọn 70/20/10 trong kế hoạch (đủ Malignant
# ở test set để confusion matrix có ý nghĩa). Đổi ở đây nếu cần thử tỉ lệ khác.
SPLIT_RATIO = (0.7, 0.2, 0.1)

# Các cột dùng làm proxy nhóm ảnh nghi cùng 1 ca bệnh (nhiều góc chụp) khi chia
# split, để tránh rò rỉ dữ liệu (xem hoi-dap.md / tom-tat-de-tai.md)
GROUP_PROXY_COLUMNS = ["center", "age", "gender", "bones_type", "tumor_type"]

# ----------------------------------------------------------------------
# Ảnh
# ----------------------------------------------------------------------
IMG_SIZE = 224
# Mean/std của MedCLIP (ảnh xám, xem medclip/constants.py) — dùng cho nhánh MedCLIP
MEDCLIP_IMG_MEAN = 0.5862785803043838
MEDCLIP_IMG_STD = 0.27950088968644304
# Mean/std ImageNet chuẩn — dùng cho nhánh baseline CNN/ViT
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# Mean/std của CLIP gốc (khác ImageNet, khác MedCLIP) — dùng cho nhánh CLIP+LoRA đối chứng
CLIP_IMG_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_IMG_STD = (0.26862954, 0.26130258, 0.27577711)

# ----------------------------------------------------------------------
# Nhiệm vụ 1 — Baseline CNN/ViT
# ----------------------------------------------------------------------
BASELINE_BACKBONE = "resnet50"  # "resnet50" hoặc "vit_b16" (timm)
BASELINE_BATCH_SIZE = 32
BASELINE_LR = 1e-4
BASELINE_EPOCHS = 15

# ----------------------------------------------------------------------
# Nhiệm vụ 2 — MedCLIP zero-shot
# ----------------------------------------------------------------------
# Cách gộp N điểm cosine/lớp thành 1 điểm/lớp trước khi argmax, khớp
# PromptClassifier gốc của MedCLIP (medclip/modeling_medclip.py:247-285):
# "max" = mặc định gốc MedCLIP, "mean" = chế độ ensemble=True gốc.
ZEROSHOT_PROMPT_AGGREGATION = "max"

# ----------------------------------------------------------------------
# Nhiệm vụ 3 — MedCLIP + LoRA few-shot
# ----------------------------------------------------------------------
# Khớp NGUYÊN VĂN giá trị mặc định của CLIP-LoRA gốc (CLIP-LoRA/run_utils.py:32-40,
# argparse --r/--alpha/--dropout_rate/--encoder/--params/--batch_size/--lr — họ
# chạy cố định 1 bộ này cho cả 11 dataset, không dò riêng từng dataset, xem
# CLIP-LoRA/README.md: "fixed hyperparameters for every task"). Kế thừa nguyên
# để giữ đúng tinh thần "1 cấu hình cố định", không tự đặt số theo cảm tính.
LORA_R = 2
LORA_ALPHA = 1
LORA_DROPOUT = 0.25
LORA_TARGET = "both"  # "vision" | "text" | "both" — khớp CLIP-LoRA --encoder default
LORA_TARGET_MODULES = ("query", "key", "value")  # khớp CLIP-LoRA --params default ['q','k','v']

FEWSHOT_N_SHOTS = 16  # số ảnh/lớp lấy từ train khi chỉ chạy 1 mức shot (run_lora_train mặc định)
FEWSHOT_SWEEP_SHOTS = [1, 2, 4, 8, 16]  # các mức shot chạy khi sweep — đúng chuẩn CoOp/CLIP-LoRA
FEWSHOT_SWEEP_SEEDS = [1, 2, 3]  # mỗi mức shot chạy lặp qua từng seed này (đổi ảnh few-shot được bốc), gộp mean±std
LORA_BATCH_SIZE = 32  # khớp CLIP-LoRA --batch_size default
LORA_LR = 2e-4  # khớp CLIP-LoRA --lr default
LORA_N_ITERS = 500  # tương đương --n_iters của CLIP-LoRA; TỔNG số bước thật = LORA_N_ITERS * n_shots
                     # (khớp lora.py:73 "total_iters = args.n_iters * args.shots", xem run_lora_train())

# Chọn checkpoint LoRA theo val (đánh giá định kỳ trong lúc train, giữ bước có
# val_macro_f1 cao nhất) hay lấy checkpoint ở bước train cuối cùng (đúng
# protocol CLIP-LoRA gốc — xem CLIP-LoRA/lora.py, biến VALIDATION=False, họ
# không dùng val cho bất kỳ việc gì, kể cả chọn checkpoint).
# - True (mặc định): ổn định hơn ở 1-2 shot, nhưng dùng thêm nhãn ngoài đúng
#   K-shot công bố -> không còn là "true few-shot" chặt theo Perez et al.
#   (NeurIPS 2021), dù val.csv là split có sẵn (không trích từ ngân sách
#   K-shot) nên vẫn đúng quy ước chung của CoOp/Tip-Adapter/CLIP-LoRA.
# - False: bám sát protocol CLIP-LoRA gốc 100%, "true few-shot" chặt hơn,
#   nhưng kết quả (đặc biệt 1-2 shot) có thể nhiễu hơn -> dựa hẳn vào
#   mean±std đa-seed (FEWSHOT_SWEEP_SEEDS) để thể hiện độ ổn định thay vì lọc
#   qua val. Xem README mục 6.7 để biết nên chọn cách nào khi chạy báo cáo cuối.
LORA_USE_VAL_CHECKPOINT_SELECTION = True

# Nhiệt độ/scale cho cosine similarity logits (giống logit_scale của CLIP/MedCLIP)
LOGIT_SCALE = 100.0

# ----------------------------------------------------------------------
# Nhiệm vụ 4 (đối chứng) — CLIP gốc + LoRA, cùng dataset/split/few-shot với
# nhiệm vụ 3, để trả lời câu hỏi "MedCLIP có thật sự tốt hơn CLIP gốc trên
# ảnh xương không, hay pretrain y khoa (X-quang ngực) không giúp ích cho
# domain xương?". Dùng CHUNG LORA_R/LORA_ALPHA/LORA_DROPOUT/FEWSHOT_* ở trên
# với nhiệm vụ 3 để so sánh có kiểm soát (chỉ đổi backbone, không đổi capacity
# LoRA hay số shot).
# ----------------------------------------------------------------------
CLIP_BACKBONE = "ViT-B/16"
CLIP_LORA_ENABLE = ("q", "k", "v")  # tương đương LORA_TARGET_MODULES của MedCLIP

DEVICE = "cuda"  # đổi thành "cpu" nếu máy không có GPU (các wrapper đều device-agnostic)
