# Phân lớp ảnh X-quang xương với MedCLIP + LoRA

Source code triển khai đầy đủ pipeline thực nghiệm cho đề tài phân lớp ảnh X-quang xương 3 lớp (Normal / Benign / Malignant) trên bộ dữ liệu BTXRD: **tiền xử lý dữ liệu** (chia split an toàn, chống rò rỉ), **4 phương pháp phân lớp** (Baseline CNN, MedCLIP zero-shot, MedCLIP + LoRA few-shot, CLIP gốc + LoRA đối chứng), và **đánh giá/so sánh kết quả** giữa 4 phương pháp. Xem thêm định hướng đề tài ở [tom-tat-de-tai.md](tom-tat-de-tai.md).

## 1. Bốn nhiệm vụ

| # | Nhiệm vụ | Train? | Mô tả |
|---|---|---|---|
| 1 | Baseline CNN | Có (full train set) | Fine-tune ResNet50 (ImageNet-pretrained) trên toàn bộ train split — mốc so sánh truyền thống, không liên quan MedCLIP. |
| 2 | MedCLIP zero-shot | Không | So cosine similarity giữa ảnh và prompt văn bản 3 lớp bằng MedCLIP pretrain gốc, không dùng dữ liệu BTXRD để train. |
| 3 | MedCLIP + LoRA (few-shot) | Có (rất ít ảnh) | Đóng băng MedCLIP, chỉ train ma trận LoRA (rank thấp) chèn vào Q/K/V attention. Chạy quét qua **1, 2, 4, 8, 16 ảnh/lớp** (`FEWSHOT_SWEEP_SHOTS`), mỗi mức lặp 3 seed — không chỉ 1 con số. |
| 4 | CLIP gốc + LoRA (đối chứng) | Có (rất ít ảnh) | Y hệt nhiệm vụ 3 (cùng split/few-shot/siêu tham số, cùng **1/2/4/8/16-shot**) nhưng dùng CLIP gốc thay MedCLIP — trả lời câu hỏi "pretrain y khoa (X-quang ngực) có thật sự giúp ích trên ảnh xương hay không". |

Kết quả 4 nhiệm vụ gộp vào 1 bảng + biểu đồ qua `compare_results.py` — đây là kết quả cốt lõi của đề tài.

## 2. Cấu trúc project

Không nằm trong repo Git này — tự chuẩn bị theo hướng dẫn ở mục 5 (`MedCLIP/`, `CLIP-LoRA/`, `Dataset/` đều trong `.gitignore`):

```
Code/
├── MedCLIP/            # repo gốc MedCLIP (cài qua pip install -e, đã vá 2 bug — xem mục 5)
├── CLIP-LoRA/           # repo gốc CLIP-LoRA (tham khảo + vendor 1 phần vào src/lora/)
├── Dataset/
│   ├── classification.xlsx   # nhãn 3 lớp + metadata từng ảnh
│   ├── BTXRD/images/          # 3746 ảnh X-quang gốc
│   └── splits/                 # sinh bởi data/prepare_split.py — mục 3
├── src/
│   ├── configs/config.py        # mọi hằng số: đường dẫn, seed, split ratio, siêu tham số LoRA...
│   ├── data/
│   │   ├── prepare_split.py      # tiền xử lý — chia split an toàn (mục 3)
│   │   └── dataset.py             # BoneXrayDataset, sample_few_shot()
│   ├── common/                  # seed.py, device.py, transforms.py, metrics.py — dùng chung 4 task
│   ├── lora/                    # vendor từ CLIP-LoRA: LinearLoRA (MedCLIP) + PlainMultiheadAttentionLoRA (CLIP)
│   ├── models/
│   │   ├── medclip_wrapper.py / lora_medclip.py    # nhiệm vụ 3
│   │   └── clip_wrapper.py / lora_clip.py           # nhiệm vụ 4
│   ├── prompts_bone.py           # prompt văn bản 3 lớp — dùng chung MedCLIP/CLIP để so sánh công bằng
│   ├── pipelines/
│   │   ├── baseline.py / zeroshot.py                     # nhiệm vụ 1 / 2
│   │   ├── lora_train.py                                   # nhiệm vụ 3 (run_lora_train, run_lora_shots_sweep)
│   │   └── clip_zeroshot.py / clip_lora_train.py         # nhiệm vụ 4
│   ├── compare_results.py         # gộp bảng so sánh + biểu đồ
│   └── predict.py                  # demo dự đoán 1 ảnh bất kỳ — không phải nhiệm vụ chính, xem mục 8
└── notebooks/             # 7 notebook mỏng, chỉ gọi lại hàm trong src/pipelines/ — chạy trên Colab
```

## 3. Tiền xử lý dữ liệu

- **Nguồn**: `Dataset/BTXRD/images/` (3746 ảnh) + nhãn ở cột `neoplasm` trong `classification.xlsx` (`no tumor`/`benign`/`malignant`). Không dùng `btxrd-v2.1` (thiếu lớp Normal).
- **Rò rỉ dữ liệu**: dataset không có patient ID thật, nhiều ảnh nghi cùng 1 ca chụp nhiều góc. Xử lý bằng gom nhóm proxy `(center, age, gender, bones_type, tumor_type)`, mỗi nhóm gán nhãn đa số, luôn giữ trọn 1 nhóm trong 1 tập khi chia (không bao giờ xé lẻ).
- **Chia split**: stratified theo nhãn trong từng lớp, tỉ lệ **70/20/10** (train/val/test) tính theo tổng số ảnh, seed cố định (`cfg.SEED=42`), chạy 1 lần: `python -m data.prepare_split` → ghi `Dataset/splits/{train,val,test}.csv` cố định cho mọi pipeline đọc lại.
- **Kết quả thật**: train 2630 / val 759 / test 357 ảnh, 0 nhóm bị xé lẻ, tỉ lệ 3 lớp mỗi tập sát tỉ lệ gốc (~50/41/9%). Malignant ở test chỉ 29 ảnh — macro-F1 trên support nhỏ này dao động khá mạnh, xem mục 8.
- **Tiền xử lý ảnh** (lúc load, không sinh file mới, xem `common/transforms.py`):
  - Ảnh BTXRD có tỉ lệ khung hình rất không đồng nhất (từ gần vuông đến dài/hẹp bất thường). Có 2 cách xử lý phổ biến, cả 2 đều **không dùng** vì lý do riêng:
    - *Resize thẳng về 224×224* (kéo giãn ảnh cho vừa khung vuông) → làm **méo hình xương** (tỉ lệ dài/rộng thật bị sai lệch).
    - *CenterCrop* (cách CLIP gốc làm mặc định) → cắt lấy đúng vùng vuông ở giữa, vứt phần rìa → với ảnh dài/hẹp có nguy cơ **cắt mất đúng vùng u/tổn thương** nếu nó không nằm ở giữa khung hình.
  - Cách đang dùng: **`PadToSquare`** — đệm thêm viền đen vào 2 bên cạnh ngắn hơn để ảnh thành hình vuông trước (giữ nguyên 100% nội dung ảnh gốc, không cắt gì cả), rồi mới resize về 224×224. Đánh đổi: ảnh có thêm viền đen thừa (không chứa thông tin), nhưng đổi lại đảm bảo **không bao giờ mất vùng chẩn đoán**.
  - Dataset lẫn cả ảnh mode grayscale và RGB (851/3746 ảnh RGB nhưng thực chất vẫn là ảnh xám lưu dưới định dạng RGB, không có nội dung màu) — mỗi transform tự chuẩn hoá về đúng số kênh cần (1 kênh cho MedCLIP, 3 kênh cho baseline/CLIP), không phụ thuộc mode gốc của file.
  - Mỗi backbone normalize theo mean/std riêng: MedCLIP (ảnh xám), ImageNet (baseline), CLIP (nhiệm vụ 4) — bắt buộc phải khớp đúng thống kê mà mỗi model pretrain gốc dùng.

## 4. Triển khai MedCLIP + LoRA

### 4.1. Những phần lấy nguyên văn từ CLIP-LoRA

- **Code gắn LoRA** (`lora/layers.py::LinearLoRA` — công thức `h = Wx + scaling·B@A@x`, cách khởi tạo A/B, cách áp dropout trước khi tính phần điều chỉnh LoRA): **vendor nguyên văn** từ `CLIP-LoRA/loralib/layers.py`.
- **Toàn bộ 8 siêu tham số LoRA** (`configs/config.py`): lấy đúng **giá trị mặc định thật** mà CLIP-LoRA dùng cố định cho cả 11 dataset benchmark của họ (`CLIP-LoRA/run_utils.py:32-40`):

  | Tham số | CLIP-LoRA gốc | `config.py` |
  |---|---|---|
  | encoder | `both` | `LORA_TARGET = "both"` |
  | params (ma trận LoRA) | `[q,k,v]` | `LORA_TARGET_MODULES` |
  | r / alpha / dropout | `2 / 1 / 0.25` | `LORA_R/ALPHA/DROPOUT` |
  | lr / batch_size | `2e-4 / 32` | khớp |
  | n_iters | `500 × shots` | `LORA_N_ITERS × n_shots` |
  | logit_scale | `100` (cố định, không phải giá trị học được) | `LOGIT_SCALE = 100.0` |

- **Vòng lặp train**: cùng cấu trúc — loss = cross-entropy trên `logit_scale × cosine_similarity(ảnh, text_feature từng lớp)`, optimizer AdamW, `CosineAnnealingLR`, chỉ tham số LoRA được cập nhật (`mark_only_lora_as_trainable`), checkpoint chỉ lưu phần LoRA (`lora_state_dict`, nhẹ vài trăm KB) — khớp `CLIP-LoRA/lora.py`.
- **Cách đánh giá few-shot**: quét nhiều mức shot (1/2/4/8/16) × nhiều seed rồi lấy mean±std, không báo cáo 1 con số đơn — đúng convention CoOp/CLIP-LoRA.

### 4.2. Những điểm thay đổi

**Khác bắt buộc**

CLIP-LoRA gốc chỉ biết gắn LoRA vào `nn.MultiheadAttention` (kiến trúc attention của CLIP, Q/K/V gộp chung 1 tensor). MedCLIP không dùng kiến trúc đó — vision encoder là Swin Transformer, text encoder là BERT, cả hai có Q/K/V là 3 `nn.Linear` **tách rời sẵn**. Nên code gốc của CLIP-LoRA đơn giản là không chạy được trên MedCLIP, bắt buộc phải viết `apply_lora_medclip()` (`models/lora_medclip.py`) riêng — nhưng chỉ là phần **duyệt-module để tìm đúng layer cần gắn** cho phù hợp cấu trúc MedCLIP, còn phần lõi vẫn dùng thẳng `LinearLoRA` đã vendor y hệt ở mục 4.1, **không viết thuật toán LoRA mới**. Đã kiểm chứng: gắn đúng 72 layer (36 vision + 36 text), output không đổi trước khi train (do B khởi tạo = 0).

**Khác có chủ đích**

- **Tiền xử lý ảnh**: dùng `PadToSquare` thay vì `RandomResizedCrop`/`CenterCrop` của CLIP-LoRA gốc — lý do thuộc về đặc thù ảnh X-quang, xem mục 3.
- **Cách chọn checkpoint**: có thêm chế độ `val_best` (tắt được, `LORA_USE_VAL_CHECKPOINT_SELECTION=False` sẽ về đúng 100% protocol gốc) — CLIP-LoRA gốc nạp val nhưng không dùng cho bất kỳ việc gì (`CLIP-LoRA/lora.py:37`, `VALIDATION = False` hardcode). Đây là đánh đổi về phương pháp luận few-shot, không liên quan ảnh y khoa — xem chi tiết ở mục 7.

## 5. Cài đặt

Repo này chỉ chứa code tự viết (`src/`, `notebooks/`) — không kèm 2 repo tham khảo/vendor (`MedCLIP/`, `CLIP-LoRA/`) hay dataset (`Dataset/`), cần tự chuẩn bị trước:

```bash
# 1) Clone 2 repo tham khảo vào đúng vị trí (sibling với src/), y hệt cấu trúc ở mục 2
git clone https://github.com/RyanWangZf/MedCLIP.git
git clone https://github.com/MaxZanella/CLIP-LoRA.git

# 2) Cài MedCLIP dạng editable
pip install -e MedCLIP --no-deps   # --no-deps để tránh bị ghim theo transformers<=4.24.0 quá cũ

# 3) Cài dependency của src/ (bao gồm transformers bản mới hơn, và ftfy/regex
#    cần cho package `clip` cục bộ nằm trong CLIP-LoRA/clip/ — dùng cho nhiệm vụ 4)
pip install -r src/requirements.txt
```

Nhiệm vụ 4 dùng package `clip` **cục bộ** có sẵn trong `CLIP-LoRA/clip/` (không cài `openai-clip` riêng) — `models/clip_wrapper.py` tự thêm `CLIP-LoRA/` vào `sys.path`. Lần đầu chạy sẽ tự tải checkpoint CLIP ViT-B/16 gốc (~340MB) về `~/.cache/clip`.

**Dataset**: đặt ảnh BTXRD vào `Dataset/BTXRD/images/` và nhãn vào `Dataset/classification.xlsx` (xem cấu trúc cột ở mục 3) trước khi chạy `python -m data.prepare_split`.

## 6. Hai cách chạy

Logic thật nằm trong `src/pipelines/*.py` (mỗi file có 1 hàm `run_xxx(...)`) — chạy local hay Colab đều gọi đúng cùng 1 code.

### Cách 1 — Local (dòng lệnh)

```bash
cd src
python -m data.prepare_split                 # bước 0, chỉ chạy 1 lần
python -m pipelines.baseline                  # nhiệm vụ 1
python -m pipelines.zeroshot                   # nhiệm vụ 2
python -c "from pipelines.lora_train import run_lora_shots_sweep; run_lora_shots_sweep()"           # nhiệm vụ 3
python -m pipelines.clip_zeroshot
python -c "from pipelines.clip_lora_train import run_clip_lora_shots_sweep; run_clip_lora_shots_sweep()"  # nhiệm vụ 4
python compare_results.py                      # gộp bảng + biểu đồ

# Demo — dự đoán thử 1 ảnh bất kỳ (không phải nhiệm vụ chính, chỉ minh hoạ)
python predict.py "duong/dan/toi/anh.jpg"
```

Đổi siêu tham số không cần sửa `config.py`, gọi thẳng trong Python, ví dụ: `run_lora_shots_sweep(shots_list=[4,8,16], seeds=[1,2,3], r=8, target="vision")`.

### Cách 2 — Google Colab (notebook)

`notebooks/` có 7 notebook mỏng, chạy theo thứ tự: `01_prepare_split` → `02_train_baseline` → `03_eval_zeroshot` → `04_train_lora` (nên bật GPU) → `05_compare_results` → `06_clip_lora_compare` → `07_demo_predict` (dự đoán thử 1 ảnh, xem mục 8). Mỗi notebook chỉ gọi lại hàm `run_xxx()` trong `src/pipelines/`, không có logic riêng — cell đầu tiên cần bỏ comment và sửa `PROJECT_ROOT` cho khớp đường dẫn trên Drive.

**Lưu ý khi mở notebook**: các file `.ipynb` trong repo được sinh sẵn nhưng **chưa từng chạy** — mọi cell đều rỗng output. Đây không phải lỗi, chỉ vì output chỉ được lưu vào file sau khi thực sự bấm chạy (Run/Shift+Enter). Sau khi chạy, kết quả (log tiến trình, dict metrics, ảnh confusion matrix) hiện ngay dưới từng cell, không cần mở panel nào khác.

## 7. Hai chế độ chọn checkpoint — `val_best` vs `final_iter`

- **`val_best`** (mặc định): cứ mỗi `val_every` bước, đánh giá trên `val.csv`, giữ lại checkpoint có val macro-F1 cao nhất — tránh lấy đúng trạng thái ở bước cuối, có thể đã overfit lên vài ảnh few-shot.
- **`final_iter`**: train đúng số bước cố định rồi lấy thẳng checkpoint cuối — đúng 100% protocol CLIP-LoRA gốc (họ không dùng val cho bất kỳ việc gì, xem `CLIP-LoRA/lora.py:37`).
- Đổi qua `configs/config.py::LORA_USE_VAL_CHECKPOINT_SELECTION`, hoặc truyền `use_val_checkpoint=True/False` khi gọi `run_lora_train()`/`run_clip_lora_train()`. Mỗi file kết quả tự ghi khoá `"checkpoint_selection"` để biết đã chạy chế độ nào.
- **Đánh đổi**: theo tranh luận "true few-shot learning" (Perez et al., NeurIPS 2021), dùng thêm val để chọn checkpoint — dù val là split có sẵn, không trích từ ngân sách K-shot — vẫn là dùng thêm nhãn ngoài đúng con số few-shot công bố. `val_best` ổn định hơn ở 1-2 shot nhưng kém "sạch" hơn theo tranh luận này; `final_iter` bám sát CLIP-LoRA gốc, dễ biện minh trước hội đồng, nhưng std đa-seed có thể lớn hơn ở shot thấp. Nếu dùng `val_best` cho số liệu báo cáo cuối, nên nêu rõ đây là deviation có chủ đích (lý do: BTXRD mất cân bằng lớp nặng hơn 11 benchmark gốc của CLIP-LoRA), không trình bày như tái hiện đúng nguyên protocol.

## 8. Ghi chú thêm

- **Demo dự đoán 1 ảnh** (`src/predict.py`, notebook `07_demo_predict.ipynb`): không phải nhiệm vụ chính, chỉ để xem trực quan model dự đoán ra sao trên 1 ảnh cụ thể. Tự ưu tiên dùng checkpoint MedCLIP + LoRA đã train (`outputs/checkpoints/lora_medclip.pt`) nếu có, tự rơi về zero-shot nếu chưa train. Chạy: `python predict.py duong/dan/anh.jpg`.
- Đánh giá dùng chung `common/metrics.py` cho cả 4 nhiệm vụ: accuracy, macro-F1, precision/recall/F1 từng lớp — ưu tiên macro-F1 vì Malignant chỉ ~9% dữ liệu. Minh chứng cụ thể: CLIP zero-shot có accuracy cao hơn MedCLIP zero-shot (0.457 vs 0.300) nhưng macro-F1 gần ngang nhau và CLIP dự đoán 0% đúng Malignant (dồn về lớp đa số Benign) — accuracy cao không đồng nghĩa mô hình tốt hơn.
- Few-shot báo cáo dạng **đường cong theo số shot** (`FEWSHOT_SWEEP_SHOTS=[1,2,4,8,16]`), mỗi mức chạy qua nhiều seed (`FEWSHOT_SWEEP_SEEDS=[1,2,3]`) rồi lấy mean±std — không báo cáo 1 con số đơn, đúng chuẩn CoOp/CLIP-LoRA. Seed few-shot chỉ quyết định ảnh nào được bốc từ `train.csv`; `val.csv`/`test.csv` cố định tuyệt đối, không đổi theo seed.
- `apply_lora_medclip()`/`apply_lora_clip()` đã kiểm chứng: gắn đúng 72 layer (MedCLIP: 36 vision + 36 text) / 24 layer (CLIP: 12 vision + 12 text), output giữ nguyên trước khi train (B=0).
- `configs/config.py` là nơi duy nhất cần sửa khi muốn đổi siêu tham số mặc định. Checkpoint pretrain (MedCLIP, CLIP) tự tải về, không cần tải tay.
- Chạy đủ (15 epoch baseline, sweep 1/2/4/8/16-shot × 3 seed cho cả 2 backbone) nên thực hiện trên Colab GPU — CPU sẽ rất lâu.
