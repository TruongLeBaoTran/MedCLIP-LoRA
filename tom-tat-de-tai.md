# Phân lớp ảnh X-quang xương với MedCLIP + LoRA

## Đề tài
Phân lớp ảnh X-quang xương thành 3 lớp: **Normal / Benign / Malignant**, dùng MedCLIP (Med-VLM) kết hợp LoRA để fine-tune trong bối cảnh dữ liệu ít (few-shot). Có thêm 1 nhiệm vụ đối chứng dùng CLIP gốc thay MedCLIP để trả lời câu hỏi pretrain y khoa (X-quang ngực) có thật sự giúp ích trên ảnh xương hay không.

## Dataset — BTXRD
- 3.746 ảnh, nhãn ở cột `neoplasm` trong `classification.xlsx`: Normal ~50% / Benign ~41% / Malignant ~9%.
- Ảnh gốc ở `Dataset/BTXRD/images/` (đủ cả 3 lớp, dùng cột `image_filename` lấy đúng tên file — có lẫn `.jpeg`/`.jpg`). Không dùng `btxrd-v2.1` (chỉ có ảnh có tumor, thiếu hoàn toàn lớp Normal).

### Các vấn đề tiền xử lý đã xử lý
- **Rò rỉ dữ liệu (leakage)**: dataset không có patient ID thật, nhiều ảnh nghi cùng 1 ca bệnh chụp nhiều góc. Xử lý bằng gom nhóm proxy `(center, age, gender, bones_type, tumor_type)`, mỗi nhóm gán nhãn đa số, luôn giữ trọn 1 nhóm trong cùng 1 tập khi chia train/val/test — không bao giờ xé lẻ.
- **Kích thước/tỉ lệ ảnh không đồng nhất** (~150×310 đến 3594×4881, vừa dọc vừa ngang) → pad về hình vuông trước rồi mới resize 224×224, không resize thẳng (méo hình) hay CenterCrop (nguy cơ cắt mất đúng vùng tổn thương).
- **Color mode lẫn lộn** (2895 ảnh grayscale, 851 ảnh RGB — đã kiểm tra, RGB không có nội dung màu thật, chỉ khác cách encode file) → mỗi transform tự chuẩn hoá về đúng số kênh cần cho từng model, không phụ thuộc mode gốc của file.
- **Mất cân bằng lớp nặng** (Malignant chỉ ~9%) → stratified split theo `neoplasm`, weighted loss cho baseline, báo cáo bằng macro-F1 + per-class precision/recall (không chỉ accuracy — accuracy cao có thể chỉ do đoán dồn về lớp đa số).
- Kết quả chia thật: **train 2630 / val 759 / test 357 ảnh, 0 nhóm bị xé lẻ**, tỉ lệ 3 lớp mỗi tập sát tỉ lệ gốc.

## 4 nhiệm vụ chính (cốt lõi kỹ thuật của đề tài)

Bốn cái này chạy xong, đổ chung vào **1 bảng so sánh + đường cong theo số shot + confusion matrix** — đó là kết quả cốt lõi của cả báo cáo. Mọi thứ khác (data loader, tiền xử lý, script phụ trợ...) chỉ để 4 cái này chạy được và có số để trình bày.

### 1. Baseline CNN
Fine-tune ResNet50 (ImageNet-pretrained) trên toàn bộ train split. Không đụng gì tới MedCLIP — mốc so sánh truyền thống.

### 2. MedCLIP zero-shot
Không train. Load checkpoint MedCLIP có sẵn, viết prompt text cho 3 lớp, đo accuracy bằng cosine similarity ảnh–prompt.

### 3. MedCLIP + LoRA few-shot
Train, nhưng chỉ train phần LoRA (rank thấp, chèn vào attention Q/K/V), MedCLIP gốc đóng băng hoàn toàn. Phần kỹ thuật buộc phải tự viết: `apply_lora()` của CLIP-LoRA chỉ nhận diện `nn.MultiheadAttention` (kiến trúc attention của CLIP gốc), còn MedCLIP dùng Swin Transformer (vision) + BERT (text) có Q/K/V là 3 `nn.Linear` tách rời sẵn, khác cấu trúc. Tuy vậy phần lõi thuật toán LoRA (`LinearLoRA`) và toàn bộ siêu tham số (r=2, alpha=1, dropout=0.25, lr=2e-4, batch_size=32, n_iters=500×shots...) đều lấy **nguyên văn** từ CLIP-LoRA gốc, không tự đặt theo cảm tính. Chạy quét qua **1/2/4/8/16 ảnh/lớp** (không chỉ 1 mức), mỗi mức lặp 3 seed, báo cáo mean±std — đúng chuẩn báo cáo few-shot VLM (CoOp/Tip-Adapter/CLIP-LoRA).

### 4. CLIP gốc + LoRA (đối chứng)
Y hệt nhiệm vụ 3 (cùng split/few-shot/siêu tham số LoRA, chỉ đổi backbone) nhưng dùng CLIP gốc thay MedCLIP. Trả lời câu hỏi: pretrain y khoa (MedCLIP pretrain trên X-quang **ngực**) có thật sự giúp ích trên ảnh X-quang **xương** hay không, so với CLIP pretrain ảnh tổng quát — không so trực tiếp với số liệu công bố trong paper CLIP-LoRA (khác domain/dataset, không hợp lệ), mà tự chạy lại trên cùng BTXRD để so sánh có kiểm soát.

## Kết quả đầu ra
- Bảng so sánh accuracy / macro-F1 / precision-recall từng lớp giữa 4 phương pháp trên cùng test set.
- Đường cong accuracy/macro-F1 theo số shot (1/2/4/8/16) cho cả 2 nhánh LoRA, có error bar thể hiện độ lệch chuẩn qua 3 seed.
- Biểu đồ so sánh trực tiếp MedCLIP + LoRA vs CLIP gốc + LoRA trên cùng 1 trục — trả lời câu hỏi trung tâm của nhiệm vụ 4.
- Confusion matrix cho mỗi phương pháp.

Chi tiết cách chạy, cấu trúc code, và các điểm khác biệt có chủ đích so với CLIP-LoRA gốc — xem [README.md](README.md).
