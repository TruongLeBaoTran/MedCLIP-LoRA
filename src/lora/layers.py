"""
Vendored từ CLIP-LoRA (Zanella & Ben Ayed, "Low-Rank Few-Shot Adaptation of
Vision-Language Models", CVPRW 2024) — https://github.com/MaxZanella/CLIP-LoRA
File gốc: loralib/layers.py

Giữ 3 class:
- LoRALayer, LinearLoRA: generic, không phụ thuộc CLIP, áp dụng được cho bất
  kỳ nn.Linear nào (dùng cho MedCLIP — Swin/BERT có Q/K/V tách rời).
- PlainMultiheadAttentionLoRA: dành riêng cho nn.MultiheadAttention của CLIP
  gốc (dùng cho nhiệm vụ 4 — CLIP + LoRA đối chứng). MedCLIP không dùng tới
  class này vì không có nn.MultiheadAttention.
Bỏ các class CLIP-specific không dùng tới: Embedding, Conv1d/2d/3d, MergedLinear.

Giữ nguyên logic gốc, không sửa (trừ comment tiếng Việt thêm vào).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def set_param(curr_mod, name, param=None, mode="update"):
    r"""Refer to https://github.com/Baijiong-Lin/MOML/blob/main/MTL/utils.py"""
    if "." in name:
        n = name.split(".")
        module_name = n[0]
        rest = ".".join(n[1:])
        for name, mod in curr_mod.named_children():
            if module_name == name:
                return set_param(mod, rest, param, mode=mode)
    else:
        if mode == "update":
            delattr(curr_mod, name)
            setattr(curr_mod, name, param)
        elif mode == "get":
            if hasattr(curr_mod, name):
                p = getattr(curr_mod, name)
                return p


class LoRALayer:
    def __init__(self, r: int, lora_alpha: int, fan_in_fan_out: bool = False, dropout_rate: float = 0):
        self.r = r
        self.lora_alpha = lora_alpha
        self.dropout_rate = dropout_rate
        if self.r > 0:
            self.scaling = self.lora_alpha / math.sqrt(self.r)
        # Đánh dấu weight đang ở trạng thái CHƯA merge (A, B còn tách rời W)
        self.merged = False
        # Đặt True nếu layer cần thay có weight lưu dạng (fan_in, fan_out)
        self.fan_in_fan_out = fan_in_fan_out
        # các tham số cần LoRA {'param_name': 'lora_name'}
        self.params_with_lora = {}

    def register_lora_param(self):
        r"""Đăng ký ma trận LoRA A, B — A khởi tạo Kaiming, B khởi tạo 0
        (nên lúc mới gắn, B@A = 0, output y hệt model gốc chưa gắn LoRA)."""
        for param_name, lora_name in self.params_with_lora.items():
            assert len(eval(f"self.{param_name}").size()) == 2
            self.register_parameter(
                f"{lora_name}_lora_A",
                nn.Parameter(eval(f"self.{param_name}").new_zeros((self.r, eval(f"self.{param_name}").size()[1]))),
            )
            self.register_parameter(
                f"{lora_name}_lora_B",
                nn.Parameter(eval(f"self.{param_name}").new_zeros((eval(f"self.{param_name}").size()[0], self.r))),
            )
            eval(f"self.{param_name}").requires_grad = False

    def init_lora_param(self):
        for param_name, lora_name in self.params_with_lora.items():
            if hasattr(self, f"{lora_name}_lora_A"):
                nn.init.kaiming_uniform_(eval(f"self.{lora_name}_lora_A"), a=math.sqrt(5))
                nn.init.zeros_(eval(f"self.{lora_name}_lora_B"))

    def transpose(self, w: torch.Tensor):
        return w.transpose(0, 1) if self.fan_in_fan_out else w

    def merge_BA(self, param_name: str):
        lora_name = self.params_with_lora[param_name]
        return self.transpose(
            (eval(f"self.{lora_name}_lora_B") @ eval(f"self.{lora_name}_lora_A")).view(eval(f"self.{param_name}").shape)
        )

    def merge_lora_param(self):
        r"""p_new = p + scaling * B @ A, vẫn giữ được đạo hàm tới A, B."""
        for param_name, lora_name in self.params_with_lora.items():
            p = set_param(self, param_name, mode="get")
            p_new = p.detach() + self.merge_BA(param_name) * self.scaling
            set_param(self, param_name, param=p_new, mode="update")

    def add_lora_data(self):
        r"""KHÔNG differentiable — cộng thẳng vào .data."""
        for param_name, lora_name in self.params_with_lora.items():
            eval(f"self.{param_name}").data += self.merge_BA(param_name) * self.scaling

    def sub_lora_data(self):
        r"""KHÔNG differentiable — trừ thẳng khỏi .data."""
        for param_name, lora_name in self.params_with_lora.items():
            eval(f"self.{param_name}").data -= self.merge_BA(param_name) * self.scaling

    def lora_train(self, mode: bool = True):
        if mode:
            if self.merged and self.r > 0:
                self.sub_lora_data()
            self.merged = False
        else:
            if not self.merged and self.r > 0:
                self.add_lora_data()
            self.merged = True


class LinearLoRA(nn.Linear, LoRALayer):
    """Bọc 1 nn.Linear có sẵn thành bản có LoRA: h = Wx + scaling * B @ A @ x.
    Dùng được cho BẤT KỲ nn.Linear nào — đây chính là điểm mấu chốt để áp
    dụng LoRA lên MedCLIP (Swin attention qkv tách rời + BERT attention),
    khác với PlainMultiheadAttentionLoRA gốc chỉ hợp với nn.MultiheadAttention.
    """

    def __init__(
        self,
        existing_linear: nn.Linear,
        r: int = 0,
        lora_alpha: int = 1,
        fan_in_fan_out: bool = False,
        dropout_rate: float = 0.0,
        **kwargs,
    ):
        super().__init__(in_features=existing_linear.in_features, out_features=existing_linear.out_features)
        self.load_state_dict(existing_linear.state_dict())
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, fan_in_fan_out=fan_in_fan_out)

        self.params_with_lora = {"weight": "w"}
        if r > 0:
            self.register_lora_param()
        self.init_lora_param()
        self.weight.data = self.transpose(self.weight.data)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

    def train(self, mode: bool = True):
        super().train(mode)
        self.lora_train(mode)

    def forward(self, x: torch.Tensor, **kwargs):
        if self.dropout is None:
            if self.r > 0 and not self.merged:
                self.merge_lora_param()
                result = nn.Linear.forward(self, x, **kwargs)
                self.sub_lora_data()
                return result
            return nn.Linear.forward(self, x, **kwargs)

        original_output = nn.Linear.forward(self, x)
        if self.training and self.dropout.p > 0:
            x = self.dropout(x)

        if self.r > 0 and not self.merged:
            lora_adjustment = torch.matmul(x, self.merge_BA("weight").transpose(0, 1)) * self.scaling
            return original_output + lora_adjustment
        return original_output


class PlainMultiheadAttentionLoRA(nn.Module):
    """Bọc 1 nn.MultiheadAttention có sẵn (kiến trúc attention của CLIP gốc)
    thành bản có LoRA trên Q/K/V/O — tách in_proj_weight gộp của
    nn.MultiheadAttention thành 3 Linear q_proj/k_proj/v_proj riêng (copy đúng
    trọng số gốc), rồi bọc từng cái bằng LinearLoRA. Chỉ dùng cho nhiệm vụ 4
    (CLIP gốc + LoRA) — MedCLIP không có nn.MultiheadAttention nên không cần
    class này (xem models/lora_medclip.py).
    """

    def __init__(
        self,
        existing_mha: nn.MultiheadAttention,
        enable_lora: list = ["q", "k", "v", "o"],
        r: int = 0,
        lora_alpha: int = 1,
        dropout_rate: float = 0.0,
        **kwargs,
    ):
        super().__init__()

        self.dropout = 0  # module này không dùng để huấn luyện lại block gốc
        self.embed_dim = existing_mha.embed_dim
        self.kdim = existing_mha.kdim
        self.vdim = existing_mha.vdim
        self._qkv_same_embed_dim = existing_mha._qkv_same_embed_dim
        self.num_heads = existing_mha.num_heads
        self.batch_first = existing_mha.batch_first
        self.head_dim = existing_mha.head_dim
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=existing_mha.in_proj_bias is not None)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=existing_mha.in_proj_bias is not None)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=existing_mha.in_proj_bias is not None)
        self.proj = nn.Linear(self.embed_dim, self.embed_dim, bias=existing_mha.out_proj.bias is not None)

        # Tách in_proj_weight gộp (3*embed_dim x embed_dim) thành 3 Linear riêng
        with torch.no_grad():
            existing_weight = existing_mha.in_proj_weight.data
            existing_bias = existing_mha.in_proj_bias.data if existing_mha.in_proj_bias is not None else None

            self.q_proj.weight.data.copy_(existing_weight[: self.embed_dim, :])
            if existing_bias is not None:
                self.q_proj.bias.data.copy_(existing_bias[: self.embed_dim])

            self.k_proj.weight.data.copy_(existing_weight[self.embed_dim : 2 * self.embed_dim, :])
            if existing_bias is not None:
                self.k_proj.bias.data.copy_(existing_bias[self.embed_dim : 2 * self.embed_dim])

            self.v_proj.weight.data.copy_(existing_weight[2 * self.embed_dim :, :])
            if existing_bias is not None:
                self.v_proj.bias.data.copy_(existing_bias[2 * self.embed_dim :])

            self.proj.weight.data.copy_(existing_mha.out_proj.weight.data)
            if self.proj.bias is not None:
                self.proj.bias.data.copy_(existing_mha.out_proj.bias.data)

        self.scaled_dot_product_attention = F.scaled_dot_product_attention

        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, dropout_rate=dropout_rate)

        for item in enable_lora:
            if item == "q":
                self.q_proj = LinearLoRA(self.q_proj, r=r, lora_alpha=lora_alpha, fan_in_fan_out=False, dropout_rate=dropout_rate)
            elif item == "k":
                self.k_proj = LinearLoRA(self.k_proj, r=r, lora_alpha=lora_alpha, fan_in_fan_out=False, dropout_rate=dropout_rate)
            elif item == "v":
                self.v_proj = LinearLoRA(self.v_proj, r=r, lora_alpha=lora_alpha, fan_in_fan_out=False, dropout_rate=dropout_rate)
            elif item == "o":
                self.proj = LinearLoRA(self.proj, r=r, lora_alpha=lora_alpha, fan_in_fan_out=False, dropout_rate=dropout_rate)

    def forward_module(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        need_weights=True,
        attn_mask=None,
        average_attn_weights=True,
        is_causal=False,
    ):
        if attn_mask is not None and is_causal:
            raise AssertionError("Only allow causal mask or attn_mask")
        is_batched = query.dim() == 3
        key_padding_mask = F._canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=F._none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=query.dtype,
        )

        if self.batch_first and is_batched:
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = [x.transpose(1, 0) for x in (query, key)]
                    value = key
            else:
                query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        tgt_len, bsz, embed_dim = query.shape
        src_len, _, _ = key.shape

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=F._none_or_dtype(key_padding_mask),
            other_name="key_padding_mask",
            target_type=q.dtype,
            check_other=False,
        )

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                correct_2d_size = (tgt_len, src_len)
                if attn_mask.shape != correct_2d_size:
                    raise RuntimeError(
                        f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}.")
                attn_mask = attn_mask.unsqueeze(0)
            elif attn_mask.dim() == 3:
                correct_3d_size = (bsz * self.num_heads, tgt_len, src_len)
                if attn_mask.shape != correct_3d_size:
                    raise RuntimeError(
                        f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}.")
            else:
                raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")

        if attn_mask is not None:
            if attn_mask.size(0) == 1 and attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(0)
            else:
                attn_mask = attn_mask.view(bsz, self.num_heads, -1, src_len)

        dropout_p = self.dropout if self.training else 0.0

        q = q.view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        src_len = k.size(1)
        q = q.view(bsz, self.num_heads, tgt_len, self.head_dim)
        k = k.view(bsz, self.num_heads, src_len, self.head_dim)
        v = v.view(bsz, self.num_heads, src_len, self.head_dim)

        attn_output = self.scaled_dot_product_attention(q, k, v, attn_mask, dropout_p, is_causal)
        attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(bsz * tgt_len, embed_dim)
        attn_output = self.proj(attn_output)
        attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))
        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), None
        return attn_output, None

    def train(self, mode: bool = True):
        super().train(mode)
        # Không gọi self.lora_train(mode) — merge/sub theo mode chỉ áp dụng cho
        # đường "không dropout" của LinearLoRA; ở đây luôn dùng đường có dropout
        # (forward tính adjustment trực tiếp, không merge vào .data).

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, **kwargs):
        return self.forward_module(query, key, value, **kwargs)
