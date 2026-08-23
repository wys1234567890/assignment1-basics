import torch
import torch.nn as nn
from cs336_basics.linear import Linear
from torch import cos, sin
import einops

class RotaryPositionalEmbedding(nn.Module):
    """
    RoPE旋转位置编码层。
    """
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        初始化RoPE旋转位置编码层。
        """
        super().__init__()
        assert d_k % 2 == 0, "RoPE requires d_k to be even"
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        if device is not None:
            try:
                self.device = torch.device(device)
            except Exception as e:
                raise ValueError(f"无效的设备格式: '{device}'。请使用 'cpu', 'cuda', 'cuda:0' 等格式。") from e
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "xpu" if torch.xpu.is_available() else "cpu")

        # ---- 一次性向量化构造 cos/sin 缓存（替代逐元素嵌套循环）----
        half_idx = torch.arange(0, d_k, 2, device=self.device, dtype=torch.float32)  # 0,2,...,d_k-2
        inv_freq = 1.0 / (theta ** (half_idx / d_k))              # (d_k/2,)
        positions = torch.arange(max_seq_len, device=self.device, dtype=torch.float32)
        freqs = positions.unsqueeze(1) * inv_freq.unsqueeze(0)    # (max_seq_len, d_k/2)

        self.register_buffer("cos_cache", torch.cos(freqs), persistent=False)
        self.register_buffer("sin_cache", torch.sin(freqs), persistent=False)


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数，应用RoPE旋转位置编码。
        """
        orig_dtype, orig_device = x.dtype, x.device

        seq_len = x.size(-2)  # 假设输入形状为 (..., seq_len, d_k)
        assert seq_len <= self.max_seq_len, f"输入序列长度 {seq_len} 超过了最大序列长度 {self.max_seq_len}"

        if token_positions.device != self.cos_cache.device:
            token_positions = token_positions.to(self.cos_cache.device)
        cos = self.cos_cache[token_positions]     # (..., seq_len, d_k/2)
        sin = self.sin_cache[token_positions]     # (..., seq_len, d_k/2)   
        if cos.device != x.device:
            cos, sin = cos.to(x.device), sin.to(x.device)
 
        # 在 x 所在设备上按需转 fp32（不做跨设备搬运）；reshape 是视图，不复制
        xr = x.to(dtype=torch.float32) if x.dtype != torch.float32 else x
        xr = xr.reshape(*x.shape[:-1], self.d_k // 2, 2)   # interleaved 配对 (2i, 2i+1)

        # 四个逐元素乘加完成旋转：O(d) 运算、无 matmul、无 Python 循环
        out = torch.empty_like(xr)
        out[..., 0] = xr[..., 0] * cos - xr[..., 1] * sin
        out[..., 1] = xr[..., 0] * sin + xr[..., 1] * cos

        # 纯函数式输出：不改输入、转回原 dtype/设备
        return out.reshape(x.shape).to(dtype=orig_dtype, device=orig_device)