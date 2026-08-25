import torch
import torch.nn as nn
from cs336_basics.multihead_self_attention import MultiheadSelfAttention
from cs336_basics.positionwise_feed_forward import PositionwiseFeedForward
from cs336_basics.rms_norm import RMSNorm

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, theta=None, max_seq_len=None, device=None, dtype=None):
        """
        初始化 TransformerBlock，包括多头自注意力层和前馈网络。
        """
        super().__init__()
        if device is not None:
            try:
                self.device = torch.device(device)
            except Exception as e:
                raise ValueError(f"无效的设备格式: '{device}'。请使用 'cpu', 'cuda', 'cuda:0' 等格式。") from e
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "xpu" if torch.xpu.is_available() else "cpu")
        self.dtype = dtype if dtype is not None else torch.bfloat16

        self.dmodel = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.theta = theta
        self.max_seq_len = max_seq_len

        self.attn = MultiheadSelfAttention(d_model, num_heads, theta, max_seq_len, device=self.device, dtype=self.dtype)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, device=self.device, dtype=self.dtype)
        self.ln1 = RMSNorm(d_model, device=self.device, dtype=self.dtype)
        self.ln2 = RMSNorm(d_model, device=self.device, dtype=self.dtype)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, token_positions: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播函数，执行 TransformerBlock 的计算。
        """
        orig_dtype, orig_device = x.dtype, x.device
        if x.device != self.device or x.dtype != self.dtype:
            x = x.to(self.device, dtype=self.dtype)

        # pre-norm：残差 + 子层（先归一化再进子层，输出加回残差）
        x = x + self.attn(self.ln1(x), mask=mask, token_positions=token_positions)
        x = x + self.ffn(self.ln2(x))

        return x.to(device=orig_device, dtype=orig_dtype)