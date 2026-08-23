import torch
import torch.nn as nn
from cs336_basics.linear import Linear

class PositionwiseFeedForward(nn.Module):
    """
    使用SwiGLU的位置前馈网络层。
    """
    def __init__(self, d_in: int, d_ff: int = None, device=None, dtype=None):
        """
        初始化位置前馈网络层。
        """
        super().__init__()
        self.d_in = d_in
        if d_ff is None:
            d_hidden = 8 * d_in / 3
            self.d_hidden = round(d_hidden / 64) * 64  # 将隐藏层维度调整为64的倍数
        else:
            self.d_hidden = d_ff
        if device is not None:
            try:
                self.device = torch.device(device)
            except Exception as e:
                raise ValueError(f"无效的设备格式: '{device}'。请使用 'cpu', 'cuda', 'cuda:0' 等格式。") from e
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "xpu" if torch.xpu.is_available() else "cpu")
        self.dtype = dtype if dtype is not None else torch.bfloat16
        self.linear1 = Linear(d_in, self.d_hidden, device=self.device, dtype=self.dtype)
        self.linear2 = Linear(self.d_hidden, d_in, device=self.device, dtype=self.dtype)
        self.gate = Linear(d_in, self.d_hidden, device=self.device, dtype=self.dtype)  # 用于SwiGLU激活的线性层

    def silu(self, x: torch.Tensor) -> torch.Tensor:
        """
        SiLU激活函数。
        """
        return x * torch.sigmoid(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数，执行位置前馈网络计算。
        """
        orig_dtype, orig_device = x.dtype, x.device
        if x.device != self.device or x.dtype != self.dtype:
            x = x.to(self.device, dtype=self.dtype)

        out = self.linear1(x)
        out = self.silu(out) * self.gate(x)
        out = self.linear2(out)
        return out.to(device=orig_device, dtype=orig_dtype)
        

        