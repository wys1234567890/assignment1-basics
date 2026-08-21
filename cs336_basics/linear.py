import torch
import torch.nn as nn
import math
import einops

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None) -> None:
        """
        线性层初始化，无偏置，默认bf16精度，支持CPU、GPU和XPU设备。
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        if device is not None:
            try:
                self.device = torch.device(device)
            except Exception as e:
                raise ValueError(f"无效的设备格式: '{device}'。请使用 'cpu', 'cuda', 'cuda:0' 等格式。") from e
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "xpu" if torch.xpu.is_available() else "cpu")
        self.dtype = dtype if dtype is not None else torch.bfloat16
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=self.device, dtype=self.dtype))
        std = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数，执行线性变换。
        """
        orig_dtype, orig_device = x.dtype, x.device
        if x.device != self.device or x.dtype != self.dtype:
            x = x.to(self.device, dtype=self.dtype)

        out = einops.einsum(x, self.weight, "... in_features, out_features in_features -> ... out_features")
        return out.to(device=orig_device, dtype=orig_dtype)
