import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """
    均方根归一化层。
    """
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """
        初始化均方根归一化层
        """
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        if device is not None:
            try:
                self.device = torch.device(device)
            except Exception as e:
                raise ValueError(f"无效的设备格式: '{device}'。请使用 'cpu', 'cuda', 'cuda:0' 等格式。") from e
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "xpu" if torch.xpu.is_available() else "cpu")
        self.dtype = dtype if dtype is not None else torch.bfloat16
        self.weight = nn.Parameter(torch.ones(d_model, device=self.device, dtype=self.dtype))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数，执行均方根归一化。
        """
        in_dtype, in_device = x.dtype, x.device
        x = x.to(self.device, dtype=torch.float32)  # 使用float32进行计算以提高数值稳定性
        # 计算均方根
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # 归一化
        x_norm = x / rms
        # 缩放
        out = x_norm * self.weight
        return out.to(device=in_device, dtype=in_dtype)

    
        
