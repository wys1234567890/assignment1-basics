import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None) -> None:
        """
        嵌入层初始化，默认bf16精度，支持CPU、GPU和XPU设备。
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if device is not None:
            try:
                self.device = torch.device(device)
            except Exception as e:
                raise ValueError(f"无效的设备格式: '{device}'。请使用 'cpu', 'cuda', 'cuda:0' 等格式。") from e
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "xpu" if torch.xpu.is_available() else "cpu")
        self.dtype = dtype if dtype is not None else torch.bfloat16
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=self.device, dtype=self.dtype))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数，执行嵌入查找。
        """
        assert not torch.is_floating_point(x), "输入张量必须为整数类型"
        # 高级索引：一次 gather 完成查表，任意维度的整数输入都适用
        out = self.weight[x.to(self.device)]
        return out.to(x.device)
