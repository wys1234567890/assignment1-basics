import torch

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    计算输入张量 x 在指定维度 dim 上的 softmax。
    """
    max_vals, _ = torch.max(x, dim=dim, keepdim=True)
    exp_x = torch.exp(x - max_vals)
    sum_exp_x = torch.sum(exp_x, dim=dim, keepdim=True)
    return exp_x / sum_exp_x