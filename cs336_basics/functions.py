import math
import torch
import einops

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    计算输入张量 x 在指定维度 dim 上的 softmax。
    """
    max_vals, _ = torch.max(x, dim=dim, keepdim=True)
    exp_x = torch.exp(x - max_vals)
    sum_exp_x = torch.sum(exp_x, dim=dim, keepdim=True)
    return exp_x / sum_exp_x

def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """
    计算缩放点积注意力：output = softmax(QK^T / sqrt(d_k)) @ V

    参数形状（batch 维用省略号 ... 表示，q/k/v 的 batch 维需可互相广播）：
        q:    (..., seq_len_q, d_k)
        k:    (..., seq_len_k, d_k)
        v:    (..., seq_len_k, d_v)
        mask: 可选布尔张量，形状可广播到 (..., seq_len_q, seq_len_k)；
              True 表示允许注意力，False 表示遮蔽（对应位置置为 -inf）。
    返回：
        (..., seq_len_q, d_v)
    """
    assert q.shape[-1] == k.shape[-1], f"d_k 不匹配: q 的最后一维 {q.shape[-1]} != k 的最后一维 {k.shape[-1]}"
    assert q.shape[-1] > 0, f"d_k 必须为正数，实际为 {q.shape[-1]}"
    assert k.shape[-2] == v.shape[-2], f"seq_len_k 不匹配: k 的倒数第二维 {k.shape[-2]} != v 的倒数第二维 {v.shape[-2]}"
    assert q.dtype == k.dtype == v.dtype, f"q/k/v 的 dtype 必须一致，实际为 q={q.dtype}, k={k.dtype}, v={v.dtype}"
    assert q.device == k.device == v.device, f"q/k/v 必须在同一 device，实际为 q={q.device}, k={k.device}, v={v.device}"
    # batch 维必须可广播（不可广播时 broadcast_shapes 会抛出 RuntimeError）
    torch.broadcast_shapes(q.shape[:-2], k.shape[:-2], v.shape[:-2])
    if mask is not None:
        assert mask.dtype == torch.bool, f"mask 必须是布尔张量，实际 dtype 为 {mask.dtype}"
        assert mask.device == q.device, f"mask 与 q 必须在同一 device，实际为 mask={mask.device}, q={q.device}"
        qk_shape = torch.broadcast_shapes(q.shape[:-2], k.shape[:-2]) + (q.shape[-2], k.shape[-2])
        torch.broadcast_shapes(mask.shape, qk_shape)  # mask 必须可广播到 qk 形状

    # 用 Python 浮点数做缩放：torch.tensor(q.shape[-1]) 会创建 CPU 张量，
    # 在 GPU 上除法直接报 device 不匹配；Python 标量则既无 device 也无 dtype 转换问题
    scale = math.sqrt(q.shape[-1])
    qk = einops.einsum(q, k, "... seq_len_q d_k, ... seq_len_k d_k -> ... seq_len_q seq_len_k") / scale
    if mask is not None:
        qk = qk.masked_fill(~mask, float("-inf"))
    softmax_qk = softmax(qk, dim=-1)
    output = einops.einsum(softmax_qk, v, "... seq_len_q seq_len_k, ... seq_len_k d_v -> ... seq_len_q d_v")
    return output

def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    计算交叉熵损失。
    参数：
        logits: 形状为 (batch_size, ..., num_classes) 的未归一化预测值。
        targets: 形状为 (batch_size, ...) 的整数标签，取值范围为 [0, num_classes-1]。
    返回：
        标量张量，表示平均交叉熵损失。
    """
    assert logits.device == targets.device, f"logits 和 targets 必须在同一 device，实际为 logits={logits.device}, targets={targets.device}"
    assert logits.shape[:-1] == targets.shape, f"logits 的前 n-1 维必须与 targets 形状匹配，实际为 logits={logits.shape}, targets={targets.shape}"
    assert targets.dtype in (torch.int32, torch.int64), f"targets 必须是整数类型，实际 dtype 为 {targets.dtype}"
    orig_dtype = logits.dtype
    logits = logits.to(torch.float32)  # 为了数值稳定性，先转为 float32

    logits_subtract = logits - logits.max(dim=-1, keepdim=True).values  # 减去最大值，避免 exp 溢出
    log_sum = torch.log(torch.sum(torch.exp(logits_subtract), dim=-1, keepdim=True))
    negative_log_loss = -logits_subtract + log_sum
    all_loss =  torch.gather(negative_log_loss, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)  # 选择正确类别的损失
    loss = torch.mean(all_loss)
    return loss.to(orig_dtype)



