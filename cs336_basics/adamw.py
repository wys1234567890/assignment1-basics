import torch
import torch.nn as nn

class AdamW(torch.optim.Optimizer):
    """
    AdamW 优化器实现
    """
    def __init__(self, params, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01, lr=1e-3):
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not eps >= 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not weight_decay >= 0.0:
            raise ValueError(f"Invalid weight decay value: {weight_decay}")

        defaults = dict(betas=betas, eps=eps, weight_decay=weight_decay, lr=lr)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    grad = grad.to_dense()  # 将稀疏梯度转换为密集梯度
                if grad.dtype != torch.float32:
                    grad = grad.to(torch.float32)  # 将 float16 梯度转换为 float32
                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # 第一，二矩估计的指数移动平均值
                    state['m'] = torch.zeros_like(p.data, dtype=torch.float32)
                    state['v'] = torch.zeros_like(p.data, dtype=torch.float32)

                m, v = state['m'], state['v']
                state['step'] += 1
                t = state['step']

                # 第一、二阶矩的指数移动平均（有偏估计，存回 state）
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * (grad * grad)
                state['m'], state['v'] = m, v

                # 偏差修正，仅用于本轮更新
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)

                # 梯度更新 + 解耦权重衰减
                p.data -= group['lr'] * (m_hat / (torch.sqrt(v_hat) + group['eps']))
                p.data -= group['lr'] * group['weight_decay'] * p.data

        return loss        
    