import einops
import torch
import torch.nn as nn
from cs336_basics.rotary_positional_embedding import RotaryPositionalEmbedding
from cs336_basics.functions import scaled_dot_product_attention

class MultiheadSelfAttention(nn.Module):
    """
    多头自注意力层。
    """
    def __init__(self, d_model: int, num_heads: int, theta=None, max_seq_len=None,device=None, dtype=None):
        """
        初始化多头自注意力层。
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
        
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False, device=self.device, dtype=self.dtype)
        self.k_proj = nn.Linear(d_model, d_model, bias=False, device=self.device, dtype=self.dtype)
        self.v_proj = nn.Linear(d_model, d_model, bias=False, device=self.device, dtype=self.dtype)
        self.output_proj = nn.Linear(d_model, d_model, bias=False, device=self.device, dtype=self.dtype)
        self.rope = None
        if theta is not None:
            # rope独立作用于每个头
            self.rope = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len, device=self.device)
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.output_proj.weight)


    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, token_positions: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播函数，执行多头自注意力计算。
        """
        orig_dtype, orig_device = x.dtype, x.device
        if x.device != self.device or x.dtype != self.dtype:
            x = x.to(self.device, dtype=self.dtype)

        seq_len = x.shape[-2]
        causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
        mask = causal if mask is None else (causal & mask) 

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = einops.rearrange(q, "... seq (heads d_k) -> ... heads seq d_k", heads=self.num_heads)
        k = einops.rearrange(k, "... seq (heads d_k) -> ... heads seq d_k", heads=self.num_heads)
        v = einops.rearrange(v, "... seq (heads d_k) -> ... heads seq d_k", heads=self.num_heads)

        # rope：切头后一次调用，同时作用于所有头；v 不转
        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        output = scaled_dot_product_attention(q, k, v, mask)

        # 合并头 + 输出投影
        output = einops.rearrange(output, "... heads seq d_k -> ... seq (heads d_k)")
        output = self.output_proj(output)
        return output.to(device=orig_device, dtype=orig_dtype)
        

