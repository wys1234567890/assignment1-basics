import torch
import torch.nn as nn
import einops
from cs336_basics.transformer_block import TransformerBlock
from cs336_basics.embedding import Embedding
from cs336_basics.linear import Linear
from cs336_basics.rms_norm import RMSNorm

class TransformerLM(nn.Module):
    """
    自回归Transformer语言模型。
    """
    def __init__(
        self, 
        vocab_size: int, 
        context_size: int, 
        d_model: int, 
        num_layers: int, 
        num_heads: int, 
        d_ff: int,
        rope_theta=None, 
        device=None, 
        dtype=None):
        """
        初始化 TransformerLM 模型。
        参数：
            vocab_size: 词汇表大小。
            context_size: 上下文窗口大小（最大序列长度）。
            d_model: 模型的隐藏维度。
            num_layers: TransformerBlock 的层数。
            num_heads: 多头自注意力的头数。
            d_ff: 前馈网络的隐藏维度。
            rope_theta: 可选，RoPE 的缩放因子。
            device: 可选，指定模型运行的设备。
            dtype: 可选，指定模型参数的数据类型。
        """
        super().__init__()
        if device is not None:
            try:
                self.device = torch.device(device)
            except Exception as e:
                raise ValueError(f"无效的设备格式: '{device}'。请使用 'cpu', 'cuda', 'cuda:0' 等格式。") from e
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "xpu" if torch.xpu.is_available() else "cpu")
        if dtype is not None and not dtype.is_floating_point:
            # 参数必须是浮点类型：若传入整数 dtype（例如 token 索引的 dtype），
            # 回退到 float32，避免 nn.Parameter 报 "Only Tensors of floating point ..."
            dtype = torch.float32
        self.dtype = dtype if dtype is not None else torch.bfloat16

        self.vocab_size = vocab_size
        self.context_size = context_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta

        self.token_embeddings = Embedding(vocab_size, d_model, device=self.device, dtype=self.dtype)
        self.layers = nn.Sequential(*[
            TransformerBlock(d_model, num_heads, d_ff, rope_theta, context_size,
                             device=self.device, dtype=self.dtype)
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model, device=self.device, dtype=self.dtype)
        self.lm_head = Linear(d_model, vocab_size, device=self.device, dtype=self.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。
        参数：
            x: 输入张量，形状为 (batch_size, seq_len)，包含 token 的索引。
        返回：
            logits: 输出张量，形状为 (batch_size, seq_len, vocab_size)，表示每个位置的词汇表预测分数。
        """
        orig_dtype, orig_device = x.dtype, x.device
        # 输入是整数 token 索引：只迁移设备，不做 dtype 转换（嵌入层要求整数输入）
        if x.device != self.device:
            x = x.to(self.device)

        embedding_output = self.token_embeddings(x)
        transformer_output = self.layers(embedding_output)
        normed_output = self.ln_final(transformer_output)
        logits = self.lm_head(normed_output)

        logits = logits.to(device=orig_device)
        # 仅当原始输入是浮点类型时才恢复其 dtype；整数索引输入应返回浮点 logits
        if orig_dtype.is_floating_point:
            logits = logits.to(dtype=orig_dtype)
        return logits
        