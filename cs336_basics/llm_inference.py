import argparse
import os
import time

import torch

from cs336_basics import tokenizer
from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.functions import load_checkpoint, softmax


def parse_args():
    p = argparse.ArgumentParser(description="Load a trained LLM from a checkpoint.")
    p.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint (.pt).")
    p.add_argument(
        "--vocab_path", type=str, default="../data/TinyStoriesV2-GPT4-train_vocab.json", help="Path to vocab json"
    )
    p.add_argument(
        "--merges_path", type=str, default="../data/TinyStoriesV2-GPT4-train_merges.txt", help="Path to merges txt"
    )
    # 模型结构：默认值与 train_llm.py 的默认值一致；若 checkpoint 里存了 config，则会覆盖这些值。
    p.add_argument("--vocab_size", type=int, default=None, help="None 时取 len(tokenizer.vocab)")
    p.add_argument("--context_size", type=int, default=512)
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--num_layers", type=int, default=8)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--d_ff", type=int, default=2048)
    p.add_argument("--rope_theta", type=float, default=None)
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float32"])
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--temperature", type=float, default=1.0, help="采样温度，越高越随机。")
    p.add_argument("--top_k", type=int, default=0, help="Top-k 采样，0 表示不使用 top-k。")
    p.add_argument(
        "--eos_token", type=str, default=None,
        help="结束 token 的文本（如 '<|endoftext|>'）。None 时自动从词表探测（含 endoftext/endofthetext 的 token）。",
    )
    return p.parse_args()


def read_checkpoint_config(model_path: str) -> dict | None:
    """若 checkpoint 里保存了模型结构 config，则读取；否则返回 None。"""
    try:
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    return ckpt.get("config") if isinstance(ckpt, dict) else None

def find_eos_id(tokenizer: Tokenizer, eos_text: str | None = None) -> int | None:
    """
    在词表中查找「结束 token」的 id。
    参数：
        eos_text: 指定的结束 token 文本（如 '<|endoftext|>'）。为 None 时自动探测：
                  优先匹配解码后含 'endoftext' 或 'endofthetext' 的 token。
    返回：
        int | None：结束 token 的 id；找不到则返回 None。
    """
    if eos_text:
        for tid, b in tokenizer.vocab.items():
            try:
                if b.decode("utf-8", errors="replace") == eos_text:
                    return tid
            except Exception:
                continue
        return None

    # 自动探测：常见 GPT 风格结束 token
    for tid, b in tokenizer.vocab.items():
        try:
            s = b.decode("utf-8", errors="replace").lower()
        except Exception:
            continue
        if "endoftext" in s or "endofthetext" in s:
            return tid
    return None


def temperature_softmax(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    对 logits 应用温度缩放的 softmax。
    参数：
        logits: 形状为 (batch_size, vocab_size) 的张量。
        temperature: 温度参数，越高越随机。temperature=1.0 时为标准 softmax。
    返回：
        形状为 (batch_size, vocab_size) 的概率分布张量。
    """
    if temperature <= 0:
        raise ValueError("temperature 必须大于 0。")
    scaled_logits = logits / temperature
    return softmax(scaled_logits, dim=-1)

def top_k_sampling(probs: torch.Tensor, top_k: int) -> torch.Tensor:
    """
    对概率分布应用 top-k 采样。
    参数：
        probs: 形状为 (batch_size, vocab_size) 的概率分布张量。
        top_k: 保留的最大概率的前 k 个 token，其他 token 的概率设为 0。
    返回：
        形状为 (batch_size, vocab_size) 的概率分布张量，已应用 top-k 采样。
    """
    if top_k <= 0:
        return probs  # 不使用 top-k，直接返回原始概率分布
    top_k = min(top_k, probs.size(-1))
    # 获取每个样本的 top-k 索引和对应的概率
    topk_probs, topk_indices = torch.topk(probs, k=top_k, dim=-1)
    # 创建一个全零的张量，并将 top-k 的概率放回对应位置
    new_probs = torch.zeros_like(probs)
    new_probs.scatter_(-1, topk_indices, topk_probs)
    # 归一化，使得概率和为 1
    new_probs /= new_probs.sum(dim=-1, keepdim=True)
    return new_probs

def generate_reply(user_input: str, tokenizer: Tokenizer, model: TransformerLM) -> str:
    """
    根据用户输入生成回复。
    参数：
        user_input: 用户输入的字符串。
        tokenizer: 用于编码和解码的 Tokenizer 实例。
        model: 用于生成回复的 TransformerLM 实例。
    返回：
        生成的回复字符串。
    """
    params = parse_args()
    # 将用户输入编码为 token 索引
    input_tokens = tokenizer.encode(user_input)
    input_tensor = torch.tensor(input_tokens, dtype=torch.int64, device=params.device).unsqueeze(0)  # (1, seq_len)
    max_length = model.context_size  # 最大生成长度为模型的上下文长度
    eos_id = find_eos_id(tokenizer, getattr(params, "eos_token", None))
    eos_text = tokenizer.vocab[eos_id].decode("utf-8", errors="replace") if eos_id is not None else None
    if eos_text is not None:
        print(f"[eos] 结束标记 {eos_text!r}（生成到它即停止输出）")

    generated_ids: list[int] = []
    reply = ""
    with torch.no_grad():
        while input_tensor.size(1) < max_length:
            logits = model(input_tensor)  # (1, seq_len, vocab_size)
            last_logits = logits[0, -1, :]  # 取最后一个 token 的 logits，形状为 (vocab_size,)

            # 应用温度缩放的 softmax
            probs = temperature_softmax(last_logits.unsqueeze(0), params.temperature)  # (1, vocab_size)

            # 应用 top-k 采样
            probs = top_k_sampling(probs, params.top_k)  # (1, vocab_size)

            # 从概率分布中采样下一个 token
            next_token_id = torch.multinomial(probs.squeeze(0), num_samples=1).item()

            # 情况一：模型以单个结束 token id（如 256）输出 -> 立即停止
            if eos_id is not None and next_token_id == eos_id:
                break

            # 将采样得到的 token 添加到输入序列中
            input_tensor = torch.cat([input_tensor, torch.tensor([[next_token_id]], dtype=torch.int64, device=params.device)], dim=1)
            generated_ids.append(next_token_id)

            # 情况二（通用兜底）：无论结束标记是单个 token 还是被拆成多个 token（旧/新编码），
            # 只要「已生成文本」里出现了结束标记文本，就截断到它之前并停止。
            if eos_text is not None:
                cur_text = tokenizer.decode(generated_ids)
                if eos_text in cur_text:
                    reply = cur_text.split(eos_text, 1)[0]
                    break

    if reply == "":
        reply = tokenizer.decode(generated_ids)
        # 循环因达到长度上限而结束时，再兜底截断一次
        if eos_text is not None and eos_text in reply:
            reply = reply.split(eos_text, 1)[0]
    return reply

def main():
    params = parse_args()
    tokenizer = Tokenizer.from_files(params.vocab_path, params.merges_path, special_tokens=["<|endoftext|>"])

    # 1) 确定模型结构：优先用 checkpoint 内保存的 config，否则用命令行/默认值
    config = read_checkpoint_config(params.model_path) or {}
    vocab_size = config.get("vocab_size") or params.vocab_size or len(tokenizer.vocab)
    context_size = config.get("context_size", params.context_size)
    d_model = config.get("d_model", params.d_model)
    num_layers = config.get("num_layers", params.num_layers)
    num_heads = config.get("num_heads", params.num_heads)
    d_ff = config.get("d_ff", params.d_ff)
    rope_theta = config.get("rope_theta", params.rope_theta)
    dtype = torch.bfloat16 if config.get("dtype", params.dtype) == "bfloat16" else torch.float32

    print("模型结构:")
    for k, v in dict(
        vocab_size=vocab_size, context_size=context_size, d_model=d_model, num_layers=num_layers,
        num_heads=num_heads, d_ff=d_ff, rope_theta=rope_theta, dtype=str(dtype),
    ).items():
        print(f"  {k}: {v}")

    # 2) 构建模型
    model = TransformerLM(
        vocab_size=vocab_size,
        context_size=context_size,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
        device=params.device,
        dtype=dtype,
    ).to(params.device)

    # 3) 从 checkpoint 加载权重（推理不需要优化器，传 None）
    iteration = load_checkpoint(params.model_path, model, optimizer=None)
    model.eval()

    print(f"\n模型权重已从 {params.model_path} 加载成功。")
    print(f"checkpoint 里的训练迭代数: {iteration}")

    chat(tokenizer, model)

def chat(tokenizer: Tokenizer, model: TransformerLM):
    print("🤖 Bot: What can I help you with? (Type 'exit' to quit)")
    print("-" * 50)
    
    while True:
        user_input = input("👤 User: ").strip()
        
        # 退出判断
        if user_input.lower() in ['exit', 'quit', 'q']:
            print("🤖 Bot: Goodbye! 👋")
            break
        
        # 空输入处理
        if not user_input:
            print("🤖 Bot: Please say something...")
            continue
        
        # 模拟生成回复（延迟输出）
        print("🤖 Bot: ", end="", flush=True)
        reply = generate_reply(user_input, tokenizer, model)
        
        # 逐字打印（模拟打字效果）
        for char in reply:
            print(char, end="", flush=True)
            time.sleep(0.05)  # 每字间隔0.05秒
        print("\n")  # 回复结束后换行


if __name__ == "__main__":
    main()