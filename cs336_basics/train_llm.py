import os
import argparse

import numpy as np
import torch

from cs336_basics.tokenizer import Tokenizer
from cs336_basics.functions import (
    data_loader,
    cross_entropy_loss,
    save_checkpoint,
    load_checkpoint,
    cosine_schedule_with_warmup,
    gradient_clipping,
)
from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.adamw import AdamW


def betas_type(s: str):
    a, b = s.split(",")
    return (float(a), float(b))


def parse_args():
    p = argparse.ArgumentParser(description="Train an LLM on TinyStories")

    # data paths
    p.add_argument("--train_name", type=str, default="train_4080_1", help="Name of this run (used as save dir)")
    p.add_argument("--save_path", type=str, default="../models", help="Root dir to save the run under")
    p.add_argument(
        "--vocab_path", type=str, default="../data/TinyStoriesV2-GPT4-train_vocab.json", help="Path to vocab json"
    )
    p.add_argument(
        "--merges_path", type=str, default="../data/TinyStoriesV2-GPT4-train_merges.txt", help="Path to merges txt"
    )
    p.add_argument(
        "--train_data_path", type=str, default="../data/TinyStoriesV2-GPT4-train.txt", help="Path to training text"
    )
    p.add_argument(
        "--valid_data_path", type=str, default="../data/TinyStoriesV2-GPT4-valid.txt", help="Path to validation text"
    )
    p.add_argument("--force_tokenize", action="store_true", help="Re-tokenize even if a cached .bin already exists")

    # model
    p.add_argument("--vocab_size", type=int, default=None, help="If None, use len(tokenizer.vocab)")
    p.add_argument("--context_size", type=int, default=256, help="Context window length (max sequence length)")
    p.add_argument("--d_model", type=int, default=512, help="Hidden dimension")
    p.add_argument("--num_layers", type=int, default=4, help="Number of transformer layers")
    p.add_argument("--num_heads", type=int, default=16, help="Number of attention heads")
    p.add_argument("--d_ff", type=int, default=1344, help="Feed-forward hidden dimension")
    p.add_argument("--rope_theta", type=float, default=10000, help="RoPE theta (optional)")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float32"], help="Parameter dtype")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")

    # training
    p.add_argument("--batch_size", type=int, default=128, help="Micro-batch size per optimizer step")
    p.add_argument("--num_steps", type=int, default=8000, help="Total number of optimizer steps")
    p.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation steps (effective batch = batch*grad_accum)")
    p.add_argument("--learning_rate", type=float, default=6e-4, help="Peak learning rate")
    p.add_argument("--min_lr", type=float, default=6e-5, help="Floor of the cosine schedule")
    p.add_argument("--warmup_steps", type=int, default=1000, help="Linear warmup steps")
    p.add_argument("--grad_clip", type=float, default=1.0, help="Global gradient-norm clip value")
    p.add_argument("--weight_decay", type=float, default=0.1, help="AdamW weight decay")
    p.add_argument("--betas", type=betas_type, default=(0.9, 0.95), help="AdamW betas, e.g. '0.9,0.95'")
    p.add_argument("--eps", type=float, default=1e-8, help="AdamW epsilon")

    # logging / checkpointing
    p.add_argument("--eval_interval", type=int, default=250, help="Log train loss + run eval every N steps")
    p.add_argument("--eval_batches", type=int, default=50, help="Number of fixed windows used for val loss")
    p.add_argument("--save_interval", type=int, default=500, help="Save a checkpoint every N steps")
    p.add_argument("--resume_from", type=str, default=None, help="Path to a checkpoint .pt to resume from")

    return p.parse_args()


def tokenize_to_bin(src_path: str, bin_path: str, tokenizer: Tokenizer, flush_size: int = 1_000_000) -> int:
    """
    流式 BPE 编码：逐行读取文本，攒够 flush_size 个 token 就落盘一次（uint16 裸数组）。
    这样不会同时把整份 2.2GB 文本和全量 Python list 常驻内存。
    """
    total = 0
    buf: list[int] = []
    with open(src_path, "r", encoding="utf-8") as fin, open(bin_path, "wb") as fout:
        for line in fin:
            ids = tokenizer.encode(line.rstrip("\n"))
            if not ids:
                continue
            buf.extend(ids)
            while len(buf) >= flush_size:
                np.array(buf[:flush_size], dtype=np.uint16).tofile(fout)
                buf = buf[flush_size:]
                total += flush_size
        if buf:
            np.array(buf, dtype=np.uint16).tofile(fout)
            total += len(buf)
    return total


def get_token_array(txt_path: str, bin_path: str, tokenizer: Tokenizer, force: bool = False):
    """返回 (memmap uint16 数组, token 总数)。已缓存且未 force 时直接复用 .bin。"""
    if os.path.exists(bin_path) and not force:
        total = os.path.getsize(bin_path) // 2  # uint16 -> 每 token 2 字节
        print(f"[data] 复用已缓存 token 文件 {bin_path}（{total:,} tokens）", flush=True)
    else:
        total = tokenize_to_bin(txt_path, bin_path, tokenizer)
        print(f"[data] 分词完成 {txt_path} -> {bin_path}（{total:,} tokens）", flush=True)
    # 用 'c'（copy-on-write）映射：保持 mmap 按需读写的好处，同时避免 torch.from_numpy
    # 对只读数组发出的 "not writable" 警告；训练只读不改写文件本身。
    return np.memmap(bin_path, dtype=np.uint16, mode="c"), total


@torch.no_grad()
def evaluate(
    dataset, model, batch_size: int, context_length: int, device, num_batches: int = 50, seed: int = 0
) -> float:
    """在固定的随机窗口上计算平均验证损失（seed 固定保证可复现）。"""
    model.eval()
    device = torch.device(device)
    gen = torch.Generator().manual_seed(seed)
    num_possible = len(dataset) - context_length
    if num_possible <= 0:
        model.train()
        return float("nan")

    total, n = 0.0, 0
    for _ in range(num_batches):
        ix = torch.randint(0, num_possible, (batch_size,), generator=gen).tolist()
        x = torch.stack([torch.from_numpy(dataset[i : i + context_length]) for i in ix]).long().to(device)
        y = torch.stack([torch.from_numpy(dataset[i + 1 : i + context_length + 1]) for i in ix]).long().to(device)
        total += cross_entropy_loss(model(x), y).item()
        n += 1
    model.train()
    return total / max(n, 1)


def plot_losses(save_dir: str, train_records, val_records):
    """把训练/验证 loss 画成一张双曲线图（如果 matplotlib 可用），否则回退导出 CSV。"""
    records_csv = os.path.join(save_dir, "loss_history.csv")
    with open(records_csv, "w", encoding="utf-8") as f:
        f.write("step,train_loss,val_loss\n")
        tr = dict(train_records)
        vr = dict(val_records)
        for step in sorted(set(tr) | set(vr)):
            f.write(f"{step},{tr.get(step, '')},{vr.get(step, '')}\n")
    print(f"[plot] 损失记录已保存到 {records_csv}", flush=True)

    try:
        os.environ.setdefault("MPLCONFIGDIR", os.path.join(save_dir, ".mplconfig"))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # matplotlib 未安装等
        print(f"[plot] matplotlib 不可用，跳过画图（{e}）。可用 `uv pip install matplotlib` 安装。", flush=True)
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    if train_records:
        ax.plot(*zip(*train_records), label="train loss", color="tab:blue")
    if val_records:
        ax.plot(*zip(*val_records), label="val loss", color="tab:orange", marker="o", linestyle="--")
    ax.set_xlabel("step")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title("Training vs Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    out = os.path.join(save_dir, "loss_curve.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot] 损失曲线已保存到 {out}", flush=True)


def main():
    args = parse_args()
    print("Training configuration:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    torch.manual_seed(0)
    np.random.seed(0)

    # --- run dirs ---
    save_dir = os.path.join(args.save_path, args.train_name)
    os.makedirs(save_dir, exist_ok=True)

    # --- tokenizer + vocab size ---
    # 必须把 '<|endoftext|>' 声明为特殊 token：否则 encode() 会把它拆成 8 个 token，
    # 模型就学不到「单个 256 = 结束」。声明后 encode('<|endoftext|>') 会得到 [256]。
    tokenizer = Tokenizer.from_files(args.vocab_path, args.merges_path, special_tokens=["<|endoftext|>"])
    if args.vocab_size is None:
        args.vocab_size = len(tokenizer.vocab)
    print(f"[data] vocab_size = {args.vocab_size}", flush=True)

    # --- token arrays (memmap'd uint16) ---
    train_bin = os.path.join(save_dir, "train_tokens.bin")
    valid_bin = os.path.join(save_dir, "valid_tokens.bin")
    train_dataset, train_total = get_token_array(args.train_data_path, train_bin, tokenizer, args.force_tokenize)
    valid_dataset, _ = get_token_array(args.valid_data_path, valid_bin, tokenizer, args.force_tokenize)

    # --- model + optimizer ---
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_size=args.context_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=args.device,
        dtype=dtype,
    ).to(args.device)

    # 把模型结构配置一并存进 checkpoint，推理时（llm_inference）可据此自动重建模型。
    model_config = dict(
        vocab_size=args.vocab_size,
        context_size=args.context_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        dtype=args.dtype,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=args.betas,
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    # --- resume ---
    start_step = 0
    if args.resume_from and os.path.exists(args.resume_from):
        start_step = load_checkpoint(args.resume_from, model, optimizer)
        print(f"[resume] 从 {args.resume_from} 恢复，从 step {start_step} 继续", flush=True)
    else:
        print("[resume] 未指定有效 --resume_from，从零开始训练", flush=True)

    tokens_per_step = args.batch_size * args.context_size * args.grad_accum
    print(
        f"[train] 每步 {tokens_per_step:,} tokens，共 {args.num_steps} 步 "
        f"≈ {args.num_steps * tokens_per_step / 1e6:.1f}M tokens",
        flush=True,
    )

    # --- training loop ---
    train_records: list[tuple[int, float]] = []
    val_records: list[tuple[int, float]] = []
    running_loss, running_count = 0.0, 0
    model.train()

    for step in range(start_step, args.num_steps):
        # 1) LR schedule (cosine with warmup), applied to every param group
        lr = cosine_schedule_with_warmup(step, args.warmup_steps, args.num_steps, args.min_lr, args.learning_rate)
        for g in optimizer.param_groups:
            g["lr"] = lr

        # 2) forward/backward with gradient accumulation
        optimizer.zero_grad()
        log_train_loss = 0.0
        for _ in range(args.grad_accum):
            x, y = data_loader(train_dataset, args.batch_size, args.context_size, device=args.device)
            loss = cross_entropy_loss(model(x), y) / args.grad_accum
            log_train_loss += loss.item()
            loss.backward()
        if args.grad_clip:
            gradient_clipping(model.parameters(), args.grad_clip)
        optimizer.step()

        running_loss += log_train_loss
        running_count += 1

        # 3) periodic logging + validation
        if (step + 1) % args.eval_interval == 0:
            avg_train = running_loss / max(running_count, 1)
            val_loss = evaluate(
                valid_dataset, model, args.batch_size, args.context_size,
                args.device, num_batches=args.eval_batches,
            )
            train_records.append((step + 1, avg_train))
            val_records.append((step + 1, val_loss))
            print(
                f"[step {step + 1}/{args.num_steps}] train_loss={avg_train:.4f} "
                f"val_loss={val_loss:.4f} lr={lr:.2e}",
                flush=True,
            )
            running_loss, running_count = 0.0, 0

        # 4) periodic checkpoint
        if (step + 1) % args.save_interval == 0:
            save_checkpoint(model, optimizer, step + 1, os.path.join(save_dir, f"checkpoint_step_{step + 1}.pt"), config=model_config)

    # --- final save + plot ---
    save_checkpoint(model, optimizer, args.num_steps, os.path.join(save_dir, "checkpoint_final.pt"), config=model_config)
    plot_losses(save_dir, train_records, val_records)
    print(f"[done] 训练完成。checkpoint 与损失图保存在 {save_dir}", flush=True)


if __name__ == "__main__":
    main()
