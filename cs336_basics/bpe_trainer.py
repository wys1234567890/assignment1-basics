import json
import os
import resource
import time
from multiprocessing import Pool

import regex as re

try:
    # 包内导入（pytest / python -m 场景）
    from .MinValueStore import MinValueStore
    from .pretokenization_example import find_chunk_boundaries
except ImportError:
    # 直接运行 python cs336_basics/bpe_trainer.py 时没有包上下文，退回到脚本目录导入
    from MinValueStore import MinValueStore
    from pretokenization_example import find_chunk_boundaries

def train_bpe_tokenizer(
        input_path: str,
        vocab_size: int,
        special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a BPE tokenizer on the given input file.
    input_path: Path to the input text file.
    vocab_size: Desired size of the vocabulary.
    special_tokens: List of special tokens to include in the vocabulary.
    Returns a tuple of (vocab, merges) where vocab is a dictionary mapping
    token indices to byte strings, and merges is a list of tuples representing the merge operations.
    """
    #初始化vocab字典，将0-255的整数映射到对应的字节表示，并将特殊token添加到字典中
    vocab_dict = {i: bytes([i]) for i in range(256)}
    vocab_dict.update({i + 256: token.encode("utf-8") for i, token in enumerate(special_tokens)})
    vocab_set = set(vocab_dict.values())
    merges = []
    assert len(vocab_dict) <= vocab_size
    
    with open(input_path, "rb") as f:
        num_processes = 8
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        with Pool(processes=num_processes) as pool:
            chunks = []
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunks.append(f.read(end - start))
            # 每个worker在自己那块chunk里直接统计pretoken频率，
            # 只返回一个小dict，避免把几亿个pretoken原样pickle回父进程
            chunk_counts = pool.starmap(count_pretokens_in_chunk,
                                        [(chunk, special_tokens) for chunk in chunks])

        all_pretokens_freq_dict = {}
        for cc in chunk_counts:
            for pretoken, count in cc.items():
                all_pretokens_freq_dict[pretoken] = all_pretokens_freq_dict.get(pretoken, 0) + count
        all_pretokens_tokenized_dict, all_pairs_pretokens_dict = tokenize_pretokens(set(all_pretokens_freq_dict.keys()))
        all_pretokens_tokenized_freq_dict = {pretoken: (all_pretokens_tokenized_dict[pretoken], all_pretokens_freq_dict[pretoken]) for pretoken in all_pretokens_tokenized_dict.keys()}
        byte_pairs_freq_store = get_byte_pair_freq_store(all_pretokens_tokenized_freq_dict)
        
        while len(vocab_dict) < vocab_size:
            if len(byte_pairs_freq_store) == 0:
                break                    
            freq, token1, token2 = byte_pairs_freq_store.pop_min()
            new_token = token1 + token2
            vocab_dict[len(vocab_dict)] = new_token
            vocab_set.add(new_token)
            merges.append((token1, token2))
            update_tokenized_pretokens(all_pairs_pretokens_dict, all_pretokens_tokenized_dict, all_pretokens_freq_dict, byte_pairs_freq_store, (token1, token2))
            print(f"merge: {token1} + {token2} -> {new_token}, vocab size: {len(vocab_dict)}")
        
    return vocab_dict, merges
            

def pretokenize_file(chunk: bytes, special_tokens: list[str]):
    chunk_str = chunk.decode("utf-8")
    
    # Step 1: 用特殊标记分割文本，这样它们既是边界又不参与合并统计
    if special_tokens:
        # 转义特殊标记，因为可能包含正则特殊字符（如 |, *, . 等）
        escaped_tokens = [re.escape(token) for token in special_tokens]
        pattern = "|".join(escaped_tokens)
        # split 会保留分隔符，但我们需要在分隔符处切分，分隔符本身不参与预分词
        parts = re.split(pattern, chunk_str)
    else:
        parts = [chunk_str]
    
    # Step 2: 对每个部分分别进行预分词
    pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pretokens = []
    for part in parts:
        # 跳过空字符串（比如两个特殊标记相邻时）
        if part:
            tokens = re.findall(pat, part)
            pretokens.extend([token.encode("utf-8") for token in tokens])
    
    return pretokens


def count_pretokens_in_chunk(chunk: bytes, special_tokens: list[str]) -> dict[bytes, int]:
    """对chunk做预分词并直接统计每个pretoken的频率，返回{pretoken: count}。"""
    counts = {}
    for pretoken in pretokenize_file(chunk, special_tokens):
        counts[pretoken] = counts.get(pretoken, 0) + 1
    return counts


def compute_pretokens_frequencies(pretokens):
    """
    计算每个pretoken的频率，返回一个字典，键为pretoken，值为其出现的次数
    """
    pretokens_freq_dict = {}
    for pretoken in pretokens:
        pretokens_freq_dict[pretoken] = pretokens_freq_dict.get(pretoken, 0) + 1
    return pretokens_freq_dict


def tokenize_pretokens(pretokens: set) -> tuple[dict[bytes, list[bytes]], dict[tuple[bytes, bytes], set[bytes]]]:
    pretokens_tokenized_dict = {pretoken: [pretoken[i: i + 1] for i in range(len(pretoken))] for pretoken in pretokens}
    pairs_pretokens_dict = {}
    for pretoken, tokenized in pretokens_tokenized_dict.items():
        for i in range(len(tokenized) - 1):
            pair = (tokenized[i], tokenized[i + 1])
            pairs_pretokens_dict.setdefault(pair, set()).add(pretoken)

    return pretokens_tokenized_dict, pairs_pretokens_dict

def update_tokenized_pretokens(pairs_pretokens_dict: dict[tuple[bytes, bytes], set[bytes]],
                               pretokens_tokenized_dict: dict[bytes, list[bytes]], 
                               pretokens_freq_dict: dict[bytes, int], 
                               byte_pairs_freq_store: MinValueStore, 
                               merge: tuple[bytes, bytes]) -> None:
    """
    将指定的字节对合并为一个新token，并增量更新pair计数。

    只处理tokenization里包含相邻对 (t1, t2) 的pretoken（通过反向索引
    pairs_pretokens_dict 找到，而不是全表扫描）。

    store 中的值恒为 -count。对每个tokenization发生变化的pretoken，
    统计合并前后pair的差（旧-新，按pretoken频率加权），所有pretoken
    处理完后统一应用到store：仍在store中的pair更新为 -新count，
    计数归零的pair移除，新增pair添加。

    pretokens_tokenized_dict 被原地更新；反向索引与它同步维护。
    不变量：pairs_pretokens_dict[X] 恒等于"当前tokenization包含相邻对 X
    的所有pretoken"。
    """
    t1, t2 = merge
    new_token = t1 + t2
    delta = {}  # pair -> (旧count - 新count) 的加权和

    for pretoken in list(pairs_pretokens_dict.get(merge, ())):
        tokenized = pretokens_tokenized_dict[pretoken]
        new_tokenized = []
        i = 0
        while i < len(tokenized):
            if i < len(tokenized) - 1 and tokenized[i] == t1 and tokenized[i + 1] == t2:
                new_tokenized.append(new_token)
                i += 2
            else:
                new_tokenized.append(tokenized[i])
                i += 1
        pretokens_tokenized_dict[pretoken] = new_tokenized
        if new_tokenized == tokenized:
            continue  # 该pretoken不受影响，pair计数不变

        f = pretokens_freq_dict[pretoken]
        # 从旧相邻对对应的索引条目中摘除该pretoken
        for k in range(len(tokenized) - 1):
            pair = (tokenized[k], tokenized[k + 1])
            delta[pair] = delta.get(pair, 0) + f
            members = pairs_pretokens_dict.get(pair)
            if members is not None:
                members.discard(pretoken)
                if not members:
                    del pairs_pretokens_dict[pair]
        # 挂入新相邻对对应的索引条目
        for k in range(len(new_tokenized) - 1):
            pair = (new_tokenized[k], new_tokenized[k + 1])
            delta[pair] = delta.get(pair, 0) - f
            pairs_pretokens_dict.setdefault(pair, set()).add(pretoken)

    # 统一应用一次。新store值 = 旧store值 + (旧count - 新count) = -新count
    for pair, d in delta.items():
        if d == 0:
            continue
        if pair in byte_pairs_freq_store:
            new_value = byte_pairs_freq_store[pair] + d
            if new_value == 0:
                byte_pairs_freq_store.remove(pair[0], pair[1])
            else:
                byte_pairs_freq_store.update(new_value, pair[0], pair[1])
        else:
            new_count = -d
            if new_count > 0:
                byte_pairs_freq_store.add(-new_count, pair[0], pair[1])


def get_byte_pair_freq_store(pretokens_tokenized_freq_dict: dict[bytes, tuple[list[bytes], int]]) -> MinValueStore:
    """
    获取每个字节对的频率，返回一个MinValueStore对象
    """
    byte_pairs_freq_store = MinValueStore()
    byte_pairs_freq_dict = {}
    for pretoken, (tokenized, freq) in pretokens_tokenized_freq_dict.items():
        for i in range(len(tokenized) - 1):
            byte_pairs_freq_dict[(tokenized[i], tokenized[i + 1])] = byte_pairs_freq_dict.get((tokenized[i], tokenized[i + 1]), 0) + freq

    for pair, freq in byte_pairs_freq_dict.items():
        byte_pairs_freq_store.add(-freq, pair[0], pair[1])
    return byte_pairs_freq_store

def bytes_to_unicode() -> dict[int, str]:
    """
    GPT-2's bytes-to-unicode remapping: map every byte (0-255) to a single
    printable unicode character, so token bytestrings can be written to a
    plain-text file without ambiguity. In particular, the space byte b' '
    becomes 'Ġ', so serialized tokens never contain a literal space.
    """
    # These 188 bytes can be used as-is, since they are not whitespace or control characters.
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


def save_tokenizer_files(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    output_dir: str | os.PathLike,
    base_name: str,
) -> tuple[str, str]:
    """
    Serialize a trained BPE vocabulary and merges to disk in the same format as
    the assignment's reference files (and the format that Tokenizer.from_files
    should read back):

      - {base_name}_vocab.json : a JSON dict mapping token string -> token id
      - {base_name}_merges.txt  : one merge per line, "token1 token2", in order
                                  of creation

    Tokens are written with GPT-2's bytes->unicode remapping, so every byte
    (including space / control bytes) becomes a single printable character and
    each merges line splits unambiguously into exactly two fields on a single
    space. Special tokens such as <|endoftext|> are printable ASCII, so they
    are written literally.

    Returns the paths of the two files written.
    """
    byte_to_char = bytes_to_unicode()

    def token_to_str(token: bytes) -> str:
        return "".join(byte_to_char[b] for b in token)

    os.makedirs(output_dir, exist_ok=True)

    vocab_path = os.path.join(output_dir, f"{base_name}_vocab.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(
            {token_to_str(token): idx for idx, token in sorted(vocab.items())},
            f,
            ensure_ascii=False,
            indent=2,
        )

    merges_path = os.path.join(output_dir, f"{base_name}_merges.txt")
    with open(merges_path, "w", encoding="utf-8") as f:
        for token1, token2 in merges:
            f.write(f"{token_to_str(token1)} {token_to_str(token2)}\n")

    return vocab_path, merges_path


if __name__ == "__main__":
    input_path = r"./data/owt_train.txt"
    t0 = time.perf_counter()
    vocab, merges = train_bpe_tokenizer(input_path, 32000, ["<|endoftext|>"])
    print(f"{time.perf_counter() - t0:.1f}s")
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # Linux 单位是 KB
    print(f"{rss_kb / 1024 / 1024:.2f} GB")

    dir_name = os.path.dirname(input_path)
    base_name = os.path.splitext(os.path.basename(input_path))[0]  # 不含扩展名
    vocab_path, merges_path = save_tokenizer_files(vocab, merges, dir_name, base_name)
    print(f"vocab saved to {vocab_path}")
    print(f"merges saved to {merges_path}")
