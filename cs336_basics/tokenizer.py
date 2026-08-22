import json
from typing import Iterable
import regex as re


def _unicode_to_byte() -> dict[str, int]:
    """返回一个字典，将Unicode字符映射到字节值。"""
    byte_values = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    char_values = byte_values[:]
    next_char = 0
    for byte_value in range(256):
        if byte_value not in byte_values:
            byte_values.append(byte_value)
            char_values.append(256 + next_char)
            next_char += 1
    return {chr(char_value): byte_value for byte_value, char_value in zip(byte_values, char_values)}


class Tokenizer:
    """
    BPE分词器
    """
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] = None):
        # 约定：vocab 为 {token_id: token_bytes}，与 train_bpe_tokenizer 的返回一致。
        self.vocab = vocab
        self.token_to_id = {token_bytes: token_id for token_id, token_bytes in vocab.items()}
        self.merges = merges
        self.merges_rank_dict = {merge: rank for rank, merge in enumerate(merges)}
        self.special_tokens = special_tokens if special_tokens is not None else []

    @classmethod
    def from_files(cls,
                   vocab_filepath: str,
                   merges_filepath: str,
                   special_tokens: list[str] = None) -> "Tokenizer":
        """
        从文件中加载vocab和merges，创建Tokenizer实例
        """
        unicode_to_byte = _unicode_to_byte()

        def deserialize_token(token: str) -> bytes:
            try:
                return bytes(unicode_to_byte[char] for char in token)
            except KeyError:
                # Special tokens are stored literally rather than byte-remapped.
                return token.encode("utf-8")

        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            serialized_vocab = json.load(f)
        vocab = {
            token_id: deserialize_token(token)
            for token, token_id in serialized_vocab.items()
        }
        if special_tokens:
            existing_tokens = set(vocab.values())
            for special_token in special_tokens:
                special_bytes = special_token.encode("utf-8")
                if special_bytes not in existing_tokens:
                    vocab[len(vocab)] = special_bytes

        merges = []
        with open(merges_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                stripped_line = line.rstrip("\n")
                if stripped_line:
                    t1, t2 = stripped_line.split(" ")
                    merges.append((deserialize_token(t1), deserialize_token(t2)))

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """
        将文本编码为token列表
        """
        pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        parts = []
        if self.special_tokens:
            pattern = f"({'|'.join(re.escape(token) for token in sorted(self.special_tokens, key=len, reverse=True))})"
            parts.extend(re.split(pattern, text))
        else:
            parts = [text]

        pretokenized_parts = []
        for part in parts:
            if part in self.special_tokens:
                pretokenized_parts.append(part)
            else:
                pretokenized_parts.extend(re.findall(pat, part))

        encode_token_ids = []
        
        for part in pretokenized_parts:
            if part in self.special_tokens:
                encode_token_ids.append(self.token_to_id[part.encode("utf-8")])
            else:
                byte_part = part.encode("utf-8")
                encode_token_ids.extend(self.bpe_encode(byte_part))
        return encode_token_ids

    def bpe_encode(self, byte_part: bytes) -> list[int]:
        """
        对字节序列进行BPE编码，返回token id列表
        """
        token_ids = []
        tokenized_tokens = [byte_part[i:i + 1] for i in range(len(byte_part))]
        assert len(self.vocab) > 256 and len(self.merges) > 0, "vocab和merges必须在初始化时加载"

        while True:
            min_rank = len(self.merges) + 1
            min_rank_index = -1
            for i in range(len(tokenized_tokens) - 1):
                pair = (tokenized_tokens[i], tokenized_tokens[i + 1])
                if pair in self.merges_rank_dict:
                    rank = self.merges_rank_dict[pair]
                    if rank < min_rank:
                        min_rank = rank
                        min_rank_index = i

            if(min_rank_index == -1):
                break
            tokenized_tokens[min_rank_index:min_rank_index + 2] = [tokenized_tokens[min_rank_index] + tokenized_tokens[min_rank_index + 1]]

        for token in tokenized_tokens:
            if token in self.token_to_id:
                token_ids.append(self.token_to_id[token])
            else:
                raise ValueError(f"Token {token} not found in vocab.")
        return token_ids

              
    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[list[int]]:
        """
        将可迭代对象中的每个文本编码为token列表
        """
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        """
        将token id列表解码为文本
        """
        tokens = [self.vocab[token_id] for token_id in ids]
        byte_string = b''.join(tokens)
        return byte_string.decode("utf-8", errors="replace")
    