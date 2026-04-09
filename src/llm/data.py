from .tensor import np


class Tokenizer:
    UNK = "\ufffd"  # replacement character for unknown chars

    def __init__(self, text: str):
        chars = sorted(list(set(text)))
        if self.UNK not in chars:
            chars.append(self.UNK)
            chars.sort()
        self.chars = chars
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        self._unk_id = self.stoi[self.UNK]

    def encode(self, s: str, verbose: bool = False) -> list[int]:
        if not verbose:
            return [self.stoi.get(ch, self._unk_id) for ch in s]
        encoded = []
        n = len(s)
        for i, ch in enumerate(s):
            if i % 1000 == 0:
                print(f"\rchar encode: {i + 1}/{n}", end="", flush=True)
            encoded.append(self.stoi.get(ch, self._unk_id))
        print()
        return encoded

    def decode(self, list: list[int]) -> str:
        return "".join(self.itos.get(i, self.UNK) for i in list)

    def to_dict(self) -> dict:
        return {"type": "char", "chars": self.chars}

    @classmethod
    def from_dict(cls, d: dict) -> "Tokenizer":
        tok = cls.__new__(cls)
        tok.chars = d["chars"]
        tok.vocab_size = len(tok.chars)
        tok.stoi = {ch: i for i, ch in enumerate(tok.chars)}
        tok.itos = {i: ch for i, ch in enumerate(tok.chars)}
        tok._unk_id = tok.stoi.get(cls.UNK, 0)
        return tok


class BPETokenizer:
    def __init__(self):
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.vocab_size = 256

    def train(self, text: str, vocab_size: int, verbose: bool = False):
        num_merges = vocab_size - 256
        tokens = list(text.encode("utf-8"))
        for i in range(num_merges):
            if verbose:
                print(f"\rbpe train: {i + 1}/{num_merges}", end="", flush=True)
            stats = {}
            for pair in zip(tokens, tokens[1:]):
                stats[pair] = stats.get(pair, 0) + 1
            if not stats:
                break
            pair = max(stats, key=stats.get)
            idx = 256 + i
            new_tokens = []
            j = 0
            while j < len(tokens):
                if j < len(tokens) - 1 and tokens[j] == pair[0] and tokens[j + 1] == pair[1]:
                    new_tokens.append(idx)
                    j += 2
                else:
                    new_tokens.append(tokens[j])
                    j += 1
            tokens, self.merges[pair] = new_tokens, idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
        if verbose:
            print()
        self.vocab_size = len(self.vocab)

    def encode(self, text: str, verbose: bool = False) -> list[int]:
        tokens = list(text.encode("utf-8"))
        if not self.merges:
            return tokens
        for i, ((p0, p1), idx) in enumerate(sorted(self.merges.items(), key=lambda x: x[1])):
            if verbose and i % 10 == 0:
                print(f"\rbpe encode: {i + 1}/{len(self.merges)}", end="", flush=True)
            new_tokens, j = [], 0
            while j < len(tokens):
                if j < len(tokens) - 1 and tokens[j] == p0 and tokens[j + 1] == p1:
                    new_tokens.append(idx)
                    j += 2
                else:
                    new_tokens.append(tokens[j])
                    j += 1
            tokens = new_tokens
        if verbose:
            print()
        return tokens

    def decode(self, tokens: list[int]) -> str:
        return b"".join(self.vocab[t] for t in tokens if t in self.vocab).decode(
            "utf-8", errors="replace"
        )

    def to_dict(self) -> dict:
        return {
            "type": "bpe",
            "merges": {f"{k[0]},{k[1]}": v for k, v in self.merges.items()},
            "vocab_size": self.vocab_size,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BPETokenizer":
        tok = cls()
        tok.merges = {tuple(map(int, k.split(","))): v for k, v in d["merges"].items()}
        tok.vocab_size, tok.vocab = d["vocab_size"], {i: bytes([i]) for i in range(256)}
        for (p0, p1), idx in sorted(tok.merges.items(), key=lambda x: x[1]):
            tok.vocab[idx] = tok.vocab[p0] + tok.vocab[p1]
        return tok

    def save(self, filename: str):
        import json

        with open(filename, "w") as f:
            json.dump(self.to_dict(), f)

    def load(self, filename: str):
        import json

        with open(filename) as f:
            data = json.load(f)
        restored = BPETokenizer.from_dict(data)
        self.merges, self.vocab_size, self.vocab = (
            restored.merges,
            restored.vocab_size,
            restored.vocab,
        )


def tokenizer_from_dict(d: dict):
    """Reconstruct a Tokenizer or BPETokenizer from its dict representation."""
    kind = d.get("type")
    if kind == "char":
        return Tokenizer.from_dict(d)
    if kind == "bpe":
        return BPETokenizer.from_dict(d)
    raise ValueError(f"unknown tokenizer type: {kind}")


class Dataset:
    def __init__(self, data, block_size: int):
        # Accept either a python list or an already-converted device array.
        if isinstance(data, np.ndarray):
            self.data = data.astype(np.uint32)
        else:
            self.data = np.array(data, dtype=np.uint32)
        self.block_size = block_size

    def __len__(self) -> int:
        return int(len(self.data))

    def get_batch(self, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        ix = np.random.randint(0, len(self.data) - self.block_size - 1, (batch_size,))
        x = np.stack([self.data[i : i + self.block_size] for i in ix])
        y = np.stack([self.data[i + 1 : i + 1 + self.block_size] for i in ix])
        return x, y

    def split(self, val_frac: float = 0.1) -> tuple["Dataset", "Dataset"]:
        """Return (train, val) datasets split by position (last `val_frac` is val)."""
        assert 0.0 < val_frac < 1.0
        n = len(self.data)
        cut = int(n * (1.0 - val_frac))
        return (
            Dataset(self.data[:cut], self.block_size),
            Dataset(self.data[cut:], self.block_size),
        )
