"""
BIP-39 wordlist manager
لیست 2048 کلمه BIP-39 را مدیریت می‌کنه
"""

import os
import requests

BIP39_URL = (
    "https://raw.githubusercontent.com/trezor/python-mnemonic"
    "/master/src/mnemonic/wordlist/english.txt"
)
WORDLIST_PATH = os.path.join(os.path.dirname(__file__), "bip39_wordlist.txt")

# توکن‌های ویژه
PAD_TOKEN  = "<PAD>"   # index 0
BOS_TOKEN  = "<BOS>"   # index 1  (Beginning of Sequence)
EOS_TOKEN  = "<EOS>"   # index 2  (End of Sequence)
UNK_TOKEN  = "<UNK>"   # index 3

SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3


def download_wordlist() -> list[str]:
    """دانلود لیست کلمات BIP-39 از GitHub"""
    print("⬇️  Downloading BIP-39 wordlist...")
    r = requests.get(BIP39_URL, timeout=15)
    r.raise_for_status()
    words = [w.strip() for w in r.text.splitlines() if w.strip()]
    assert len(words) == 2048, f"Expected 2048 words, got {len(words)}"
    with open(WORDLIST_PATH, "w") as f:
        f.write("\n".join(words))
    print(f"✅ Saved {len(words)} words to {WORDLIST_PATH}")
    return words


def load_wordlist() -> list[str]:
    """بارگذاری لیست کلمات (دانلود اگه موجود نباشه)"""
    if os.path.exists(WORDLIST_PATH):
        with open(WORDLIST_PATH) as f:
            words = [w.strip() for w in f.readlines() if w.strip()]
        if len(words) == 2048:
            return words
    return download_wordlist()


class KeyTokenizer:
    """
    تبدیل seed phrase به ایندکس و برعکس

    مثال:
        tokenizer = KeyTokenizer()
        ids = tokenizer.encode("abandon ability able")
        phrase = tokenizer.decode(ids)
    """

    def __init__(self):
        words = load_wordlist()
        self.vocab = SPECIAL_TOKENS + words          # 4 + 2048 = 2052
        self.word2idx = {w: i for i, w in enumerate(self.vocab)}
        self.idx2word = {i: w for i, w in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)            # 2052

    def encode(self, phrase: str, add_bos_eos: bool = True) -> list[int]:
        """رشته کلمات → لیست ایندکس"""
        tokens = []
        if add_bos_eos:
            tokens.append(BOS_IDX)
        for word in phrase.strip().split():
            tokens.append(self.word2idx.get(word, UNK_IDX))
        if add_bos_eos:
            tokens.append(EOS_IDX)
        return tokens

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """لیست ایندکس → رشته کلمات"""
        skip = {PAD_IDX, BOS_IDX, EOS_IDX, UNK_IDX} if skip_special else set()
        return " ".join(
            self.idx2word[i] for i in ids
            if i in self.idx2word and i not in skip
        )


if __name__ == "__main__":
    tok = KeyTokenizer()
    print(f"Vocab size: {tok.vocab_size}")
    sample = "abandon ability able about above absent"
    ids = tok.encode(sample)
    back = tok.decode(ids)
    print(f"Original : {sample}")
    print(f"Encoded  : {ids}")
    print(f"Decoded  : {back}")
    assert back == sample, "❌ Round-trip failed!"
    print("✅ KeyTokenizer OK")
