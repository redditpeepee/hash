"""
توکنایزر کاراکتر-محور برای هش‌های base58/base64
"""

# Base58 alphabet (Bitcoin style)
BASE58_CHARS = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
# Base64 alphabet
BASE64_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="
# ترکیب هر دو (کاراکترهای ممکن در هش)
ALL_CHARS = sorted(set(BASE58_CHARS + BASE64_CHARS))

PAD_CHAR = "<PAD>"
UNK_CHAR = "<UNK>"

HASH_LEN     = 34   # آدرس‌های تران همیشه دقیقاً ۳۴ کاراکترند
MAX_HASH_LEN = 40   # حداکثر طول برای padding (کمی بیشتر از HASH_LEN)


class HashTokenizer:
    """
    تبدیل رشته هش به ایندکس و برعکس

    مثال:
        tok = HashTokenizer()
        ids = tok.encode("xK9mP2qR")
        restored = tok.decode(ids)
    """

    def __init__(self):
        special = [PAD_CHAR, UNK_CHAR]
        self.vocab = special + ALL_CHARS
        self.char2idx = {c: i for i, c in enumerate(self.vocab)}
        self.idx2char = {i: c for i, c in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)
        self.pad_idx = self.char2idx[PAD_CHAR]
        self.unk_idx = self.char2idx[UNK_CHAR]

    def encode(self, hash_str: str, pad_to: int = MAX_HASH_LEN) -> list[int]:
        """هش → لیست ایندکس (با padding)"""
        ids = [self.char2idx.get(c, self.unk_idx) for c in hash_str[:pad_to]]
        # padding تا طول ثابت
        ids += [self.pad_idx] * (pad_to - len(ids))
        return ids

    def decode(self, ids: list[int]) -> str:
        """لیست ایندکس → رشته هش"""
        return "".join(
            self.idx2char[i]
            for i in ids
            if i in self.idx2char and self.idx2char[i] not in (PAD_CHAR, UNK_CHAR)
        )

    def batch_encode(self, hashes: list[str], pad_to: int = MAX_HASH_LEN):
        """چند هش با هم"""
        import torch
        return torch.tensor([self.encode(h, pad_to) for h in hashes], dtype=torch.long)


if __name__ == "__main__":
    tok = HashTokenizer()
    print(f"Vocab size : {tok.vocab_size}")
    print(f"Max hash   : {MAX_HASH_LEN}")

    for h in ["xK9mP2qRs3nT", "AB12cd", "1" * 34]:
        ids = tok.encode(h)
        back = tok.decode(ids)
        print(f"  '{h}' → len={len(ids)} → '{back}'")
    print("✅ HashTokenizer OK")
