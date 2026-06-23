"""
معماری مدل Seq2Seq برای رمزگشایی هش → seed phrase
"""

import math
import torch
import torch.nn as nn
from model.tokenizer import HashTokenizer, MAX_HASH_LEN
from model.bip39 import KeyTokenizer, PAD_IDX, BOS_IDX, EOS_IDX

MAX_KEY_LEN = 26   # 24 کلمه + BOS + EOS

# ثابت‌های معماری (برای import از inference.py)
D_MODEL    = 128
NHEAD      = 4
ENC_LAYERS = 3
DEC_LAYERS = 3
DIM_FF     = 256


class PositionalEncoding(nn.Module):
    """Positional encoding استاندارد"""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class HashDecoder(nn.Module):
    """
    Transformer Encoder-Decoder

    ورودی  : هش base58/64 (توکن‌های کاراکتری)
    خروجی  : seed phrase (توکن‌های کلمه BIP-39)

    پارامترها:
        d_model      : بعد embedding اصلی
        nhead        : تعداد attention heads
        enc_layers   : تعداد لایه‌های encoder
        dec_layers   : تعداد لایه‌های decoder
        dim_ff       : بعد feedforward داخلی
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 128,
        nhead: int = 4,
        enc_layers: int = 3,
        dec_layers: int = 3,
        dim_ff: int = 256,
        dropout: float = 0.1,
        max_src_len: int = MAX_HASH_LEN,
        max_tgt_len: int = MAX_KEY_LEN,
    ):
        super().__init__()

        self.d_model = d_model
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

        # Embeddings
        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=0)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=PAD_IDX)

        # Positional encoding
        self.src_pe = PositionalEncoding(d_model, max_src_len + 10, dropout)
        self.tgt_pe = PositionalEncoding(d_model, max_tgt_len + 10, dropout)

        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=enc_layers,
            num_decoder_layers=dec_layers,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
        )

        # Output projection
        self.output_proj = nn.Linear(d_model, tgt_vocab_size)

        # Weight tying (embedding و output یه وزن دارن → کمتر overfit)
        self.output_proj.weight = self.tgt_embed.weight

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src: torch.Tensor, src_key_padding_mask=None):
        """فقط encoder — برای inference سریع‌تر"""
        x = self.src_pe(self.src_embed(src) * math.sqrt(self.d_model))
        return self.transformer.encoder(x, src_key_padding_mask=src_key_padding_mask)

    def decode_step(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None):
        """یک گام decoder"""
        y = self.tgt_pe(self.tgt_embed(tgt) * math.sqrt(self.d_model))
        out = self.transformer.decoder(
            y, memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        return self.output_proj(out)

    def forward(self, src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None):
        """Forward کامل برای training"""
        tgt_len = tgt.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            tgt_len, device=src.device
        )
        memory = self.encode(src, src_key_padding_mask)
        logits = self.decode_step(
            tgt, memory, tgt_mask, tgt_key_padding_mask
        )
        return logits

    @torch.no_grad()
    def generate(
        self,
        src: torch.Tensor,
        max_len: int = MAX_KEY_LEN,
        device: str = "cpu",
    ) -> list[int]:
        """
        Greedy decoding — یه هش → لیست ایندکس کلمه

        مثال:
            ids = model.generate(src_tensor)
        """
        self.eval()
        src = src.to(device)
        memory = self.encode(src)

        # شروع با BOS
        tgt = torch.tensor([[BOS_IDX]], dtype=torch.long, device=device)
        generated = []

        for _ in range(max_len):
            logits = self.decode_step(tgt, memory)
            next_token = logits[0, -1].argmax().item()

            if next_token == EOS_IDX:
                break
            generated.append(next_token)
            tgt = torch.cat(
                [tgt, torch.tensor([[next_token]], device=device)], dim=1
            )

        return generated

    @torch.no_grad()
    def batch_generate(
        self,
        src: torch.Tensor,
        max_len: int = MAX_KEY_LEN,
        device: str = "cpu",
    ) -> list[tuple[list[int], float]]:
        """
        Greedy decoding دسته‌ای — N هش با هم پردازش می‌شن (نه یکی‌یکی)
        سریع‌تر از تکرار generate() به ازای هر هش

        Returns:
            لیستی از (token_ids, confidence) برای هر نمونه در batch
            confidence = میانگین احتمال softmax توکن‌های انتخاب‌شده
        """
        self.eval()
        src = src.to(device)
        B = src.size(0)
        memory = self.encode(src)

        tgt = torch.full((B, 1), BOS_IDX, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        results: list[list[int]] = [[] for _ in range(B)]
        confidences: list[list[float]] = [[] for _ in range(B)]

        for _ in range(max_len):
            logits = self.decode_step(tgt, memory)          # (B, T, V)
            probs = torch.softmax(logits[:, -1], dim=-1)     # (B, V)
            conf, next_token = probs.max(dim=-1)              # (B,) , (B,)

            for i in range(B):
                if finished[i]:
                    continue
                if next_token[i].item() == EOS_IDX:
                    finished[i] = True
                else:
                    results[i].append(next_token[i].item())
                    confidences[i].append(conf[i].item())

            if finished.all():
                break

            tgt = torch.cat([tgt, next_token.unsqueeze(1)], dim=1)

        out = []
        for ids, confs in zip(results, confidences):
            avg_conf = sum(confs) / len(confs) if confs else 0.0
            out.append((ids, round(avg_conf, 4)))
        return out

    @torch.no_grad()
    def generate_constrained(
        self,
        src: torch.Tensor,
        target_words: int,
        device: str = "cpu",
    ) -> tuple[list[int], float]:
        """
        تولید با طول دقیق — مدل مجبور می‌شه دقیقاً target_words کلمه تولید کنه.
        EOS در هر گام پنهان می‌شه تا توقف زودهنگام نداشته باشیم.

        مثال:
            ids, conf = model.generate_constrained(src, target_words=12)
        """
        self.eval()
        src = src.to(device)
        memory = self.encode(src)

        tgt = torch.tensor([[BOS_IDX]], dtype=torch.long, device=device)
        generated: list[int] = []
        confidences: list[float] = []

        for _ in range(target_words):
            logits = self.decode_step(tgt, memory)          # (1, T, V)
            probs  = torch.softmax(logits[0, -1], dim=-1)   # (V,)

            # پنهان کردن EOS و PAD — نمی‌خوایم زود متوقف بشه
            probs_masked = probs.clone()
            probs_masked[EOS_IDX] = 0.0
            probs_masked[PAD_IDX] = 0.0
            total = probs_masked.sum()
            if total > 1e-9:
                probs_masked = probs_masked / total

            conf_val, next_tok = probs_masked.max(dim=-1)
            generated.append(next_tok.item())
            confidences.append(conf_val.item())

            tgt = torch.cat(
                [tgt, next_tok.unsqueeze(0).unsqueeze(0)], dim=1
            )

        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        return generated, round(avg_conf, 4)

    @torch.no_grad()
    def batch_generate_multimode(
        self,
        src: torch.Tensor,
        device: str = "cpu",
    ) -> list[list[dict]]:
        """
        اجرای هر ۳ حالت طول (12، 18، 24) برای همه نمونه‌های batch.
        Encoder فقط یک بار اجرا می‌شه — Decoder برای هر طول یک بار.

        Returns:
            list[B] از list[3] از {"words": int, "ids": list[int], "confidence": float}
            ترتیب داخلی: [نتیجه ۱۲ کلمه، نتیجه ۱۸ کلمه، نتیجه ۲۴ کلمه]
        """
        self.eval()
        src = src.to(device)
        B = src.size(0)
        memory = self.encode(src)   # (B, src_len, d_model) — یک بار محاسبه

        # ساختار خروجی: B نمونه × ۳ حالت
        all_results: list[list[dict | None]] = [[None, None, None] for _ in range(B)]

        for mode_idx, target_words in enumerate([12, 18, 24]):
            tgt = torch.full((B, 1), BOS_IDX, dtype=torch.long, device=device)
            generated:  list[list[int]]   = [[] for _ in range(B)]
            conf_lists: list[list[float]] = [[] for _ in range(B)]

            for _ in range(target_words):
                logits = self.decode_step(tgt, memory)          # (B, T, V)
                probs  = torch.softmax(logits[:, -1], dim=-1)   # (B, V)

                # پنهان کردن EOS و PAD در همه نمونه‌های batch
                probs[:, EOS_IDX] = 0.0
                probs[:, PAD_IDX] = 0.0
                row_sums = probs.sum(dim=-1, keepdim=True).clamp(min=1e-9)
                probs = probs / row_sums

                conf_vals, next_tokens = probs.max(dim=-1)      # (B,), (B,)

                for i in range(B):
                    generated[i].append(next_tokens[i].item())
                    conf_lists[i].append(conf_vals[i].item())

                tgt = torch.cat([tgt, next_tokens.unsqueeze(1)], dim=1)

            for i in range(B):
                avg_conf = (
                    sum(conf_lists[i]) / len(conf_lists[i])
                    if conf_lists[i] else 0.0
                )
                all_results[i][mode_idx] = {
                    "words":      target_words,
                    "ids":        generated[i],
                    "confidence": round(avg_conf, 4),
                }

        return all_results   # type: ignore[return-value]

    def count_params(self) -> str:
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return f"{n:,} ({n/1e6:.2f}M)"


def build_model(device="cpu") -> tuple["HashDecoder", HashTokenizer, KeyTokenizer]:
    """ساخت مدل + توکنایزرها"""
    hash_tok = HashTokenizer()
    key_tok = KeyTokenizer()

    model = HashDecoder(
        src_vocab_size=hash_tok.vocab_size,
        tgt_vocab_size=key_tok.vocab_size,
    ).to(device)

    print(f"✅ Model built | params: {model.count_params()}")
    return model, hash_tok, key_tok


if __name__ == "__main__":
    model, hash_tok, key_tok = build_model()
    # تست یه forward pass
    src = hash_tok.batch_encode(["xK9mP2qRs3nT4vU5a8b2c1d0e7f6"])
    tgt_ids = key_tok.encode("abandon ability able")
    tgt = torch.tensor([tgt_ids[:-1]], dtype=torch.long)  # بدون EOS
    out = model(src, tgt)
    print(f"Output shape: {out.shape}")  # (1, tgt_len, vocab_size)
    print("✅ Forward pass OK")
