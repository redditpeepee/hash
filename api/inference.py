"""
Pipeline رمزگشایی: Lookup اول، ML اگه پیدا نشد
+ حالت compare برای مانیتورینگ کیفیت مدل
+ caching و batch inference واقعی برای سرعت بالاتر
"""

import os
import re
import time
import threading
from collections import OrderedDict
from typing import Optional

import torch
import duckdb

from model.tokenizer import HashTokenizer, HASH_LEN
from model.bip39     import KeyTokenizer
from model.seq2seq   import HashDecoder, D_MODEL, NHEAD, ENC_LAYERS, DEC_LAYERS, DIM_FF


# ── نرمال‌سازی هش ورودی ─────────────────────────────────────────────
# آدرس‌های تران: T + 33 کاراکتر base58 = دقیقاً 34 کاراکتر
_BASE58_CHARS = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
_TRON_RE = re.compile(r"T[" + re.escape(_BASE58_CHARS) + r"]{33}")


def normalize_hash(raw: str) -> str:
    """
    استخراج و نرمال‌سازی آدرس تران (۳۴ کاراکتر ثابت) از هر ورودی.

    ورودی‌های پشتیبانی‌شده:
      - آدرس خام:      TXWbH5Nx6j4YPTZaFyDtQaLGcQJKEoizG2
      - با فاصله:      "  TXWbH5Nx6j4YPTZaFyDtQaLGcQJKEoizG2  "
      - با پیشوند URL: https://tronscan.org/#/address/TXWb...
      - با برچسب:      "آدرس: TXWbH5Nx6j4YPTZaFyDtQaLGcQJKEoizG2"
      - با خط جدید:    "TXWbH5Nx6j4YPTZaFyDtQaLGcQJKEoizG2\\n"
    """
    raw = raw.strip()
    # جستجوی الگوی دقیق آدرس تران در متن ورودی
    match = _TRON_RE.search(raw)
    if match:
        return match.group(0)
    # Fallback: اگه دقیقاً 34 کاراکتر بود، مستقیم قبول کن
    if len(raw) == HASH_LEN:
        return raw
    raise ValueError(
        f"آدرس تران معتبر (۳۴ کاراکتر) پیدا نشد — "
        f"ورودی دریافتی: '{raw[:60]}{'...' if len(raw) > 60 else ''}'"
    )


# ── LRU Cache ساده برای lookup های پرتکرار ───────────────────────────

class LRUCache:
    """Cache ساده در حافظه — برای هش‌هایی که زیاد پرسیده می‌شن"""

    def __init__(self, capacity: int = 100_000):
        self.capacity = capacity
        self.data: "OrderedDict[str, str]" = OrderedDict()
        self.lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[str]:
        with self.lock:
            if key in self.data:
                self.data.move_to_end(key)
                self.hits += 1
                return self.data[key]
            self.misses += 1
            return None

    def put(self, key: str, value: str):
        with self.lock:
            self.data[key] = value
            self.data.move_to_end(key)
            if len(self.data) > self.capacity:
                self.data.popitem(last=False)

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "size":     len(self.data),
            "hits":     self.hits,
            "misses":   self.misses,
            "hit_rate": round(self.hits / total, 4) if total else 0.0,
        }


# ── شمارنده آمار سرویس ────────────────────────────────────────────

class Stats:
    """شمارنده‌های ساده برای مانیتورینگ — بدون نیاز به دیتابیس خارجی"""

    def __init__(self):
        self.lock = threading.Lock()
        self.total_requests   = 0
        self.by_source         = {"lookup": 0, "ml": 0, "not_found": 0}
        self.compare_total     = 0
        self.compare_agree     = 0     # هش‌هایی که هم در DB و هم ML قابل حل بودن و نتیجه یکی بود
        self.compare_disagree  = 0

    def record(self, source: str):
        with self.lock:
            self.total_requests += 1
            self.by_source[source] = self.by_source.get(source, 0) + 1

    def record_compare(self, agree: bool):
        with self.lock:
            self.compare_total += 1
            if agree:
                self.compare_agree += 1
            else:
                self.compare_disagree += 1

    def snapshot(self) -> dict:
        with self.lock:
            agree_rate = (
                round(self.compare_agree / self.compare_total, 4)
                if self.compare_total else None
            )
            return {
                "total_requests": self.total_requests,
                "by_source":      dict(self.by_source),
                "compare": {
                    "total":      self.compare_total,
                    "agree":      self.compare_agree,
                    "disagree":   self.compare_disagree,
                    "agree_rate": agree_rate,
                },
            }


class DecoderPipeline:
    """
    دو لایه رمزگشایی + حالت مقایسه:
      1. Lookup در DuckDB (+ LRU cache)  → سریع، دقت 100%
      2. مدل ML (batch واقعی)            → برای هش‌های ندیده‌شده
      3. compare()                       → هر دو نتیجه با هم، برای QA مدل

    مثال:
        pipeline = DecoderPipeline("data/hashes.db", "model/checkpoints/best_model.pt")
        result = pipeline.decode("xK9mP2qRs3nT4vU5")
        both   = pipeline.compare("xK9mP2qRs3nT4vU5")
    """

    def __init__(
        self,
        db_path:       Optional[str] = None,
        model_path:    Optional[str] = None,
        device:        str           = "cpu",
        cache_size:    int           = 100_000,
    ):
        self.device     = device
        self.db_path    = db_path
        self.model_path = model_path
        self._con       = None
        self._model     = None
        self._hash_tok  = None
        self._key_tok   = None

        self.cache = LRUCache(capacity=cache_size)
        self.stats = Stats()

        self._load_db(db_path)
        self._load_model(model_path)

    # ── بارگذاری ───────────────────────────────────────────────────

    def _load_db(self, path: Optional[str]):
        if path and os.path.exists(path):
            self._con = duckdb.connect(path, read_only=True)
            count = self._con.execute("SELECT COUNT(*) FROM pairs").fetchone()[0]
            print(f"✅ DB loaded: {count:,} rows  ({path})")
        else:
            print("⚠️  No DB — lookup disabled")

    def _load_model(self, path: Optional[str]):
        if not path or not os.path.exists(path):
            print("⚠️  No model — ML fallback disabled")
            return

        self._hash_tok = HashTokenizer()
        self._key_tok  = KeyTokenizer()

        self._model = HashDecoder(
            src_vocab_size = self._hash_tok.vocab_size,
            tgt_vocab_size = self._key_tok.vocab_size,
            d_model    = D_MODEL,
            nhead      = NHEAD,
            enc_layers = ENC_LAYERS,
            dec_layers = DEC_LAYERS,
            dim_ff     = DIM_FF,
        ).to(self.device)

        ckpt = torch.load(path, map_location=self.device)
        self._model.load_state_dict(ckpt["model_state"])
        self._model.eval()
        print(f"✅ Model loaded: {path}")

    # ── lookup (با cache) ──────────────────────────────────────────

    def _lookup(self, hash_str: str) -> Optional[str]:
        cached = self.cache.get(hash_str)
        if cached is not None:
            return cached

        if not self._con:
            return None

        row = self._con.execute(
            "SELECT key FROM pairs WHERE hash = ?", [hash_str]
        ).fetchone()

        if row:
            self.cache.put(hash_str, row[0])
            return row[0]
        return None

    # ── ML inference (تکی) ───────────────────────────────────────────

    def _ml_decode(self, hash_str: str) -> tuple[Optional[str], float]:
        if not self._model:
            return None, 0.0
        src = self._hash_tok.batch_encode([hash_str]).to(self.device)
        (ids, conf), = self._model.batch_generate(src)
        key = self._key_tok.decode(ids) if ids else None
        return key, conf

    # ── ML inference (دسته‌ای واقعی — نه loop) ───────────────────────

    def _ml_decode_batch(self, hashes: list[str]) -> list[tuple[Optional[str], float]]:
        if not self._model or not hashes:
            return [(None, 0.0)] * len(hashes)
        src = self._hash_tok.batch_encode(hashes).to(self.device)
        pairs = self._model.batch_generate(src)
        return [
            (self._key_tok.decode(ids) if ids else None, conf)
            for ids, conf in pairs
        ]

    # ── public API ─────────────────────────────────────────────────

    # ── ML inference (همه حالت‌های طول) ──────────────────────────────

    def _ml_multi_decode(self, hash_str: str) -> list[dict]:
        """
        اجرای مدل برای هر ۳ طول (12، 18، 24) — encoder یک بار، decoder سه بار
        Returns: [{"words": int, "key": str|None, "confidence": float}, ...]
        """
        if not self._model:
            return [{"words": w, "key": None, "confidence": 0.0} for w in [12, 18, 24]]

        src = self._hash_tok.batch_encode([hash_str]).to(self.device)
        raw_modes = self._model.batch_generate_multimode(src, self.device)[0]

        return [
            {
                "words":      m["words"],
                "key":        self._key_tok.decode(m["ids"]) if m["ids"] else None,
                "confidence": m["confidence"],
            }
            for m in raw_modes
        ]

    def decode(self, hash_str: str) -> dict:
        """
        رمزگشایی یه هش — اول lookup، اگه نبود ML

        Returns:
            {"key": str|None, "source": "lookup"|"ml"|"not_found",
             "confidence": float|None, "ms": float}
        """
        t0 = time.perf_counter()
        try:
            hash_str = normalize_hash(hash_str)
        except ValueError:
            hash_str = hash_str.strip()

        key = self._lookup(hash_str)
        if key:
            self.stats.record("lookup")
            return {
                "key": key, "source": "lookup", "confidence": 1.0,
                "ms": round((time.perf_counter() - t0) * 1000, 2),
            }

        key, conf = self._ml_decode(hash_str)
        if key:
            self.stats.record("ml")
            return {
                "key": key, "source": "ml", "confidence": conf,
                "ms": round((time.perf_counter() - t0) * 1000, 2),
            }

        self.stats.record("not_found")
        return {
            "key": None, "source": "not_found", "confidence": None,
            "ms": round((time.perf_counter() - t0) * 1000, 2),
        }

    def compare(self, hash_str: str) -> dict:
        """
        نتیجه lookup و ML رو همزمان برمی‌گردونه — برای QA و مانیتورینگ کیفیت مدل.
        وقتی هش هم در دیتابیس هست و هم ML می‌تونه پیشش کنه، agree نشون می‌ده
        نتایج روی هم می‌خونن یا نه.

        Returns:
            {
              "lookup_key": str|None,
              "ml_key": str|None,
              "ml_confidence": float|None,
              "agree": bool|None,   ← None اگه یکی از دو نتیجه موجود نباشه
              "ms": float
            }
        """
        t0 = time.perf_counter()
        hash_str = hash_str.strip()

        lookup_key       = self._lookup(hash_str)
        ml_key, ml_conf  = self._ml_decode(hash_str)

        agree = None
        if lookup_key is not None and ml_key is not None:
            agree = (lookup_key.strip() == ml_key.strip())
            self.stats.record_compare(agree)

        return {
            "lookup_key":    lookup_key,
            "ml_key":        ml_key,
            "ml_confidence": ml_conf if ml_key else None,
            "agree":         agree,
            "ms":            round((time.perf_counter() - t0) * 1000, 2),
        }

    def multi_decode(self, hash_str: str) -> dict:
        """
        رمزگشایی با نمایش هر ۳ حالت طول (12/18/24) — هوشمندتر از decode()

        جریان کار:
          1. نرمال‌سازی ورودی (استخراج آدرس تران از هر متنی)
          2. Lookup در دیتابیس → اگه پیدا شد فوری برمی‌گرده (همراه طول واقعی)
          3. ML → encoder یک بار، decoder سه بار (12/18/24) — confidence برای هر حالت
          4. بهترین حالت = بیشترین confidence

        Returns:
            {
              "hash":   str,
              "source": "lookup" | "ml" | "not_found",
              "best":   {"words": int, "key": str, "confidence": float} | None,
              "modes":  [{"words": int, "key": str|None, "confidence": float}, ...],
              "ms":     float,
            }
        """
        t0 = time.perf_counter()
        try:
            hash_str = normalize_hash(hash_str)
        except ValueError:
            hash_str = hash_str.strip()

        # ── ۱. Lookup ─────────────────────────────────────────────────
        lookup_key = self._lookup(hash_str)
        if lookup_key:
            word_count = len(lookup_key.strip().split())
            entry = {"words": word_count, "key": lookup_key, "confidence": 1.0}
            self.stats.record("lookup")
            return {
                "hash":   hash_str,
                "source": "lookup",
                "best":   entry,
                "modes":  [entry],
                "ms":     round((time.perf_counter() - t0) * 1000, 2),
            }

        # ── ۲. ML multi-mode ──────────────────────────────────────────
        if not self._model:
            self.stats.record("not_found")
            return {
                "hash":   hash_str,
                "source": "not_found",
                "best":   None,
                "modes":  [],
                "ms":     round((time.perf_counter() - t0) * 1000, 2),
            }

        modes = self._ml_multi_decode(hash_str)

        valid_modes = [m for m in modes if m["key"]]
        best = (
            max(valid_modes, key=lambda x: x["confidence"])
            if valid_modes else None
        )

        source = "ml" if best else "not_found"
        self.stats.record(source)
        return {
            "hash":   hash_str,
            "source": source,
            "best":   best,
            "modes":  modes,
            "ms":     round((time.perf_counter() - t0) * 1000, 2),
        }

    def batch_decode(self, hashes: list[str]) -> list[dict]:
        """
        رمزگشایی دسته‌ای — lookup تک‌تک (سریع با cache)،
        ML برای باقی‌مونده‌ها به صورت batch واقعی (نه loop)
        """
        t0 = time.perf_counter()
        results: list[Optional[dict]] = [None] * len(hashes)
        ml_indices: list[int] = []
        ml_hashes:  list[str] = []

        # مرحله ۱: lookup برای همه
        for i, h in enumerate(hashes):
            h = h.strip()
            key = self._lookup(h)
            if key:
                self.stats.record("lookup")
                results[i] = {
                    "key": key, "source": "lookup", "confidence": 1.0, "ms": None
                }
            else:
                ml_indices.append(i)
                ml_hashes.append(h)

        # مرحله ۲: ML batch واحد برای باقی‌مونده‌ها
        if ml_hashes:
            ml_results = self._ml_decode_batch(ml_hashes)
            for idx, (key, conf) in zip(ml_indices, ml_results):
                if key:
                    self.stats.record("ml")
                    results[idx] = {
                        "key": key, "source": "ml", "confidence": conf, "ms": None
                    }
                else:
                    self.stats.record("not_found")
                    results[idx] = {
                        "key": None, "source": "not_found", "confidence": None, "ms": None
                    }

        total_ms = round((time.perf_counter() - t0) * 1000, 2)
        for r in results:
            r["ms"] = total_ms  # زمان کل batch (per-item غیردقیقه)
        return results

    def close(self):
        if self._con:
            self._con.close()


# ── singleton برای FastAPI ──────────────────────────────────────────

_pipeline: Optional[DecoderPipeline] = None


def get_pipeline() -> DecoderPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = DecoderPipeline(
            db_path    = os.getenv("DB_PATH",    "data/hashes.db"),
            model_path = os.getenv("MODEL_PATH", "model/checkpoints/best_model.pt"),
            device     = os.getenv("DEVICE",     "cpu"),
            cache_size = int(os.getenv("CACHE_SIZE", "100000")),
        )
    return _pipeline


if __name__ == "__main__":
    p = get_pipeline()
    test_hashes = ["xK9mP2qRs3nT4vU5", "ABC123", "1234567890abcdef1234567890abcdef12"]
    for h in test_hashes:
        result = p.decode(h)
        print(f"  [{result['source']:10s} {result['ms']:5.1f}ms] {h[:20]:20s} → {str(result['key'])[:40]}")

    print("\n--- batch ---")
    batch_res = p.batch_decode(test_hashes)
    for h, r in zip(test_hashes, batch_res):
        print(f"  [{r['source']:10s}] {h[:20]:20s} → {str(r['key'])[:40]}")

    print("\n--- stats ---")
    print(p.stats.snapshot())
