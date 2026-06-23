"""
آموزش کامل مدل روی 200M نمونه
با streaming از DuckDB (بدون نیاز به RAM زیاد)

استفاده:
    python -m model.train \
        --db    data/hashes.db \
        --out   model/checkpoints \
        --epochs 10
"""

import argparse
import os
import time
import math
import random

import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
import duckdb

from model.tokenizer import HashTokenizer
from model.bip39     import KeyTokenizer, PAD_IDX
from model.seq2seq   import HashDecoder


# ── تنظیمات آموزش ──────────────────────────────────────────────────
BATCH_SIZE      = 128
LR              = 3e-4
EPOCHS          = 10
CHUNK_SIZE      = 50_000     # تعداد ردیف در هر chunk از DuckDB
SAVE_EVERY      = 50_000     # هر چند batch یه checkpoint ذخیره میشه
WARMUP_STEPS    = 1_000
MAX_GRAD_NORM   = 1.0

# اندازه مدل (برای Railway CPU — متعادل)
D_MODEL    = 128
NHEAD      = 4
ENC_LAYERS = 3
DEC_LAYERS = 3
DIM_FF     = 256


class DuckDBStreamDataset(IterableDataset):
    """
    Dataset که به صورت streaming از DuckDB می‌خونه
    مناسب برای 200M رکورد بدون مشکل RAM
    """

    def __init__(self, db_path: str, hash_tok, key_tok,
                 chunk_size: int = CHUNK_SIZE, shuffle_chunks: bool = True):
        self.db_path      = db_path
        self.hash_tok     = hash_tok
        self.key_tok      = key_tok
        self.chunk_size   = chunk_size
        self.shuffle_chunks = shuffle_chunks

    def __iter__(self):
        con = duckdb.connect(self.db_path, read_only=True)
        total = con.execute("SELECT COUNT(*) FROM pairs").fetchone()[0]
        offsets = list(range(0, total, self.chunk_size))

        if self.shuffle_chunks:
            random.shuffle(offsets)

        for offset in offsets:
            rows = con.execute(
                "SELECT hash, key FROM pairs LIMIT ? OFFSET ?",
                [self.chunk_size, offset]
            ).fetchall()

            if self.shuffle_chunks:
                random.shuffle(rows)

            for h, k in rows:
                src = torch.tensor(self.hash_tok.encode(h), dtype=torch.long)
                tgt = torch.tensor(self.key_tok.encode(k),  dtype=torch.long)
                yield src, tgt

        con.close()


def collate_fn(batch):
    srcs, tgts = zip(*batch)
    src_t = torch.stack(srcs)
    max_t = max(t.size(0) for t in tgts)
    tgt_t = torch.zeros(len(tgts), max_t, dtype=torch.long)
    for i, t in enumerate(tgts):
        tgt_t[i, :t.size(0)] = t
    return src_t, tgt_t


def get_lr(step: int, d_model: int, warmup: int) -> float:
    """Transformer learning rate schedule"""
    if step == 0:
        return 0.0
    return d_model ** -0.5 * min(step ** -0.5, step * warmup ** -1.5)


class Trainer:
    def __init__(self, db_path: str, out_dir: str):
        self.db_path = db_path
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

        self.device   = "cuda" if torch.cuda.is_available() else "cpu"
        self.hash_tok = HashTokenizer()
        self.key_tok  = KeyTokenizer()

        self.model = HashDecoder(
            src_vocab_size = self.hash_tok.vocab_size,
            tgt_vocab_size = self.key_tok.vocab_size,
            d_model    = D_MODEL,
            nhead      = NHEAD,
            enc_layers = ENC_LAYERS,
            dec_layers = DEC_LAYERS,
            dim_ff     = DIM_FF,
        ).to(self.device)

        self.criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LR, betas=(0.9, 0.98))
        self.scaler    = torch.cuda.amp.GradScaler() if self.device == "cuda" else None

        self.global_step = 0
        self.best_loss   = float("inf")

        print(f"✅ Trainer ready")
        print(f"   Device  : {self.device}")
        print(f"   Params  : {self.model.count_params()}")
        print(f"   Out dir : {out_dir}")

    def _adjust_lr(self):
        lr = get_lr(self.global_step + 1, D_MODEL, WARMUP_STEPS)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def train_epoch(self, epoch: int) -> float:
        dataset = DuckDBStreamDataset(self.db_path, self.hash_tok, self.key_tok)
        loader  = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn,
                             num_workers=0)

        self.model.train()
        total_loss = 0.0
        n_batches  = 0
        t0 = time.time()

        for batch_idx, (src, tgt) in enumerate(loader):
            src = src.to(self.device)
            tgt = tgt.to(self.device)
            inp = tgt[:, :-1]
            lbl = tgt[:, 1:]

            self._adjust_lr()

            if self.scaler:   # GPU با mixed precision
                with torch.cuda.amp.autocast():
                    logits = self.model(src, inp)
                    loss   = self.criterion(
                        logits.reshape(-1, self.key_tok.vocab_size),
                        lbl.reshape(-1)
                    )
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), MAX_GRAD_NORM)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:             # CPU
                logits = self.model(src, inp)
                loss   = self.criterion(
                    logits.reshape(-1, self.key_tok.vocab_size),
                    lbl.reshape(-1)
                )
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), MAX_GRAD_NORM)
                self.optimizer.step()

            total_loss      += loss.item()
            n_batches       += 1
            self.global_step += 1

            # لاگ هر 500 batch
            if batch_idx % 500 == 0:
                elapsed = time.time() - t0
                avg     = total_loss / n_batches
                speed   = (batch_idx + 1) * BATCH_SIZE / elapsed
                cur_lr  = self.optimizer.param_groups[0]["lr"]
                print(
                    f"  E{epoch} | step={self.global_step:,}"
                    f"  loss={avg:.4f}  lr={cur_lr:.2e}"
                    f"  speed={speed:,.0f} samples/s"
                )

            # ذخیره checkpoint
            if self.global_step % SAVE_EVERY == 0:
                self.save(f"step_{self.global_step}.pt")

        return total_loss / max(n_batches, 1)

    def save(self, filename: str):
        path = os.path.join(self.out_dir, filename)
        torch.save({
            "epoch":        self.global_step,
            "model_state":  self.model.state_dict(),
            "optimizer":    self.optimizer.state_dict(),
            "loss":         self.best_loss,
        }, path)
        print(f"💾 Saved: {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.global_step = ckpt["epoch"]
        self.best_loss   = ckpt["loss"]
        print(f"📂 Loaded checkpoint: {path} (step={self.global_step})")

    def run(self, epochs: int):
        print(f"\n🚀 Starting training for {epochs} epochs\n")
        for epoch in range(1, epochs + 1):
            t0   = time.time()
            loss = self.train_epoch(epoch)
            elapsed = time.time() - t0
            print(f"\n✅ Epoch {epoch}/{epochs} done | avg_loss={loss:.4f} | time={elapsed/60:.1f}min\n")

            if loss < self.best_loss:
                self.best_loss = loss
                self.save("best_model.pt")

        self.save("final_model.pt")
        print("🎉 Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db",     default="data/hashes.db")
    parser.add_argument("--out",    default="model/checkpoints")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--resume", default=None, help="Path to checkpoint")
    args = parser.parse_args()

    trainer = Trainer(args.db, args.out)
    if args.resume:
        trainer.load(args.resume)
    trainer.run(args.epochs)
