"""
تست امکان‌سنجی ML — آیا مدل می‌تونه الگو یاد بگیره؟

قبل از اینکه ساعت‌ها روی training وقت بذاری،
این اسکریپت با 10K نمونه مشخص می‌کنه
آیا ML اصلاً قادر به تعمیمه یا نه.

نتیجه‌گیری:
  ✅ val_loss کاهش می‌یابد → الگو وجود داره → full training مفیده
  ❌ val_loss ثابت می‌مونه → رابطه تصادفیه → فقط Lookup کار می‌کنه

استفاده:
    python -m model.feasibility_test --csv data/hashes.csv
"""

import argparse
import random
import csv
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ── کوچک‌ترین مدل ممکن برای تست سریع ──────────────────────────────
TINY_D_MODEL  = 64
TINY_HEADS    = 2
TINY_LAYERS   = 2
TINY_FF       = 128

BATCH_SIZE    = 64
EPOCHS        = 40
LR            = 1e-3
N_SAMPLES     = 10_000   # تعداد نمونه برای تست
TRAIN_RATIO   = 0.8


class PairDataset(Dataset):
    def __init__(self, pairs, hash_tok, key_tok):
        self.pairs    = pairs
        self.hash_tok = hash_tok
        self.key_tok  = key_tok

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        h, k = self.pairs[idx]
        src = torch.tensor(self.hash_tok.encode(h), dtype=torch.long)
        tgt = torch.tensor(self.key_tok.encode(k),  dtype=torch.long)
        return src, tgt


def collate_fn(batch):
    srcs, tgts = zip(*batch)
    src_t = torch.stack(srcs)
    # تراز کردن target ها به طولانی‌ترین
    max_t = max(t.size(0) for t in tgts)
    tgt_t = torch.zeros(len(tgts), max_t, dtype=torch.long)
    for i, t in enumerate(tgts):
        tgt_t[i, :t.size(0)] = t
    return src_t, tgt_t


def load_samples(csv_path: str, n: int) -> list[tuple[str, str]]:
    """خواندن n نمونه از CSV"""
    pairs = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= n:
                break
            pairs.append((row["hash"].strip(), row["key"].strip()))
    if len(pairs) < n:
        print(f"⚠️  Only found {len(pairs)} samples (expected {n})")
    return pairs


def word_accuracy(logits: torch.Tensor, tgt: torch.Tensor) -> float:
    """دقت کلمه‌به‌کلمه (توکن‌های padding رو نادیده می‌گیره)"""
    from model.bip39 import PAD_IDX
    pred = logits.argmax(-1)         # (B, T)
    mask = tgt != PAD_IDX
    correct = ((pred == tgt) & mask).sum().item()
    total   = mask.sum().item()
    return correct / total if total > 0 else 0.0


def run_feasibility(csv_path: str, output_plot: str = "feasibility_result.png"):
    from model.tokenizer import HashTokenizer
    from model.bip39     import KeyTokenizer, PAD_IDX
    from model.seq2seq   import HashDecoder

    print("=" * 60)
    print("🧪 ML Feasibility Test")
    print("=" * 60)

    # ─── بارگذاری داده ──────────────────────────────────────────
    print(f"\n📂 Loading {N_SAMPLES:,} samples from: {csv_path}")
    pairs = load_samples(csv_path, N_SAMPLES)
    random.shuffle(pairs)

    n_train = int(len(pairs) * TRAIN_RATIO)
    train_pairs = pairs[:n_train]
    val_pairs   = pairs[n_train:]
    print(f"   Train: {len(train_pairs):,} | Val: {len(val_pairs):,}")

    hash_tok = HashTokenizer()
    key_tok  = KeyTokenizer()

    train_ds = PairDataset(train_pairs, hash_tok, key_tok)
    val_ds   = PairDataset(val_pairs,   hash_tok, key_tok)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_fn)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # ─── مدل tiny ───────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n⚙️  Device : {device}")

    model = HashDecoder(
        src_vocab_size = hash_tok.vocab_size,
        tgt_vocab_size = key_tok.vocab_size,
        d_model   = TINY_D_MODEL,
        nhead     = TINY_HEADS,
        enc_layers= TINY_LAYERS,
        dec_layers= TINY_LAYERS,
        dim_ff    = TINY_FF,
    ).to(device)
    print(f"   Params: {model.count_params()}")

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []

    # ─── حلقه آموزش ─────────────────────────────────────────────
    print(f"\n🚀 Training for {EPOCHS} epochs...\n")
    t_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        t_loss, t_acc, n = 0.0, 0.0, 0
        for src, tgt in train_dl:
            src, tgt = src.to(device), tgt.to(device)
            inp = tgt[:, :-1]
            lbl = tgt[:, 1:]
            logits = model(src, inp)
            loss = criterion(logits.reshape(-1, key_tok.vocab_size), lbl.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss += loss.item()
            t_acc  += word_accuracy(logits, lbl)
            n += 1
        train_losses.append(t_loss / n)
        train_accs.append(t_acc / n)

        # Validation
        model.eval()
        v_loss, v_acc, n = 0.0, 0.0, 0
        with torch.no_grad():
            for src, tgt in val_dl:
                src, tgt = src.to(device), tgt.to(device)
                inp = tgt[:, :-1]
                lbl = tgt[:, 1:]
                logits = model(src, inp)
                v_loss += criterion(logits.reshape(-1, key_tok.vocab_size), lbl.reshape(-1)).item()
                v_acc  += word_accuracy(logits, lbl)
                n += 1
        val_losses.append(v_loss / n)
        val_accs.append(v_acc / n)

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d}/{EPOCHS}"
                f"  | train_loss={train_losses[-1]:.4f}  val_loss={val_losses[-1]:.4f}"
                f"  | train_acc={train_accs[-1]:.3%}  val_acc={val_accs[-1]:.3%}"
            )

    elapsed = time.time() - t_start
    print(f"\n⏱️  Total time: {elapsed:.1f}s")

    # ─── رسم نمودار ─────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(train_losses, label="Train Loss")
        ax1.plot(val_losses,   label="Val Loss")
        ax1.set_title("Loss Curve")
        ax1.set_xlabel("Epoch")
        ax1.legend()
        ax1.grid(True)

        ax2.plot([a * 100 for a in train_accs], label="Train Acc %")
        ax2.plot([a * 100 for a in val_accs],   label="Val Acc %")
        ax2.axhline(1 / 2048 * 100, color="red", linestyle="--", label="Random chance")
        ax2.set_title("Word Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("%")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(output_plot, dpi=150)
        print(f"📊 Plot saved: {output_plot}")
    except Exception as e:
        print(f"⚠️  Could not save plot: {e}")

    # ─── نتیجه‌گیری ─────────────────────────────────────────────
    final_val_acc  = val_accs[-1]
    random_chance  = 1 / 2048       # ≈ 0.049%
    improvement    = final_val_acc / random_chance

    print("\n" + "=" * 60)
    print("📋 RESULT")
    print("=" * 60)
    print(f"  Final val accuracy : {final_val_acc:.4%}")
    print(f"  Random chance      : {random_chance:.4%}")
    print(f"  Improvement factor : {improvement:.1f}x")
    print()

    if improvement > 5:
        print("  ✅ الگوی قابل یادگیری وجود داره!")
        print("     → Full training روی 200M نمونه توصیه میشه")
    elif improvement > 2:
        print("  ⚠️  الگوی ضعیفی وجود داره — ارزش تست با داده بیشتر رو داره")
    else:
        print("  ❌ مدل نمی‌تونه تعمیم بده")
        print("     → Lookup table تنها راه مطمئنه")
        print("     → هش احتمالاً واقعاً رمزنگاری قویه")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",  required=True, help="Path to CSV file")
    parser.add_argument("--plot", default="feasibility_result.png")
    args = parser.parse_args()
    run_feasibility(args.csv, args.plot)
