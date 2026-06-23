"""
ایمپورت CSV به DuckDB
برای 200 میلیون رکورد بهینه‌سازی شده

استفاده:
    python -m data.import_csv --csv data/hashes.csv --db data/hashes.db
"""

import argparse
import time
import duckdb
import os


def import_csv(csv_path: str, db_path: str, batch_size: int = 1_000_000):
    """
    وارد کردن CSV به DuckDB

    فرمت CSV انتظاری:
        hash,key
        xK9mP2...,abandon ability able ...
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    file_size_gb = os.path.getsize(csv_path) / 1e9
    print(f"📁 CSV size   : {file_size_gb:.2f} GB")
    print(f"📦 DB path    : {db_path}")
    print(f"⚙️  Batch size : {batch_size:,}")
    print()

    con = duckdb.connect(db_path)

    # ساخت جدول
    con.execute("""
        CREATE TABLE IF NOT EXISTS pairs (
            hash VARCHAR NOT NULL,
            key  VARCHAR NOT NULL
        )
    """)

    # ایندکس برای جستجوی سریع (بعد از import اضافه میشه)
    print("⏳ Importing data (این ممکنه چند دقیقه طول بکشه)...")
    t0 = time.time()

    # DuckDB native CSV import — سریع‌ترین روش ممکن
    con.execute(f"""
        INSERT INTO pairs
        SELECT column0 AS hash, column1 AS key
        FROM read_csv_auto('{csv_path}',
            header = true,
            columns = {{'hash': 'VARCHAR', 'key': 'VARCHAR'}}
        )
    """)

    count = con.execute("SELECT COUNT(*) FROM pairs").fetchone()[0]
    elapsed = time.time() - t0

    print(f"✅ Imported {count:,} rows in {elapsed:.1f}s")
    print(f"   Speed: {count / elapsed:,.0f} rows/sec")
    print()

    # ساخت ایندکس برای lookup سریع
    print("🔨 Building index (یه بار انجام میشه)...")
    t1 = time.time()
    con.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_hash ON pairs(hash)")
    print(f"✅ Index built in {time.time() - t1:.1f}s")

    # اطلاعات دیتابیس
    db_size_gb = os.path.getsize(db_path) / 1e9
    print(f"\n📊 DB size: {db_size_gb:.2f} GB")

    con.close()
    print("\n🎉 Import complete!")


def verify_db(db_path: str, n_samples: int = 5):
    """بررسی صحت دیتابیس"""
    con = duckdb.connect(db_path, read_only=True)
    count = con.execute("SELECT COUNT(*) FROM pairs").fetchone()[0]
    print(f"\n📊 Total rows: {count:,}")
    print(f"\nSample rows:")
    rows = con.execute(f"SELECT hash, key FROM pairs LIMIT {n_samples}").fetchall()
    for h, k in rows:
        words = k.split()
        print(f"  hash: {h[:20]}... | key: {' '.join(words[:4])}... ({len(words)} words)")

    # سرعت lookup
    sample_hash = rows[0][0]
    import time
    t = time.time()
    for _ in range(1000):
        con.execute("SELECT key FROM pairs WHERE hash = ?", [sample_hash]).fetchone()
    avg_ms = (time.time() - t)
    print(f"\n⚡ Lookup speed: {avg_ms:.2f}s per 1000 queries ({avg_ms:.3f}ms avg)")
    con.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import CSV hashes to DuckDB")
    parser.add_argument("--csv", required=True, help="Path to CSV file")
    parser.add_argument("--db",  default="data/hashes.db", help="DuckDB path")
    parser.add_argument("--verify", action="store_true", help="Only verify existing DB")
    args = parser.parse_args()

    if args.verify:
        verify_db(args.db)
    else:
        import_csv(args.csv, args.db)
        verify_db(args.db)
