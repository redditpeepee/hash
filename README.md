# Hash Decoder — راهنمای کامل

## ساختار پروژه
```
hash-decoder/
├── data/
│   └── import_csv.py       ← ایمپورت CSV به DuckDB
├── model/
│   ├── bip39.py            ← مدیریت wordlist
│   ├── tokenizer.py        ← توکنایزر هش
│   ├── seq2seq.py          ← معماری مدل + batch_generate
│   ├── feasibility_test.py ← تست امکان‌سنجی ML
│   └── train.py            ← آموزش کامل
├── api/
│   ├── inference.py        ← pipeline + cache + stats + compare
│   └── main.py             ← FastAPI
├── Dockerfile
├── railway.toml
└── requirements.txt
```

---

## مرحله ۱ — نصب
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## مرحله ۲ — ایمپورت داده
```bash
python -m data.import_csv --csv your_data.csv --db data/hashes.db
```

## مرحله ۳ — تست امکان‌سنجی ML
```bash
python -m model.feasibility_test --csv your_data.csv
```
نتیجه رو بخون — اگه `improvement > 5x` بود به مرحله بعد برو.

## مرحله ۴ — آموزش مدل
```bash
python -m model.train --db data/hashes.db --out model/checkpoints --epochs 10
```

## مرحله ۵ — اجرای پنل وب (Streamlit)
به‌جای کار با API خام، می‌تونی از پنل وب کامل استفاده کنی:
```bash
streamlit run web/app.py
```
مرورگر رو باز کن: `http://localhost:8501`

پنل سه بخش داره:
- **رمزگشایی تکی** — یه هش بده، نتیجه + منبع (lookup/ML) + اطمینان مدل رو ببین. یه حالت «مقایسه» هم داره که نتیجه lookup و ML رو کنار هم نشون می‌ده.
- **پردازش دسته‌ای** — فایل CSV یا TXT آپلود کن، نوار پیشرفت رو ببین، نتایج رو به‌صورت جدول دانلود کن.
- **داشبورد مدیریتی** — نمودار توزیع منبع پاسخ‌ها، gauge نرخ توافق lookup/ML، و وضعیت cache.

پنل مستقیماً از `DecoderPipeline` استفاده می‌کنه — اجرای جدا‌گونه FastAPI لازم نیست (مگه بخوای از بیرون هم بهش دسترسی API داشته باشی).

### قفل رمز اختیاری
اگه می‌خوای پنل رو روی یه URL عمومی (مثل Railway) دیپلوی کنی ولی فقط خودت بهش دسترسی داشته باشی:
```bash
PANEL_PASSWORD=یه-رمز-قوی streamlit run web/app.py
```
اگه این متغیر ست نشه، پنل بدون رمز مستقیم باز می‌شه (مناسب اجرای محلی).

## مرحله ۶ — اجرای API (اختیاری، اگه بخوای از بیرون هم درخواست بفرستی)
```bash
uvicorn api.main:app --reload --port 8000
```

---

## Endpoint های API (در صورت استفاده موازی با پنل)

| Method | مسیر | کار |
|---|---|---|
| GET  | `/`         | وضعیت سرور (DB/model loaded?) |
| POST | `/decode`   | رمزگشایی یه هش (lookup → ML fallback) |
| POST | `/batch`    | رمزگشایی تا 500 هش با هم — ML به‌صورت batch واقعی |
| POST | `/compare`  | نتیجه lookup و ML رو **همزمان** برمی‌گردونه (QA مدل) |
| GET  | `/metrics`  | آمار لحظه‌ای: تعداد درخواست به تفکیک منبع، نرخ توافق، cache |
| GET  | `/health`   | health check برای Railway |

### تست سریع
```bash
curl -X POST http://localhost:8000/decode \
  -H "Content-Type: application/json" \
  -d '{"hash": "xK9mP2qRs3nT4vU5"}'

curl -X POST http://localhost:8000/compare \
  -H "Content-Type: application/json" \
  -d '{"hash": "xK9mP2qRs3nT4vU5"}'

curl http://localhost:8000/metrics
```

---

## ویژگی‌های اضافه‌شده در این نسخه

### ۱. سرعت — کش + batch واقعی
- **LRU Cache** برای lookup های پرتکرار (پیش‌فرض ظرفیت: 100K، با env var `CACHE_SIZE` قابل تغییر)
- `batch_generate()` در مدل: N هش با هم در یک forward pass پردازش می‌شه (نه loop روی `generate()`)
- در `/batch`، lookup تک‌تک با cache انجام می‌شه و فقط هش‌های پیدا‌نشده به‌صورت batch به ML می‌رن

### ۲. دقت — confidence score
هر پاسخ ML یه عدد `confidence` بین 0 تا 1 داره (میانگین احتمال softmax توکن‌های انتخاب‌شده). می‌تونی threshold بگذاری و پاسخ‌های کم‌اطمینان رو فیلتر کنی.

### ۳. مانیتورینگ کیفیت مدل — `/compare`
وقتی هشی هم در دیتابیس lookup هست و هم ML می‌تونه پیشش کنه، `/compare` هر دو نتیجه رو با هم برمی‌گردونه و `agree: true/false` نشون می‌ده آیا مدل با ground truth یکی در می‌آد یا نه. این برای:
- سنجش دقت واقعی مدل روی داده‌های شناخته‌شده (بدون نیاز به test set جدا)
- تشخیص drift مدل بعد از مدتی که در production هست
استفاده کن. نتایج توی `/metrics` به‌صورت `agree_rate` تجمیع می‌شه.

### ۴. `/metrics`
بدون نیاز به دیتابیس خارجی، یه شمارنده in-memory داره که نشون می‌ده چند درصد ترافیک از lookup جواب گرفته، چند درصد از ML، و نرخ cache hit چقدره.

---

## ⚠️ توجه — فضای دیسک Railway
200M رکورد ≈ 8-15GB دیسک
Railway Hobby plan: 100GB → کافیه
Railway Free plan: 1GB → ناکافیه — فقط با ML بدون DB استفاده کن

## دیپلوی روی Railway — دو حالت ممکن
این پروژه دو Dockerfile جدا داره:
- `Dockerfile` + `railway.toml` → فقط API (FastAPI روی پورت 8000)
- `Dockerfile.streamlit` + `railway.streamlit.toml` → فقط پنل وب (Streamlit روی پورت 8501)

برای دیپلوی پنل، نام `Dockerfile.streamlit` رو در تنظیمات Railway به‌عنوان build path بده، یا قبل از push اون رو به `Dockerfile` رینیم کن. اگه به هر دو همزمان نیاز داری (هم پنل هم API برای اتوماسیون بیرونی)، باید دو سرویس جدا در Railway بسازی که از یه volume مشترک (برای DB و مدل) استفاده کنن.

## نکته درباره Authentication
فعلاً API بدون auth/rate-limit طراحی شده (مناسب استفاده داخلی). پنل وب هم به‌صورت پیش‌فرض بدون رمزه، مگه با env var `PANEL_PASSWORD` یه رمز ساده فعال کنی. اگه بعداً نیاز شد عمومی/خارجی‌تر بشه، اضافه کردن API-Key header یا rate-limit با middleware ساده‌ست — کافیه بگی تا اضافه کنم.

