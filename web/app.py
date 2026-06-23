"""
پنل وب Hash Decoder — Streamlit

اجرا:
    streamlit run web/app.py

این پنل مستقیماً از DecoderPipeline استفاده می‌کنه — نیازی به اجرای FastAPI نیست.
"""

import os
import sys
import time
import io

# اضافه کردن ریشه پروژه به sys.path تا importهای api/model کار کنن
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from api.inference import DecoderPipeline, normalize_hash


# ═══════════════════════════════════════════════════════════════════
# تنظیمات صفحه
# ═══════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="پنل رمزگشای هش",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS برای راست‌چین کردن متن فارسی
st.markdown("""
<style>
    .stApp, .stMarkdown, p, h1, h2, h3, h4, label, .stTextInput input {
        direction: rtl;
        text-align: right;
    }
    .stTextInput input, .stTextArea textarea {
        text-align: left;
        direction: ltr;
        font-family: monospace;
    }
    div[data-testid="stMetricValue"] {
        direction: ltr;
    }
    .stDataFrame {
        direction: ltr;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# قفل رمز اختیاری (فقط اگه PANEL_PASSWORD ست شده باشه فعال می‌شه)
# مفید برای زمانی که پنل روی یه URL عمومی (مثل Railway) دیپلوی می‌شه
# ═══════════════════════════════════════════════════════════════════

PANEL_PASSWORD = os.getenv("PANEL_PASSWORD", "")

if PANEL_PASSWORD:
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.title("🔐 پنل رمزگشای هش")
        pwd = st.text_input("رمز عبور", type="password")
        if st.button("ورود"):
            if pwd == PANEL_PASSWORD:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("رمز اشتباهه")
        st.stop()


# ═══════════════════════════════════════════════════════════════════
# بارگذاری Pipeline (کش می‌شه — فقط یک بار لود می‌شه)
# ═══════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_pipeline(db_path: str, model_path: str, device: str, cache_size: int):
    return DecoderPipeline(
        db_path=db_path or None,
        model_path=model_path or None,
        device=device,
        cache_size=cache_size,
    )


def get_default_paths():
    return {
        "db":    os.getenv("DB_PATH",    os.path.join(PROJECT_ROOT, "data", "hashes.db")),
        "model": os.getenv("MODEL_PATH", os.path.join(PROJECT_ROOT, "model", "checkpoints", "best_model.pt")),
        "device": os.getenv("DEVICE", "cpu"),
    }


# ═══════════════════════════════════════════════════════════════════
# Sidebar — تنظیمات و وضعیت
# ═══════════════════════════════════════════════════════════════════

defaults = get_default_paths()

st.sidebar.title("⚙️ تنظیمات")

db_path = st.sidebar.text_input("مسیر دیتابیس (DuckDB)", value=defaults["db"])
model_path = st.sidebar.text_input("مسیر مدل (.pt)", value=defaults["model"])
device = st.sidebar.selectbox("Device", ["cpu", "cuda"], index=0)

if st.sidebar.button("🔄 بارگذاری مجدد Pipeline", use_container_width=True):
    st.cache_resource.clear()
    st.rerun()

st.sidebar.divider()

pipeline = load_pipeline(db_path, model_path, device, 100_000)

st.sidebar.subheader("📡 وضعیت سرویس")

db_ok = pipeline._con is not None
model_ok = pipeline._model is not None

col_a, col_b = st.sidebar.columns(2)
col_a.metric("دیتابیس", "✅ فعال" if db_ok else "❌ غیرفعال")
col_b.metric("مدل ML", "✅ فعال" if model_ok else "❌ غیرفعال")

if db_ok:
    try:
        count = pipeline._con.execute("SELECT COUNT(*) FROM pairs").fetchone()[0]
        st.sidebar.caption(f"تعداد رکورد در دیتابیس: **{count:,}**")
    except Exception:
        pass

if not db_ok and not model_ok:
    st.sidebar.warning("⚠️ هیچ دیتابیس یا مدلی لود نشده. مسیرها رو چک کن.")


# ═══════════════════════════════════════════════════════════════════
# هدر اصلی
# ═══════════════════════════════════════════════════════════════════

st.title("🔐 پنل رمزگشای هش")
st.caption("رمزگشایی تکی، پردازش دسته‌ای فایل، و مانیتورینگ کیفیت مدل")

tab1, tab2, tab3 = st.tabs(["🔍 رمزگشایی تکی", "📂 پردازش دسته‌ای", "📊 داشبورد مدیریتی"])


# ═══════════════════════════════════════════════════════════════════
# تب ۱ — رمزگشایی تکی
# ═══════════════════════════════════════════════════════════════════

with tab1:
    st.subheader("رمزگشایی یه هش — همه حالت‌های طول")
    st.caption(
        "آدرس تران رو وارد کن (۳۴ کاراکتر). "
        "می‌تونی آدرس خام، لینک TronScan، یا هر متنی که آدرس داخلشه رو بدی."
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        hash_input = st.text_input(
            "هش / آدرس تران",
            placeholder="TT7q9cR9odRkdaM2sKKWNAU66EM32ZEviZ  یا  https://tronscan.org/#/address/T...",
            key="single_hash_input",
        )
    with col2:
        st.write("")
        st.write("")
        decode_clicked = st.button(
            "🔓 رمزگشایی (۱۲ / ۱۸ / ۲۴)", use_container_width=True, type="primary"
        )

    if decode_clicked and hash_input.strip():
        # ── نرمال‌سازی ورودی ────────────────────────────────────────
        try:
            clean_hash = normalize_hash(hash_input.strip())
            if clean_hash != hash_input.strip():
                st.info(f"📍 آدرس استخراج‌شده: `{clean_hash}`")
        except ValueError as e:
            st.error(f"❌ {e}")
            st.stop()

        with st.spinner("در حال اجرای مدل برای هر ۳ طول ممکن..."):
            result = pipeline.multi_decode(clean_hash)

        # ── نتیجه Lookup ─────────────────────────────────────────────
        if result["source"] == "lookup":
            best = result["best"]
            st.success("🟢 **از دیتابیس (Lookup) — دقت ۱۰۰٪**")

            c1, c2, c3 = st.columns(3)
            c1.metric("طول", f"{best['words']} کلمه")
            c2.metric("اطمینان", "۱۰۰٪")
            c3.metric("زمان پاسخ", f"{result['ms']} ms")

            st.code(best["key"], language=None)

        # ── نتیجه ML — سه حالت کنار هم ─────────────────────────────
        elif result["source"] == "ml":
            best_words = result["best"]["words"] if result["best"] else None
            modes = result["modes"]

            st.markdown(
                "#### 🤖 نتایج مدل — هر ۳ طول ممکن  \n"
                "<small>بالاترین confidence = محتمل‌ترین طول</small>",
                unsafe_allow_html=True,
            )
            st.caption(f"زمان پردازش: **{result['ms']} ms**")

            cols = st.columns(3)
            for col, mode in zip(cols, modes):
                is_best = (mode["words"] == best_words)
                conf_pct = f"{mode['confidence'] * 100:.1f}٪"

                with col:
                    if is_best:
                        st.markdown(
                            f"<div style='background:#eafaf1;padding:8px 12px;"
                            f"border-radius:8px;border:2px solid #2ecc71'>"
                            f"<b>🥇 {mode['words']} کلمه</b><br>"
                            f"<span style='font-size:1.3em;color:#27ae60'>{conf_pct}</span>"
                            f"<br><small>بهترین حالت</small></div>",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f"<div style='background:#f8f9fa;padding:8px 12px;"
                            f"border-radius:8px;border:1px solid #dee2e6'>"
                            f"<b>🔑 {mode['words']} کلمه</b><br>"
                            f"<span style='font-size:1.3em;color:#6c757d'>{conf_pct}</span>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                    st.write("")
                    if mode["key"]:
                        st.code(mode["key"], language=None)
                    else:
                        st.caption("— تولید نشد —")

        # ── پیدا نشد ────────────────────────────────────────────────
        else:
            st.error(
                "❌ این هش نه در دیتابیس و نه توسط مدل قابل رمزگشایی نبود.\n\n"
                "مطمئن بشید دیتابیس لود شده یا مدل آموزش دیده."
            )

    st.divider()

    # ── حالت مقایسه ──────────────────────────────────────────────
    with st.expander("🔬 حالت مقایسه (Lookup vs ML) — برای بررسی کیفیت مدل"):
        st.caption(
            "اگه هش هم در دیتابیس و هم قابل تخمین توسط ML باشه، "
            "می‌تونی هر دو نتیجه رو کنار هم ببینی و چک کنی مدل چقدر دقیقه."
        )
        compare_hash = st.text_input("هش برای مقایسه", key="compare_hash_input")
        if st.button("مقایسه کن"):
            if compare_hash.strip():
                with st.spinner("در حال مقایسه..."):
                    cmp = pipeline.compare(compare_hash.strip())

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**🟢 نتیجه Lookup (دیتابیس)**")
                    st.code(cmp["lookup_key"] or "— پیدا نشد —")
                with c2:
                    st.markdown("**🟡 نتیجه ML**")
                    st.code(cmp["ml_key"] or "— تولید نشد —")
                    if cmp["ml_confidence"] is not None:
                        st.caption(f"اطمینان: {cmp['ml_confidence']*100:.1f}%")

                if cmp["agree"] is True:
                    st.success("✅ دو نتیجه دقیقاً یکسانن — مدل درست تخمین زده")
                elif cmp["agree"] is False:
                    st.error("❌ دو نتیجه متفاوتن — مدل اشتباه تخمین زده")
                else:
                    st.info("ℹ️ برای مقایسه، هش باید هم در دیتابیس و هم قابل تخمین توسط مدل باشه.")


# ═══════════════════════════════════════════════════════════════════
# تب ۲ — پردازش دسته‌ای
# ═══════════════════════════════════════════════════════════════════

with tab2:
    st.subheader("پردازش دسته‌ای از فایل")
    st.caption("فایل CSV (با ستون hash) یا TXT (هر خط یه هش) آپلود کن")

    uploaded_file = st.file_uploader("انتخاب فایل", type=["csv", "txt"])

    hashes_to_process: list[str] = []

    if uploaded_file is not None:
        content = uploaded_file.read().decode("utf-8", errors="ignore")

        if uploaded_file.name.endswith(".csv"):
            try:
                df_in = pd.read_csv(io.StringIO(content))
                hash_col = None
                for c in df_in.columns:
                    if c.strip().lower() == "hash":
                        hash_col = c
                        break
                if hash_col is None:
                    hash_col = df_in.columns[0]
                hashes_to_process = df_in[hash_col].astype(str).str.strip().tolist()
            except Exception as e:
                st.error(f"خطا در خواندن CSV: {e}")
        else:
            hashes_to_process = [
                line.strip() for line in content.splitlines() if line.strip()
            ]

        hashes_to_process = [h for h in hashes_to_process if h]
        st.info(f"📄 تعداد هش شناسایی‌شده: **{len(hashes_to_process):,}**")

        if hashes_to_process:
            st.write("پیش‌نمایش ۵ مورد اول:")
            st.code("\n".join(hashes_to_process[:5]))

    col1, col2 = st.columns([1, 3])
    with col1:
        chunk_size = st.number_input("اندازه هر دسته (chunk)", min_value=10, max_value=2000, value=200, step=10)
    with col2:
        st.write("")
        process_clicked = st.button(
            "🚀 شروع پردازش", type="primary", disabled=(not hashes_to_process)
        )

    if process_clicked and hashes_to_process:
        progress_bar = st.progress(0, text="در حال پردازش...")
        results_all = []
        total = len(hashes_to_process)
        t_start = time.time()

        for i in range(0, total, chunk_size):
            chunk = hashes_to_process[i: i + chunk_size]
            chunk_results = pipeline.batch_decode(chunk)
            for h, r in zip(chunk, chunk_results):
                results_all.append({
                    "hash": h,
                    "key": r["key"],
                    "source": r["source"],
                    "confidence": r["confidence"],
                })
            done = min(i + chunk_size, total)
            progress_bar.progress(
                done / total, text=f"پردازش‌شده: {done:,} / {total:,}"
            )

        elapsed = time.time() - t_start
        progress_bar.progress(1.0, text="✅ پردازش کامل شد")

        result_df = pd.DataFrame(results_all)
        found_count = result_df["key"].notna().sum()

        st.success(
            f"✅ {total:,} هش در {elapsed:.1f} ثانیه پردازش شد "
            f"({total/elapsed:,.0f} هش/ثانیه) — {found_count:,} مورد پیدا شد"
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("کل", f"{total:,}")
        c2.metric("از Lookup", f"{(result_df['source']=='lookup').sum():,}")
        c3.metric("از ML", f"{(result_df['source']=='ml').sum():,}")
        c4.metric("پیدا نشد", f"{(result_df['source']=='not_found').sum():,}")

        st.dataframe(result_df, use_container_width=True, height=350)

        csv_bytes = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ دانلود نتایج (CSV)",
            data=csv_bytes,
            file_name="decode_results.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ═══════════════════════════════════════════════════════════════════
# تب ۳ — داشبورد مدیریتی
# ═══════════════════════════════════════════════════════════════════

with tab3:
    st.subheader("داشبورد مدیریتی")

    if st.button("🔄 بروزرسانی آمار"):
        st.rerun()

    snap = pipeline.stats.snapshot()
    cache_stats = pipeline.cache.stats()

    st.markdown("#### آمار کلی درخواست‌ها")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("کل درخواست‌ها", f"{snap['total_requests']:,}")
    c2.metric("از Lookup", f"{snap['by_source'].get('lookup', 0):,}")
    c3.metric("از ML", f"{snap['by_source'].get('ml', 0):,}")
    c4.metric("پیدا نشد", f"{snap['by_source'].get('not_found', 0):,}")

    st.divider()

    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown("#### توزیع منبع پاسخ‌ها")
        source_data = snap["by_source"]
        if sum(source_data.values()) > 0:
            fig = px.pie(
                names=list(source_data.keys()),
                values=list(source_data.values()),
                color=list(source_data.keys()),
                color_discrete_map={"lookup": "#2ecc71", "ml": "#f1c40f", "not_found": "#e74c3c"},
                hole=0.4,
            )
            fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("هنوز هیچ درخواستی ثبت نشده")

    with col_chart2:
        st.markdown("#### نرخ توافق Lookup vs ML (از /compare)")
        cmp = snap["compare"]
        if cmp["total"] > 0:
            agree_rate = cmp["agree_rate"] * 100
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=agree_rate,
                number={"suffix": "%"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#2ecc71" if agree_rate > 70 else "#f39c12" if agree_rate > 40 else "#e74c3c"},
                    "steps": [
                        {"range": [0, 40], "color": "#fdecea"},
                        {"range": [40, 70], "color": "#fff6e0"},
                        {"range": [70, 100], "color": "#eafaf1"},
                    ],
                },
            ))
            fig.update_layout(margin=dict(t=30, b=10, l=30, r=30), height=300)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"تعداد مقایسه‌های انجام‌شده: {cmp['total']:,} | موافق: {cmp['agree']:,} | مخالف: {cmp['disagree']:,}")
        else:
            st.info("هنوز هیچ مقایسه‌ای (تب رمزگشایی تکی → حالت مقایسه) انجام نشده")

    st.divider()

    st.markdown("#### وضعیت Cache")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("تعداد ورودی در Cache", f"{cache_stats['size']:,}")
    c2.metric("Cache Hit", f"{cache_stats['hits']:,}")
    c3.metric("Cache Miss", f"{cache_stats['misses']:,}")
    c4.metric("نرخ Hit", f"{cache_stats['hit_rate']*100:.1f}%")
