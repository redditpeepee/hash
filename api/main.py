"""
FastAPI — Hash Decoder API
"""

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

from api.inference import get_pipeline, DecoderPipeline, normalize_hash

# ── Schemas ─────────────────────────────────────────────────────────

class DecodeRequest(BaseModel):
    hash: str

    @field_validator("hash")
    @classmethod
    def clean_hash(cls, v: str) -> str:
        """نرمال‌سازی هوشمند — آدرس تران رو از هر ورودی استخراج می‌کنه"""
        try:
            return normalize_hash(v)
        except ValueError:
            # fallback برای backward-compatibility
            v = v.strip()
            if len(v) < 5 or len(v) > 50:
                raise ValueError(
                    "آدرس تران معتبر (۳۴ کاراکتر) پیدا نشد. "
                    "آدرس باید با T شروع بشه و دقیقاً ۳۴ کاراکتر base58 داشته باشه."
                )
            return v


class DecodeResponse(BaseModel):
    hash:       str
    key:        Optional[str]
    source:     str                # "lookup" | "ml" | "not_found"
    confidence: Optional[float]
    ms:         float


class BatchRequest(BaseModel):
    hashes: list[str]

    @field_validator("hashes")
    @classmethod
    def check_batch(cls, v):
        if len(v) > 500:
            raise ValueError("Max 500 hashes per batch")
        return v


class BatchResponse(BaseModel):
    results: list[DecodeResponse]
    total:   int
    found:   int


class CompareResponse(BaseModel):
    hash:          str
    lookup_key:    Optional[str]
    ml_key:        Optional[str]
    ml_confidence: Optional[float]
    agree:         Optional[bool]   # None اگه یکی از دو نتیجه موجود نباشه
    ms:            float


# ── Schemas جدید: Multi-mode ─────────────────────────────────────────

class MultiModeEntry(BaseModel):
    words:      int              # 12، 18، یا 24
    key:        Optional[str]    # عبارت seed (یا None اگه ML موفق نبود)
    confidence: float            # اطمینان مدل (1.0 برای lookup)


class MultiDecodeResponse(BaseModel):
    hash:    str
    source:  str                  # "lookup" | "ml" | "not_found"
    best:    Optional[MultiModeEntry]   # بهترین حالت (بالاترین confidence)
    modes:   list[MultiModeEntry]       # هر ۳ حالت (یا فقط یکی در صورت lookup)
    ms:      float


class StatusResponse(BaseModel):
    status:       str
    db_loaded:    bool
    model_loaded: bool


# ── Lifespan (startup / shutdown) ───────────────────────────────────

pipeline: Optional[DecoderPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    print("🚀 Loading pipeline...")
    pipeline = get_pipeline()
    print("✅ API ready")
    yield
    if pipeline:
        pipeline.close()
    print("👋 Shutdown complete")


# ── App ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="Hash Decoder API",
    description="رمزگشایی هش → seed phrase",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ───────────────────────────────────────────────────────

@app.get("/", response_model=StatusResponse)
def root():
    """وضعیت سرور"""
    return {
        "status":       "ok",
        "db_loaded":    pipeline._con    is not None if pipeline else False,
        "model_loaded": pipeline._model  is not None if pipeline else False,
    }


@app.post("/decode", response_model=DecodeResponse)
def decode(req: DecodeRequest):
    """
    رمزگشایی یه هش — اول lookup (سریع و ۱۰۰٪ دقیق)، اگه نبود ML

    مثال:
        POST /decode
        {"hash": "xK9mP2qRs3nT4vU5"}
    """
    if not pipeline:
        raise HTTPException(503, "Pipeline not initialized")
    result = pipeline.decode(req.hash)
    return DecodeResponse(hash=req.hash, **result)


@app.post("/batch", response_model=BatchResponse)
def batch_decode(req: BatchRequest):
    """
    رمزگشایی دسته‌ای — تا 500 هش در یه درخواست
    lookup تک‌تک با cache، ML به‌صورت batch واقعی (نه loop) برای سرعت بالاتر

    مثال:
        POST /batch
        {"hashes": ["abc...", "def..."]}
    """
    if not pipeline:
        raise HTTPException(503, "Pipeline not initialized")
    results = pipeline.batch_decode(req.hashes)
    responses = [
        DecodeResponse(hash=h, **r)
        for h, r in zip(req.hashes, results)
    ]
    found = sum(1 for r in responses if r.key is not None)
    return BatchResponse(results=responses, total=len(responses), found=found)


@app.post("/compare", response_model=CompareResponse)
def compare(req: DecodeRequest):
    """
    نتیجه lookup و ML رو همزمان برمی‌گردونه — برای QA و سنجش کیفیت مدل.
    وقتی هش هم در دیتابیس و هم در دسترس ML باشه، agree نشون می‌ده
    دو نتیجه با هم می‌خونن یا نه (معیار سلامت مدل روی داده واقعی).

    مثال:
        POST /compare
        {"hash": "xK9mP2qRs3nT4vU5"}
    """
    if not pipeline:
        raise HTTPException(503, "Pipeline not initialized")
    result = pipeline.compare(req.hash)
    return CompareResponse(hash=req.hash, **result)


@app.post("/decode/multi", response_model=MultiDecodeResponse)
def decode_multi(req: DecodeRequest):
    """
    رمزگشایی با نمایش هر ۳ حالت طول (12/18/24)

    - اگه هش در دیتابیس باشه → فوری با طول واقعی برمی‌گرده
    - اگه نباشه → مدل ML سه بار اجرا می‌شه (encoder یک بار، decoder سه بار)
    - بهترین حالت = بالاترین confidence

    مثال:
        POST /decode/multi
        {"hash": "TT7q9cR9odRkdaM2sKKWNAU66EM32ZEviZ"}

    Response:
        {
          "hash": "TT7q9cR9odRkdaM2sKKWNAU66EM32ZEviZ",
          "source": "ml",
          "best": {"words": 12, "key": "found distance faculty ...", "confidence": 0.87},
          "modes": [
            {"words": 12, "key": "found distance faculty ...", "confidence": 0.87},
            {"words": 18, "key": "found distance faculty ...", "confidence": 0.61},
            {"words": 24, "key": "found distance faculty ...", "confidence": 0.44}
          ],
          "ms": 142.5
        }
    """
    if not pipeline:
        raise HTTPException(503, "Pipeline not initialized")
    result = pipeline.multi_decode(req.hash)
    return MultiDecodeResponse(**result)


@app.get("/metrics")
def metrics():
    """
    آمار لحظه‌ای سرویس: تعداد درخواست‌ها به تفکیک منبع،
    نرخ توافق lookup/ML، و وضعیت cache
    """
    if not pipeline:
        raise HTTPException(503, "Pipeline not initialized")
    return {
        "requests": pipeline.stats.snapshot(),
        "cache":    pipeline.cache.stats(),
    }


@app.get("/health")
def health():
    """Health check برای Railway"""
    return {"status": "healthy"}
