"""
Qwen3-TTS server: voice cloning with the Qwen3-TTS Base model.

Backends (BACKEND env)
  faster   faster-qwen3-tts: CUDA-graph decode, typically 5-10x the stock speed (default)
  stock    the reference qwen_tts implementation

Endpoints
  GET  /                      test web UI
  GET  /livez                 process is up
  GET  /healthz               model loaded and ready (503 until then)
  GET  /v1/audio/voices       saved voices (OpenAI-TTS style, used by Home Assistant)
  POST /v1/voices             save a voice: multipart file + name + ref_text (+ language)
  DELETE /v1/voices/{name}    delete a saved voice
  GET  /v1/voices/{name}/reference   the stored reference clip
  POST /v1/audio/speech       OpenAI-compatible TTS with a saved voice
  POST /v1/audio/clone        one-shot clone: multipart file + ref_text + text (nothing saved)
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

# --------------------------------------------------------------------------- config

MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
BACKEND = os.getenv("BACKEND", "faster").lower()       # faster | stock
DTYPE = os.getenv("DTYPE", "bfloat16")
ATTN_IMPL = os.getenv("ATTN_IMPL", "sdpa")            # sdpa | flash_attention_2 | eager
DEVICE = os.getenv("DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
MAX_SEQ_LEN = int(os.getenv("MAX_SEQ_LEN", "2048"))   # faster backend: static KV-cache size (prompt + output tokens)
DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))
VOICES_DIR = DATA_DIR / "voices"
DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "Auto")
DEFAULT_VOICE = os.getenv("DEFAULT_VOICE", "")
MAX_REF_SECONDS = float(os.getenv("MAX_REF_SECONDS", "30"))
REF_TAIL_SILENCE = float(os.getenv("REF_TAIL_SILENCE", "0.5"))  # stops the clip's last phoneme bleeding into output
CHUNK_CHARS = int(os.getenv("CHUNK_CHARS", "350"))     # long input is split on sentences
CHUNK_GAP_MS = int(os.getenv("CHUNK_GAP_MS", "120"))
MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "8000"))

STATIC_DIR = Path(__file__).parent / "static"
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("qwen3-tts")

# --------------------------------------------------------------------------- model state
#
# All GPU work (loading, CUDA-graph capture, prompt building, generation) runs on ONE
# dedicated thread. CUDA graphs must be replayed consistently, and a single GPU can only
# usefully run one generation at a time anyway, so this doubles as the request queue.

_gpu = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu")
_engine = None        # object with generate_voice_clone(): FasterQwen3TTS or Qwen3TTSModel
_base = None          # the qwen_tts.Qwen3TTSModel (prompt building, languages)
_model_error: Optional[str] = None
_sample_rate = 24000
_prompt_cache: dict[str, object] = {}   # voice name -> voice_clone_prompt items


async def on_gpu(fn, *args):
    return await asyncio.get_running_loop().run_in_executor(_gpu, fn, *args)


def _load_model() -> None:
    global _engine, _base, _model_error, _sample_rate
    t0 = time.time()
    try:
        if BACKEND == "faster":
            from faster_qwen3_tts import FasterQwen3TTS

            log.info("loading %s with faster-qwen3-tts on %s (dtype=%s, attn=%s, max_seq_len=%d)",
                     MODEL_ID, DEVICE, DTYPE, ATTN_IMPL, MAX_SEQ_LEN)
            _engine = FasterQwen3TTS.from_pretrained(
                MODEL_ID, device=DEVICE, dtype=getattr(torch, DTYPE),
                attn_implementation=ATTN_IMPL, max_seq_len=MAX_SEQ_LEN,
            )
            _base = _engine.model
            log.info("capturing CUDA graphs")
            _engine.warmup()
        elif BACKEND == "stock":
            from qwen_tts import Qwen3TTSModel

            log.info("loading %s with stock qwen_tts on %s (dtype=%s, attn=%s)", MODEL_ID, DEVICE, DTYPE, ATTN_IMPL)
            _engine = Qwen3TTSModel.from_pretrained(
                MODEL_ID, device_map=DEVICE, dtype=getattr(torch, DTYPE), attn_implementation=ATTN_IMPL,
            )
            _base = _engine
        else:
            raise ValueError(f"BACKEND must be 'faster' or 'stock', got {BACKEND!r}")

        if getattr(_base.model, "tts_model_type", "base") != "base":
            raise RuntimeError(f"{MODEL_ID} is not a Base (voice clone) model")

        # one short real generation so the first request isn't slow
        warm = (np.full(24000, 1e-4, dtype=np.float32), 24000)
        with torch.inference_mode():
            prompt = _base.create_voice_clone_prompt(ref_audio=warm, x_vector_only_mode=True)
            _, _sample_rate = _call_engine("Ready.", "English", prompt, {"max_new_tokens": 32})
        vram = f", {torch.cuda.memory_allocated() / 1024**3:.2f} GB VRAM allocated" if torch.cuda.is_available() else ""
        log.info("model ready (%s backend) in %.1fs%s", BACKEND, time.time() - t0, vram)
    except Exception as e:  # keep the process alive so /healthz can report the error
        _engine = _base = None
        _model_error = f"{type(e).__name__}: {e}"
        log.exception("model failed to load")


def _call_engine(text: str, language: str, prompt, kw: dict):
    """One generation call, adapting sampling kwargs to the backend."""
    if BACKEND == "faster":
        # faster-qwen3-tts takes these directly; its sub-talker sampling is fixed at graph capture
        allowed = ("temperature", "top_p", "top_k", "repetition_penalty", "max_new_tokens")
        args = {k: v for k, v in kw.items() if k in allowed}
        if "max_new_tokens" in args:
            args["max_new_tokens"] = min(args["max_new_tokens"], MAX_SEQ_LEN)
        wavs, sr = _engine.generate_voice_clone(text=text, language=language, voice_clone_prompt=prompt, **args)
    else:
        args = dict(kw)
        # keep the sub-talker (residual codebooks) in step with the main temperature/top_p
        if "temperature" in args:
            args["subtalker_temperature"] = args["temperature"]
        if "top_p" in args:
            args["subtalker_top_p"] = args["top_p"]
        wavs, sr = _engine.generate_voice_clone(text=text, language=language, voice_clone_prompt=prompt, **args)
    return np.asarray(wavs[0], dtype=np.float32).reshape(-1), int(sr)


# --------------------------------------------------------------------------- voices on disk


def _voice_dir(name: str) -> Path:
    if not NAME_RE.match(name):
        raise HTTPException(400, "voice name: letters, digits, '-', '_', '.' only (max 64)")
    return VOICES_DIR / name


def _read_meta(name: str) -> dict:
    meta_path = _voice_dir(name) / "meta.json"
    if not meta_path.exists():
        raise HTTPException(404, f"voice '{name}' not found")
    return json.loads(meta_path.read_text())


def _list_voices() -> list[dict]:
    out = []
    if VOICES_DIR.exists():
        for d in sorted(VOICES_DIR.iterdir()):
            m = d / "meta.json"
            if d.is_dir() and m.exists():
                try:
                    out.append(json.loads(m.read_text()))
                except Exception:
                    log.warning("bad meta.json in %s", d)
    return out


def _to_ref_wav(src: Path, dst: Path) -> float:
    """Convert any audio ffmpeg can read to 24 kHz mono 16-bit WAV. Returns duration in seconds."""
    cmd = ["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", str(src),
           "-ac", "1", "-ar", "24000", "-sample_fmt", "s16", str(dst)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise HTTPException(400, f"could not decode audio: {r.stderr.strip()[:300]}")
    info = sf.info(str(dst))
    return info.frames / info.samplerate


async def _save_upload(upload: UploadFile, workdir: Path) -> Path:
    raw = workdir / ("upload" + Path(upload.filename or "audio").suffix)
    with raw.open("wb") as f:
        shutil.copyfileobj(upload.file, f)
    if raw.stat().st_size == 0:
        raise HTTPException(400, "empty audio file")
    return raw


def _build_prompt(ref_wav: str, ref_text: str):
    """Voice-clone prompt from a reference clip. Runs on the GPU thread."""
    audio, sr = sf.read(ref_wav, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    icl = bool(ref_text)
    if icl and REF_TAIL_SILENCE > 0:
        audio = np.concatenate([audio, np.zeros(int(REF_TAIL_SILENCE * sr), dtype=np.float32)])
    with torch.inference_mode():
        return _base.create_voice_clone_prompt(
            ref_audio=(audio, int(sr)), ref_text=ref_text or None, x_vector_only_mode=not icl,
        )


def _get_prompt(name: str):
    """Cached prompt for a saved voice. Runs on the GPU thread."""
    if name not in _prompt_cache:
        meta = _read_meta(name)
        _prompt_cache[name] = _build_prompt(str(_voice_dir(name) / "ref.wav"), meta.get("ref_text", ""))
    return _prompt_cache[name]


# --------------------------------------------------------------------------- generation


def _split_text(text: str, limit: int) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return [text]
    sentences = re.split(r"(?<=[.!?。！？;])\s+", text)
    chunks, cur = [], ""
    for s in sentences:
        while len(s) > limit:  # a single very long sentence: break on commas, then hard
            cut = s.rfind(",", 0, limit)
            cut = cut + 1 if cut > limit // 3 else limit
            chunks.append(s[:cut].strip())
            s = s[cut:].strip()
        if s and not re.search(r"\w", s) and (cur or chunks):   # stray punctuation left by a hard cut
            if cur:
                cur += s
            else:
                chunks[-1] += s
            continue
        if cur and len(cur) + 1 + len(s) > limit:
            chunks.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        chunks.append(cur)
    return [c for c in chunks if c]


class GenParams(BaseModel):
    temperature: Optional[float] = Field(None, ge=0.05, le=2.0)
    top_p: Optional[float] = Field(None, gt=0.0, le=1.0)
    top_k: Optional[int] = Field(None, ge=1, le=500)
    repetition_penalty: Optional[float] = Field(None, ge=1.0, le=2.0)
    max_new_tokens: Optional[int] = Field(None, ge=16, le=4096)
    seed: Optional[int] = None

    def kwargs(self) -> dict:
        fields = ("temperature", "top_p", "top_k", "repetition_penalty", "max_new_tokens")
        return {k: getattr(self, k) for k in fields if getattr(self, k) is not None}


def _generate(text: str, language: str, params: GenParams, prompt) -> np.ndarray:
    """Runs on the GPU thread."""
    if _engine is None:
        raise HTTPException(503, _model_error or "model is still loading")
    chunks = _split_text(text, CHUNK_CHARS)
    gap = np.zeros(int(_sample_rate * CHUNK_GAP_MS / 1000), dtype=np.float32)
    if params.seed is not None:
        torch.manual_seed(params.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(params.seed)
    t0 = time.time()
    pieces: list[np.ndarray] = []
    with torch.inference_mode():
        for i, chunk in enumerate(chunks):
            wav, _ = _call_engine(chunk, language, prompt, params.kwargs())
            if i:
                pieces.append(gap)
            pieces.append(wav)
    audio = np.concatenate(pieces) if pieces else np.zeros(1, dtype=np.float32)
    secs, took = len(audio) / _sample_rate, time.time() - t0
    log.info("generated %.1fs audio in %.1fs (%.2fx real time, %d chunk%s)", secs, took,
             secs / took if took else 0, len(chunks), "" if len(chunks) == 1 else "s")
    return audio


def _clone_once(ref_wav: str, ref_text: str, text: str, language: str, params: GenParams) -> np.ndarray:
    """Runs on the GPU thread."""
    if _engine is None:
        raise HTTPException(503, _model_error or "model is still loading")
    return _generate(text, language, params, _build_prompt(ref_wav, ref_text))


MEDIA = {
    "wav": "audio/wav", "mp3": "audio/mpeg", "flac": "audio/flac",
    "opus": "audio/ogg", "aac": "audio/aac", "pcm": "audio/L16",
}


def _encode(audio: np.ndarray, fmt: str, speed: float) -> bytes:
    fmt = fmt.lower()
    if fmt not in MEDIA:
        raise HTTPException(400, f"response_format must be one of {sorted(MEDIA)}")
    audio = np.clip(audio, -1.0, 1.0)
    buf = io.BytesIO()
    sf.write(buf, audio, _sample_rate, format="WAV", subtype="PCM_16")
    wav = buf.getvalue()
    if fmt == "wav" and abs(speed - 1.0) < 1e-3:
        return wav
    # atempo accepts 0.5..2.0 per stage; chain stages for anything beyond that
    filters, s = [], max(0.25, min(4.0, speed))
    while s > 2.0:
        filters.append("atempo=2.0"); s /= 2.0
    while s < 0.5:
        filters.append("atempo=0.5"); s /= 0.5
    if abs(s - 1.0) > 1e-3:
        filters.append(f"atempo={s:.4f}")
    out = {
        "wav": ["-f", "wav"], "mp3": ["-f", "mp3", "-b:a", "128k"], "flac": ["-f", "flac"],
        "opus": ["-f", "ogg", "-c:a", "libopus", "-b:a", "64k"], "aac": ["-f", "adts", "-c:a", "aac", "-b:a", "128k"],
        "pcm": ["-f", "s16le", "-ac", "1", "-ar", str(_sample_rate)],
    }[fmt]
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-f", "wav", "-i", "pipe:0"]
    if filters:
        cmd += ["-filter:a", ",".join(filters)]
    cmd += out + ["pipe:1"]
    r = subprocess.run(cmd, input=wav, capture_output=True)
    if r.returncode != 0:
        raise HTTPException(500, f"encoding failed: {r.stderr.decode()[:300]}")
    return r.stdout


# --------------------------------------------------------------------------- app

app = FastAPI(title="Qwen3-TTS", version="1.1")


@app.on_event("startup")
def _startup() -> None:
    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    _gpu.submit(_load_model)


@app.get("/livez")
def livez():
    return {"ok": True}


@app.get("/healthz")
def healthz():
    if _engine is not None:
        info = {"ok": True, "model": MODEL_ID, "backend": BACKEND, "device": DEVICE}
        if torch.cuda.is_available():
            info["vram_allocated_gb"] = round(torch.cuda.memory_allocated() / 1024**3, 2)
        return info
    status = 500 if _model_error else 503
    return JSONResponse({"ok": False, "backend": BACKEND, "error": _model_error or "loading"}, status_code=status)


@app.get("/v1/info")
def info():
    langs = []
    if _base is not None:
        try:
            langs = sorted(_base.model.get_supported_languages() or [])
        except Exception:
            pass
    return {"model": MODEL_ID, "backend": BACKEND, "ready": _engine is not None, "error": _model_error,
            "languages": ["Auto", *[str(l).capitalize() for l in langs if str(l).lower() != "auto"]],
            "default_voice": DEFAULT_VOICE, "formats": sorted(MEDIA)}


@app.get("/v1/audio/voices")
@app.get("/v1/voices")
def voices():
    v = _list_voices()
    return {"voices": [x["name"] for x in v], "details": v}


@app.post("/v1/voices")
async def create_voice(
    name: str = Form(...),
    file: UploadFile = File(...),
    ref_text: str = Form(""),
    language: str = Form("Auto"),
    overwrite: bool = Form(False),
):
    d = _voice_dir(name)
    if d.exists() and not overwrite:
        raise HTTPException(409, f"voice '{name}' already exists (send overwrite=true to replace)")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        raw = await _save_upload(file, tmp)
        wav = tmp / "ref.wav"
        dur = await run_in_threadpool(_to_ref_wav, raw, wav)
        if dur < 1.5:
            raise HTTPException(400, f"reference is only {dur:.1f}s; use 5–20 s of clean speech")
        if dur > MAX_REF_SECONDS:
            raise HTTPException(400, f"reference is {dur:.1f}s; keep it under {MAX_REF_SECONDS:.0f}s")
        d.mkdir(parents=True, exist_ok=True)
        shutil.move(str(wav), d / "ref.wav")
    meta = {"name": name, "ref_text": ref_text.strip(), "language": language,
            "mode": "icl" if ref_text.strip() else "x_vector_only",
            "duration_s": round(dur, 2), "created": int(time.time())}
    (d / "meta.json").write_text(json.dumps(meta, indent=2))
    _prompt_cache.pop(name, None)
    if _engine is not None:
        await on_gpu(_get_prompt, name)   # validate + warm the cache now
    return meta


@app.delete("/v1/voices/{name}")
def delete_voice(name: str):
    d = _voice_dir(name)
    if not d.exists():
        raise HTTPException(404, f"voice '{name}' not found")
    shutil.rmtree(d)
    _prompt_cache.pop(name, None)
    return {"deleted": name}


@app.get("/v1/voices/{name}/reference")
def voice_reference(name: str):
    p = _voice_dir(name) / "ref.wav"
    if not p.exists():
        raise HTTPException(404, f"voice '{name}' not found")
    return FileResponse(p, media_type="audio/wav")


class SpeechRequest(GenParams):
    input: str = Field(..., min_length=1)
    voice: Optional[str] = None
    model: Optional[str] = None             # accepted for OpenAI compatibility, ignored
    response_format: str = "mp3"
    speed: float = Field(1.0, ge=0.25, le=4.0)
    language: Optional[str] = None
    instructions: Optional[str] = None      # accepted for OpenAI compatibility, ignored (Base model)


@app.post("/v1/audio/speech")
async def speech(req: SpeechRequest):
    if len(req.input) > MAX_INPUT_CHARS:
        raise HTTPException(413, f"input is over {MAX_INPUT_CHARS} characters")
    name = req.voice or DEFAULT_VOICE
    if not name:
        v = _list_voices()
        if not v:
            raise HTTPException(400, "no voices saved yet: POST /v1/voices first")
        name = v[0]["name"]
    if _engine is None:
        raise HTTPException(503, _model_error or "model is still loading")
    meta = _read_meta(name)
    language = req.language or meta.get("language") or DEFAULT_LANGUAGE
    prompt = await on_gpu(_get_prompt, name)
    audio = await on_gpu(_generate, req.input, language, req, prompt)
    data = await run_in_threadpool(_encode, audio, req.response_format, req.speed)
    return Response(data, media_type=MEDIA[req.response_format.lower()],
                    headers={"Content-Disposition": f'inline; filename="speech.{req.response_format.lower()}"'})


@app.post("/v1/audio/clone")
async def clone_once(
    file: UploadFile = File(...),
    text: str = Form(...),
    ref_text: str = Form(""),
    language: str = Form(DEFAULT_LANGUAGE),
    response_format: str = Form("wav"),
    speed: float = Form(1.0),
    temperature: Optional[float] = Form(None),
    top_p: Optional[float] = Form(None),
    top_k: Optional[int] = Form(None),
    repetition_penalty: Optional[float] = Form(None),
    seed: Optional[int] = Form(None),
):
    if len(text) > MAX_INPUT_CHARS:
        raise HTTPException(413, f"text is over {MAX_INPUT_CHARS} characters")
    params = GenParams(temperature=temperature, top_p=top_p, top_k=top_k,
                       repetition_penalty=repetition_penalty, seed=seed)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        raw = await _save_upload(file, tmp)
        wav = tmp / "ref.wav"
        dur = await run_in_threadpool(_to_ref_wav, raw, wav)
        if dur > MAX_REF_SECONDS:
            raise HTTPException(400, f"reference is {dur:.1f}s; keep it under {MAX_REF_SECONDS:.0f}s")
        audio = await on_gpu(_clone_once, str(wav), ref_text.strip(), text, language, params)
    data = await run_in_threadpool(_encode, audio, response_format, speed)
    return Response(data, media_type=MEDIA[response_format.lower()])


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
