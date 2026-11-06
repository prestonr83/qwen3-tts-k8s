"""
Qwen3-TTS engine worker. Holds the model on the GPU and does nothing else.

The web server (server.py) starts one worker at a time, in one of two Python environments:
  fast      faster-qwen3-tts (CUDA-graph decoding) on qwen-tts-hf / transformers 5
  original  the reference qwen-tts 0.1.1 on transformers 4.57.3 (the first deployment, unchanged)

Switching engines stops this process, which frees all of its VRAM, and starts the other one.

Endpoints (127.0.0.1 only)
  GET  /health     {ready, error, engine, versions, vram}
  POST /generate   JSON {ref_wav, ref_text, text, language, params, seed} -> audio/wav (PCM16)
  POST /prompt     JSON {ref_wav, ref_text} -> warm the voice-prompt cache
"""

from __future__ import annotations

import argparse
import asyncio
import io
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

ENGINE = os.getenv("ENGINE", "fast")
MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
DTYPE = os.getenv("DTYPE", "bfloat16")
ATTN_IMPL = os.getenv("ATTN_IMPL", "sdpa")
DEVICE = os.getenv("DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
MAX_SEQ_LEN = int(os.getenv("MAX_SEQ_LEN", "2048"))
# fast engine only: faster-qwen3-tts pads the reference with 0.5 s of silence so its last
# sound doesn't bleed into the output. The original engine passes the clip unchanged.
REF_TAIL_SILENCE = float(os.getenv("REF_TAIL_SILENCE", "0.5"))

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
                    format=f"%(asctime)s %(levelname)s [{ENGINE}] %(message)s")
log = logging.getLogger("qwen3-tts-worker")

_gpu = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu")   # all CUDA work on one thread
_engine = None     # has generate_voice_clone()
_base = None       # qwen_tts.Qwen3TTSModel (prompt building, languages)
_error: Optional[str] = None
_sample_rate = 24000
_versions: dict = {}
_prompt_cache: dict = {}   # (path, mtime, ref_text) -> prompt items


def _load() -> None:
    global _engine, _base, _error, _sample_rate, _versions
    t0 = time.time()
    try:
        import transformers

        _versions = {"torch": torch.__version__, "transformers": transformers.__version__}
        if ENGINE == "fast":
            import qwen_compat

            qwen_compat.apply()
            import faster_qwen3_tts
            from faster_qwen3_tts import FasterQwen3TTS

            _versions["faster-qwen3-tts"] = getattr(faster_qwen3_tts, "__version__", "?")
            log.info("loading %s with faster-qwen3-tts on %s (dtype=%s, attn=%s, max_seq_len=%d)",
                     MODEL_ID, DEVICE, DTYPE, ATTN_IMPL, MAX_SEQ_LEN)
            _engine = FasterQwen3TTS.from_pretrained(
                MODEL_ID, device=DEVICE, dtype=getattr(torch, DTYPE),
                attn_implementation=ATTN_IMPL, max_seq_len=MAX_SEQ_LEN,
            )
            _base = _engine.model
            log.info("capturing CUDA graphs")
            _engine.warmup()
        elif ENGINE == "original":
            from importlib.metadata import PackageNotFoundError, version

            from qwen_tts import Qwen3TTSModel

            try:
                _versions["qwen-tts"] = version("qwen-tts")
            except PackageNotFoundError:
                _versions["qwen-tts"] = "?"
            log.info("loading %s with qwen-tts %s on %s (dtype=%s, attn=%s)",
                     MODEL_ID, _versions["qwen-tts"], DEVICE, DTYPE, ATTN_IMPL)
            _engine = Qwen3TTSModel.from_pretrained(
                MODEL_ID, device_map=DEVICE, dtype=getattr(torch, DTYPE), attn_implementation=ATTN_IMPL,
            )
            _base = _engine
        else:
            raise ValueError(f"ENGINE must be 'fast' or 'original', got {ENGINE!r}")

        if getattr(_base.model, "tts_model_type", "base") != "base":
            raise RuntimeError(f"{MODEL_ID} is not a Base (voice clone) model")

        warm = (np.full(24000, 1e-4, dtype=np.float32), 24000)
        with torch.inference_mode():
            prompt = _base.create_voice_clone_prompt(ref_audio=warm, x_vector_only_mode=True)
            _, _sample_rate = _call("Ready.", "English", prompt, {"max_new_tokens": 32})
        vram = f", {torch.cuda.memory_allocated() / 1024**3:.2f} GB VRAM" if torch.cuda.is_available() else ""
        log.info("model ready in %.1fs%s %s", time.time() - t0, vram, _versions)
    except Exception as e:
        _engine = _base = None
        _error = f"{type(e).__name__}: {e}"
        log.exception("model failed to load")


def _call(text: str, language: str, prompt, kw: dict):
    if ENGINE == "fast":
        allowed = ("temperature", "top_p", "top_k", "repetition_penalty", "max_new_tokens")
        args = {k: v for k, v in kw.items() if k in allowed}
        if "max_new_tokens" in args:
            args["max_new_tokens"] = min(args["max_new_tokens"], MAX_SEQ_LEN)
    else:
        # same as the first deployment: sub-talker sampling follows the main settings
        args = dict(kw)
        if "temperature" in args:
            args["subtalker_temperature"] = args["temperature"]
        if "top_p" in args:
            args["subtalker_top_p"] = args["top_p"]
    wavs, sr = _engine.generate_voice_clone(text=text, language=language, voice_clone_prompt=prompt, **args)
    return np.asarray(wavs[0], dtype=np.float32).reshape(-1), int(sr)


def _prompt(ref_wav: str, ref_text: str):
    key = (ref_wav, os.path.getmtime(ref_wav), ref_text)
    if key in _prompt_cache:
        return _prompt_cache[key]
    icl = bool(ref_text)
    with torch.inference_mode():
        if ENGINE == "fast":
            audio, sr = sf.read(ref_wav, dtype="float32", always_2d=False)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if icl and REF_TAIL_SILENCE > 0:
                audio = np.concatenate([audio, np.zeros(int(REF_TAIL_SILENCE * sr), dtype=np.float32)])
            ref = (audio, int(sr))
        else:
            ref = ref_wav   # exactly what the first deployment passed
        items = _base.create_voice_clone_prompt(ref_audio=ref, ref_text=ref_text or None, x_vector_only_mode=not icl)
    if len(_prompt_cache) > 64:
        _prompt_cache.clear()
    _prompt_cache[key] = items
    return items


def _split(text: str, limit: int) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return [text]
    chunks, cur = [], ""
    for s in re.split(r"(?<=[.!?。！？;])\s+", text):
        while len(s) > limit:
            cut = s.rfind(",", 0, limit)
            cut = cut + 1 if cut > limit // 3 else limit
            chunks.append(s[:cut].strip())
            s = s[cut:].strip()
        if s and not re.search(r"\w", s) and (cur or chunks):
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


def _generate(req: "GenRequest") -> tuple[np.ndarray, float]:
    if _engine is None:
        raise HTTPException(503, _error or "model is still loading")
    prompt = _prompt(req.ref_wav, req.ref_text)
    if req.seed is not None:
        torch.manual_seed(req.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(req.seed)
    chunks = _split(req.text, req.chunk_chars)
    gap = np.zeros(int(_sample_rate * req.chunk_gap_ms / 1000), dtype=np.float32)
    t0 = time.time()
    pieces: list[np.ndarray] = []
    with torch.inference_mode():
        for i, chunk in enumerate(chunks):
            wav, _ = _call(chunk, req.language, prompt, req.params)
            if i:
                pieces.append(gap)
            pieces.append(wav)
    audio = np.concatenate(pieces) if pieces else np.zeros(1, dtype=np.float32)
    took = time.time() - t0
    secs = len(audio) / _sample_rate
    log.info("generated %.1fs audio in %.1fs (%.2fx real time, %d chunk%s)", secs, took,
             secs / took if took else 0, len(chunks), "" if len(chunks) == 1 else "s")
    return audio, took


async def _on_gpu(fn, *args):
    return await asyncio.get_running_loop().run_in_executor(_gpu, fn, *args)


app = FastAPI(title="Qwen3-TTS worker")


@app.on_event("startup")
def _startup():
    _gpu.submit(_load)


class GenRequest(BaseModel):
    ref_wav: str
    ref_text: str = ""
    text: str
    language: str = "Auto"
    params: dict = {}
    seed: Optional[int] = None
    chunk_chars: int = 350
    chunk_gap_ms: int = 120


class PromptRequest(BaseModel):
    ref_wav: str
    ref_text: str = ""


@app.get("/health")
def health():
    body = {"ready": _engine is not None, "error": _error, "engine": ENGINE, "model": MODEL_ID,
            "versions": _versions, "sample_rate": _sample_rate}
    if torch.cuda.is_available():
        body["vram_gb"] = round(torch.cuda.memory_allocated() / 1024**3, 2)
    if _base is not None:
        try:
            body["languages"] = sorted(str(l) for l in (_base.model.get_supported_languages() or []))
        except Exception:
            pass
    return JSONResponse(body)


@app.post("/prompt")
async def warm_prompt(req: PromptRequest):
    if _engine is None:
        raise HTTPException(503, _error or "model is still loading")
    await _on_gpu(_prompt, req.ref_wav, req.ref_text)
    return {"ok": True}


@app.post("/generate")
async def generate(req: GenRequest):
    audio, took = await _on_gpu(_generate, req)
    buf = io.BytesIO()
    sf.write(buf, np.clip(audio, -1.0, 1.0), _sample_rate, format="WAV", subtype="PCM_16")
    return Response(buf.getvalue(), media_type="audio/wav",
                    headers={"X-Gen-Seconds": f"{took:.3f}", "X-Audio-Seconds": f"{len(audio) / _sample_rate:.3f}"})


if __name__ == "__main__":
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8101)
    a = ap.parse_args()
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")
