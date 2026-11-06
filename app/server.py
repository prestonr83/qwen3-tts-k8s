"""
Qwen3-TTS web server: UI, voices, presets, saved generations, OpenAI-compatible API.

The model itself runs in a separate engine worker process (worker.py), one at a time:
  fast      faster-qwen3-tts (CUDA graphs)            /opt/engines/fast
  original  qwen-tts 0.1.1 on transformers 4.57.3     /opt/engines/original
Switching stops the running worker (freeing its VRAM) and starts the other.

Endpoints
  GET  /                              web UI
  GET  /livez                         server process is up (used by all k8s probes)
  GET  /healthz                       200 once the active engine is ready
  GET  /v1/engine                     active engine and its status
  POST /v1/engine                     {"engine": "fast" | "original"}: switch
  GET  /v1/audio/voices               saved voices (OpenAI-TTS style, used by Home Assistant)
  POST /v1/voices                     save a voice: multipart file + name + ref_text (+ language)
  DELETE /v1/voices/{name}            delete a voice
  GET  /v1/voices/{name}/reference    the stored reference clip
  POST /v1/audio/speech               OpenAI-compatible TTS with a saved voice ("save": true keeps it)
  POST /v1/audio/clone                one-shot clone: multipart file + ref_text + text
  GET  /v1/generations                saved generations, newest first
  GET  /v1/generations/{id}/audio     a saved generation's WAV
  DELETE /v1/generations/{id}         delete one ("all" deletes every one)
  GET  /v1/presets                    saved presets
  PUT  /v1/presets/{name}             save a preset (voice, language, engine, sliders, seed, format)
  DELETE /v1/presets/{name}           delete a preset
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional

import httpx
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

# --------------------------------------------------------------------------- config

ENGINES = {
    "fast": {"label": "Fast", "python": os.getenv("ENGINE_PYTHON_FAST", "/opt/engines/fast/bin/python"),
             "desc": "faster-qwen3-tts, CUDA graphs"},
    "original": {"label": "Original", "python": os.getenv("ENGINE_PYTHON_ORIGINAL", "/opt/engines/original/bin/python"),
                 "desc": "qwen-tts 0.1.1, transformers 4.57.3"},
}
DEFAULT_ENGINE = os.getenv("ENGINE", "fast")
WORKER_PORT = int(os.getenv("WORKER_PORT", "8101"))
WORKER_URL = f"http://127.0.0.1:{WORKER_PORT}"
MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))
VOICES_DIR = DATA_DIR / "voices"
GEN_DIR = DATA_DIR / "generations"
PRESETS_FILE = DATA_DIR / "presets.json"
STATE_FILE = DATA_DIR / "state.json"
DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "Auto")
DEFAULT_VOICE = os.getenv("DEFAULT_VOICE", "")
MAX_REF_SECONDS = float(os.getenv("MAX_REF_SECONDS", "30"))
CHUNK_CHARS = int(os.getenv("CHUNK_CHARS", "350"))
CHUNK_GAP_MS = int(os.getenv("CHUNK_GAP_MS", "120"))
MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "8000"))
MAX_GENERATIONS = int(os.getenv("MAX_GENERATIONS", "200"))
LOAD_TIMEOUT = float(os.getenv("ENGINE_LOAD_TIMEOUT", "1800"))

STATIC_DIR = Path(__file__).parent / "static"
WORKER_SCRIPT = Path(__file__).parent / "worker.py"
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
PRESET_RE = re.compile(r"^[^/\\]{1,64}$")
GEN_ID_RE = re.compile(r"^[0-9a-f]{12}$")
LANGUAGES = ["Auto", "English", "Chinese", "Japanese", "Korean", "German", "French",
             "Russian", "Portuguese", "Spanish", "Italian"]

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s [server] %(message)s")
log = logging.getLogger("qwen3-tts")
logging.getLogger("httpx").setLevel(logging.WARNING)


# --------------------------------------------------------------------------- engine manager


class EngineManager:
    def __init__(self) -> None:
        self.engine: str = DEFAULT_ENGINE
        self.status: str = "stopped"      # stopped | starting | ready | error
        self.error: Optional[str] = None
        self.health: dict = {}
        self.proc: Optional[subprocess.Popen] = None
        self.started_at = 0.0
        self.lock = asyncio.Lock()
        self._watch: Optional[asyncio.Task] = None
        self.client = httpx.AsyncClient(base_url=WORKER_URL, timeout=httpx.Timeout(None, connect=5.0))

    def snapshot(self) -> dict:
        return {"engine": self.engine, "status": self.status, "error": self.error,
                "loading_seconds": round(time.time() - self.started_at) if self.status == "starting" else None,
                "versions": self.health.get("versions", {}), "vram_gb": self.health.get("vram_gb"),
                "model": MODEL_ID,
                "available": [{"id": k, "label": v["label"], "desc": v["desc"],
                               "installed": Path(v["python"]).exists()} for k, v in ENGINES.items()]}

    async def start(self, engine: str) -> None:
        async with self.lock:
            if engine not in ENGINES:
                raise HTTPException(400, f"engine must be one of {sorted(ENGINES)}")
            py = ENGINES[engine]["python"]
            if not Path(py).exists():
                raise HTTPException(400, f"engine '{engine}' is not installed in this image ({py})")
            await self._stop()
            self.engine, self.status, self.error, self.health = engine, "starting", None, {}
            self.started_at = time.time()
            env = dict(os.environ, ENGINE=engine, PYTHONUNBUFFERED="1")
            log.info("starting %s engine", engine)
            self.proc = subprocess.Popen([py, str(WORKER_SCRIPT), "--port", str(WORKER_PORT)],
                                         env=env, cwd=str(WORKER_SCRIPT.parent))
            _write_json(STATE_FILE, {"engine": engine})
            self._watch = asyncio.create_task(self._watch_loop(self.proc))

    async def _stop(self) -> None:
        if self._watch:
            self._watch.cancel()
            self._watch = None
        p, self.proc = self.proc, None
        if p and p.poll() is None:
            log.info("stopping %s engine (pid %d)", self.engine, p.pid)
            p.send_signal(signal.SIGTERM)
            try:
                await run_in_threadpool(p.wait, 20)
            except subprocess.TimeoutExpired:
                p.kill()
                await run_in_threadpool(p.wait)
        self.status = "stopped"

    async def _watch_loop(self, proc: subprocess.Popen) -> None:
        while True:
            await asyncio.sleep(1.0 if self.status == "starting" else 5.0)
            if proc is not self.proc:
                return
            code = proc.poll()
            if code is not None:
                self.status, self.error = "error", f"{self.engine} engine exited (code {code}); see pod logs"
                log.error(self.error)
                return
            try:
                h = (await self.client.get("/health", timeout=5.0)).json()
            except Exception:
                if self.status == "starting" and time.time() - self.started_at > LOAD_TIMEOUT:
                    self.status, self.error = "error", "engine did not come up in time"
                    return
                continue
            self.health = h
            if h.get("error"):
                self.status, self.error = "error", h["error"]
            elif h.get("ready"):
                if self.status != "ready":
                    log.info("%s engine ready after %.0fs", self.engine, time.time() - self.started_at)
                self.status, self.error = "ready", None

    def require_ready(self) -> None:
        if self.status == "ready":
            return
        if self.status == "starting":
            raise HTTPException(503, f"{ENGINES[self.engine]['label']} engine is loading, try again shortly")
        raise HTTPException(503, self.error or "engine is not running")

    async def generate(self, ref_wav: Path, ref_text: str, text: str, language: str, params: dict,
                       seed: Optional[int]) -> tuple[bytes, dict]:
        self.require_ready()
        body = {"ref_wav": str(ref_wav), "ref_text": ref_text, "text": text, "language": language,
                "params": params, "seed": seed, "chunk_chars": CHUNK_CHARS, "chunk_gap_ms": CHUNK_GAP_MS}
        try:
            r = await self.client.post("/generate", json=body)
        except httpx.HTTPError as e:
            raise HTTPException(503, f"engine unavailable: {e}")
        if r.status_code != 200:
            try:
                detail = r.json().get("detail", r.text)
            except Exception:
                detail = r.text
            raise HTTPException(r.status_code, detail)
        return r.content, {"gen_seconds": float(r.headers.get("x-gen-seconds", 0)),
                           "audio_seconds": float(r.headers.get("x-audio-seconds", 0)),
                           "engine": self.engine}

    async def warm(self, ref_wav: Path, ref_text: str) -> None:
        if self.status != "ready":
            return
        r = await self.client.post("/prompt", json={"ref_wav": str(ref_wav), "ref_text": ref_text})
        if r.status_code != 200:
            raise HTTPException(400, f"could not build a voice prompt from this clip: {r.text[:300]}")


engines = EngineManager()


# --------------------------------------------------------------------------- storage helpers


def _write_json(path: Path, data) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return default
    except Exception:
        log.warning("unreadable %s", path)
        return default


def _voice_dir(name: str) -> Path:
    if not NAME_RE.match(name):
        raise HTTPException(400, "voice name: letters, digits, '-', '_', '.' only (max 64)")
    return VOICES_DIR / name


def _read_meta(name: str) -> dict:
    meta = _read_json(_voice_dir(name) / "meta.json", None)
    if meta is None:
        raise HTTPException(404, f"voice '{name}' not found")
    return meta


def _list_voices() -> list[dict]:
    out = []
    if VOICES_DIR.exists():
        for d in sorted(VOICES_DIR.iterdir(), key=lambda p: p.name.lower()):
            meta = _read_json(d / "meta.json", None) if d.is_dir() else None
            if meta:
                out.append(meta)
    return out


def _to_ref_wav(src: Path, dst: Path) -> float:
    cmd = ["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", str(src),
           "-ac", "1", "-ar", "24000", "-sample_fmt", "s16", str(dst)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise HTTPException(400, f"could not decode audio: {r.stderr.strip()[:300]}")
    info = sf.info(str(dst))
    return info.frames / info.samplerate


async def _save_upload(upload: UploadFile, workdir: Path) -> Path:
    raw = workdir / ("upload" + Path(upload.filename or "audio").suffix[:10])
    with raw.open("wb") as f:
        shutil.copyfileobj(upload.file, f)
    if raw.stat().st_size == 0:
        raise HTTPException(400, "empty audio file")
    return raw


MEDIA = {"wav": "audio/wav", "mp3": "audio/mpeg", "flac": "audio/flac",
         "opus": "audio/ogg", "aac": "audio/aac", "pcm": "audio/L16"}


def _encode(wav: bytes, fmt: str, speed: float) -> bytes:
    fmt = fmt.lower()
    if fmt not in MEDIA:
        raise HTTPException(400, f"response_format must be one of {sorted(MEDIA)}")
    if fmt == "wav" and abs(speed - 1.0) < 1e-3:
        return wav
    filters, s = [], max(0.25, min(4.0, speed))
    while s > 2.0:
        filters.append("atempo=2.0"); s /= 2.0
    while s < 0.5:
        filters.append("atempo=0.5"); s /= 0.5
    if abs(s - 1.0) > 1e-3:
        filters.append(f"atempo={s:.4f}")
    out = {"wav": ["-f", "wav"], "mp3": ["-f", "mp3", "-b:a", "128k"], "flac": ["-f", "flac"],
           "opus": ["-f", "ogg", "-c:a", "libopus", "-b:a", "64k"],
           "aac": ["-f", "adts", "-c:a", "aac", "-b:a", "128k"],
           "pcm": ["-f", "s16le", "-ac", "1", "-ar", "24000"]}[fmt]
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-f", "wav", "-i", "pipe:0"]
    if filters:
        cmd += ["-filter:a", ",".join(filters)]
    r = subprocess.run(cmd + out + ["pipe:1"], input=wav, capture_output=True)
    if r.returncode != 0:
        raise HTTPException(500, f"encoding failed: {r.stderr.decode()[:300]}")
    return r.stdout


def _list_generations() -> list[dict]:
    items = []
    if GEN_DIR.exists():
        for m in GEN_DIR.glob("*.json"):
            meta = _read_json(m, None)
            if meta and (GEN_DIR / f"{meta['id']}.wav").exists():
                items.append(meta)
    return sorted(items, key=lambda x: x.get("created", 0), reverse=True)


def _save_generation(wav: bytes, meta: dict) -> dict:
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    gid = uuid.uuid4().hex[:12]
    (GEN_DIR / f"{gid}.wav").write_bytes(wav)
    meta = {"id": gid, "created": time.time(), **meta}
    _write_json(GEN_DIR / f"{gid}.json", meta)
    for old in _list_generations()[MAX_GENERATIONS:]:
        _delete_generation(old["id"])
    return meta


def _delete_generation(gid: str) -> None:
    for ext in ("wav", "json"):
        (GEN_DIR / f"{gid}.{ext}").unlink(missing_ok=True)


# --------------------------------------------------------------------------- app

app = FastAPI(title="Qwen3-TTS", version="2.0")


@app.on_event("startup")
async def _startup() -> None:
    for d in (VOICES_DIR, GEN_DIR):
        d.mkdir(parents=True, exist_ok=True)
    engine = _read_json(STATE_FILE, {}).get("engine", DEFAULT_ENGINE)
    if engine not in ENGINES or not Path(ENGINES[engine]["python"]).exists():
        engine = DEFAULT_ENGINE
    try:
        await engines.start(engine)
    except HTTPException as e:
        engines.status, engines.error = "error", e.detail


@app.on_event("shutdown")
async def _shutdown() -> None:
    await engines._stop()


@app.get("/livez")
def livez():
    return {"ok": True}


@app.get("/healthz")
def healthz():
    s = engines.snapshot()
    return JSONResponse({"ok": s["status"] == "ready", **s}, status_code=200 if s["status"] == "ready" else 503)


@app.get("/v1/engine")
def get_engine():
    return engines.snapshot()


class EngineSwitch(BaseModel):
    engine: str


@app.post("/v1/engine")
async def set_engine(req: EngineSwitch):
    if req.engine == engines.engine and engines.status in ("starting", "ready"):
        return engines.snapshot()
    await engines.start(req.engine)
    return engines.snapshot()


@app.get("/v1/info")
def info():
    langs = engines.health.get("languages") or []
    if langs:
        langs = ["Auto", *[l.capitalize() for l in langs if l.lower() != "auto"]]
    return {"model": MODEL_ID, "languages": langs or LANGUAGES, "default_voice": DEFAULT_VOICE,
            "formats": sorted(MEDIA), "engine": engines.snapshot()}


# ---- voices


@app.get("/v1/audio/voices")
@app.get("/v1/voices")
def voices():
    v = _list_voices()
    return {"voices": [x["name"] for x in v], "details": v}


@app.post("/v1/voices")
async def create_voice(name: str = Form(...), file: UploadFile = File(...), ref_text: str = Form(""),
                       language: str = Form("Auto"), overwrite: bool = Form(False)):
    d = _voice_dir(name)
    if d.exists() and not overwrite:
        raise HTTPException(409, f"voice '{name}' already exists (tick 'Replace' to overwrite)")
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
    _write_json(d / "meta.json", meta)
    await engines.warm(d / "ref.wav", meta["ref_text"])
    return meta


@app.delete("/v1/voices/{name}")
def delete_voice(name: str):
    d = _voice_dir(name)
    if not d.exists():
        raise HTTPException(404, f"voice '{name}' not found")
    shutil.rmtree(d)
    return {"deleted": name}


@app.get("/v1/voices/{name}/reference")
def voice_reference(name: str):
    p = _voice_dir(name) / "ref.wav"
    if not p.exists():
        raise HTTPException(404, f"voice '{name}' not found")
    return FileResponse(p, media_type="audio/wav")


# ---- speech


class SpeechRequest(BaseModel):
    input: str = Field(..., min_length=1)
    voice: Optional[str] = None
    model: Optional[str] = None             # OpenAI compatibility, ignored
    instructions: Optional[str] = None      # OpenAI compatibility, ignored (Base model)
    response_format: str = "mp3"
    speed: float = Field(1.0, ge=0.25, le=4.0)
    language: Optional[str] = None
    temperature: Optional[float] = Field(None, ge=0.05, le=2.0)
    top_p: Optional[float] = Field(None, gt=0.0, le=1.0)
    top_k: Optional[int] = Field(None, ge=1, le=500)
    repetition_penalty: Optional[float] = Field(None, ge=1.0, le=2.0)
    max_new_tokens: Optional[int] = Field(None, ge=16, le=4096)
    seed: Optional[int] = None
    save: bool = False                      # keep it in the UI's generation history

    def params(self) -> dict:
        keys = ("temperature", "top_p", "top_k", "repetition_penalty", "max_new_tokens")
        return {k: getattr(self, k) for k in keys if getattr(self, k) is not None}


def _audio_headers(stats: dict, gen: Optional[dict]) -> dict:
    h = {"X-Engine": stats["engine"], "X-Gen-Seconds": f"{stats['gen_seconds']:.2f}",
         "X-Audio-Seconds": f"{stats['audio_seconds']:.2f}",
         "Access-Control-Expose-Headers": "X-Engine, X-Gen-Seconds, X-Audio-Seconds, X-Generation-Id"}
    if gen:
        h["X-Generation-Id"] = gen["id"]
    return h


@app.post("/v1/audio/speech")
async def speech(req: SpeechRequest):
    if len(req.input) > MAX_INPUT_CHARS:
        raise HTTPException(413, f"input is over {MAX_INPUT_CHARS} characters")
    fmt = req.response_format.lower()
    if fmt not in MEDIA:
        raise HTTPException(400, f"response_format must be one of {sorted(MEDIA)}")
    name = req.voice or DEFAULT_VOICE
    if not name:
        v = _list_voices()
        if not v:
            raise HTTPException(400, "no voices saved yet: POST /v1/voices first")
        name = v[0]["name"]
    meta = _read_meta(name)
    language = req.language or meta.get("language") or DEFAULT_LANGUAGE
    wav, stats = await engines.generate(_voice_dir(name) / "ref.wav", meta.get("ref_text", ""),
                                        req.input, language, req.params(), req.seed)
    gen = None
    if req.save:
        kept = await run_in_threadpool(_encode, wav, "wav", req.speed)
        gen = await run_in_threadpool(_save_generation, kept, {
            "text": req.input, "voice": name, "language": language, "engine": stats["engine"],
            "params": {**req.params(), "speed": req.speed, "seed": req.seed},
            "gen_seconds": round(stats["gen_seconds"], 2), "audio_seconds": round(stats["audio_seconds"] / req.speed, 2)})
    data = await run_in_threadpool(_encode, wav, fmt, req.speed)
    return Response(data, media_type=MEDIA[fmt], headers={
        **_audio_headers(stats, gen), "Content-Disposition": f'inline; filename="speech.{fmt}"'})


@app.post("/v1/audio/clone")
async def clone_once(file: UploadFile = File(...), text: str = Form(...), ref_text: str = Form(""),
                     language: str = Form(DEFAULT_LANGUAGE), response_format: str = Form("wav"),
                     speed: float = Form(1.0), temperature: Optional[float] = Form(None),
                     top_p: Optional[float] = Form(None), top_k: Optional[int] = Form(None),
                     repetition_penalty: Optional[float] = Form(None), seed: Optional[int] = Form(None)):
    if len(text) > MAX_INPUT_CHARS:
        raise HTTPException(413, f"text is over {MAX_INPUT_CHARS} characters")
    params = {k: v for k, v in dict(temperature=temperature, top_p=top_p, top_k=top_k,
                                    repetition_penalty=repetition_penalty).items() if v is not None}
    with tempfile.TemporaryDirectory(dir=DATA_DIR) as tmp:   # on the PVC so the worker can read it
        tmp = Path(tmp)
        raw = await _save_upload(file, tmp)
        wav_path = tmp / "ref.wav"
        dur = await run_in_threadpool(_to_ref_wav, raw, wav_path)
        if dur > MAX_REF_SECONDS:
            raise HTTPException(400, f"reference is {dur:.1f}s; keep it under {MAX_REF_SECONDS:.0f}s")
        wav, stats = await engines.generate(wav_path, ref_text.strip(), text, language, params, seed)
    data = await run_in_threadpool(_encode, wav, response_format, speed)
    return Response(data, media_type=MEDIA[response_format.lower()], headers=_audio_headers(stats, None))


# ---- generations


@app.get("/v1/generations")
def generations():
    return {"generations": _list_generations()}


@app.get("/v1/generations/{gid}/audio")
def generation_audio(gid: str):
    p = GEN_DIR / f"{gid}.wav"
    if not GEN_ID_RE.match(gid) or not p.exists():
        raise HTTPException(404, "generation not found")
    return FileResponse(p, media_type="audio/wav", filename=f"qwen3-tts-{gid}.wav")


@app.delete("/v1/generations/{gid}")
def delete_generation(gid: str):
    if gid == "all":
        n = 0
        for g in _list_generations():
            _delete_generation(g["id"]); n += 1
        return {"deleted": n}
    if not GEN_ID_RE.match(gid) or not (GEN_DIR / f"{gid}.json").exists():
        raise HTTPException(404, "generation not found")
    _delete_generation(gid)
    return {"deleted": gid}


# ---- presets


class Preset(BaseModel):
    voice: Optional[str] = None
    language: Optional[str] = None
    engine: Optional[str] = None
    response_format: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None
    speed: Optional[float] = None
    seed: Optional[int] = None
    text: Optional[str] = None


@app.get("/v1/presets")
def presets():
    data = _read_json(PRESETS_FILE, {})
    return {"presets": [{"name": k, **v} for k, v in sorted(data.items(), key=lambda kv: kv[0].lower())]}


@app.put("/v1/presets/{name}")
def save_preset(name: str, preset: Preset):
    name = name.strip()
    if not PRESET_RE.match(name):
        raise HTTPException(400, "preset name: 1–64 characters, no slashes")
    data = _read_json(PRESETS_FILE, {})
    data[name] = preset.model_dump(exclude_none=True)
    _write_json(PRESETS_FILE, data)
    return {"name": name, **data[name]}


@app.delete("/v1/presets/{name}")
def delete_preset(name: str):
    data = _read_json(PRESETS_FILE, {})
    if name not in data:
        raise HTTPException(404, f"preset '{name}' not found")
    del data[name]
    _write_json(PRESETS_FILE, data)
    return {"deleted": name}


# ---- UI


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
