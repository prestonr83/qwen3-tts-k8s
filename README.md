# qwen3-tts-k8s

[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) voice cloning on k3s, on the same GPU node as chatterbox
(`k3s-server-01`, time-sliced). It runs a small FastAPI server from this repo around the
`Qwen3-TTS-12Hz-1.7B-Base` model, with two engines you can switch between from the UI:

| Engine | What it is | Environment in the image |
|---|---|---|
| **Fast** | [faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts) 0.5.4: CUDA-graph decoding, several times faster | `/opt/engines/fast`: qwen-tts-hf on transformers 5.20.2 (+ `app/qwen_compat.py`) |
| **Original** | the reference `qwen-tts` 0.1.1, unchanged; what the Hugging Face demo runs | `/opt/engines/original`: transformers 4.57.3 |

The two can't share one Python environment (they need different transformers majors), so each is a venv on top of
one shared PyTorch. The web server (`app/server.py`) runs the model in a worker process (`app/worker.py`) using the
selected engine's environment, one at a time: switching stops the worker, which frees its VRAM, and starts the other
(about the same time as a pod start, since weights come from the PVC cache). The choice survives restarts.

Also included:

- a web UI in the style of the Hugging Face demo: saved voices, reference upload/record, waveform players with
  one-at-a-time playback, generation history you can replay, download and delete, and saved presets
- an OpenAI-compatible `/v1/audio/speech` endpoint and `/v1/audio/voices`, so Home Assistant's
  OpenAI TTS integration works the same way it does with chatterbox
- saved voices on a PVC, with the clone prompt cached in memory after first use

GitHub Actions builds the image. Nothing is built on the node.

```
Dockerfile, requirements.txt        image: shared PyTorch + the two engine venvs
app/server.py                       web server: UI, voices, presets, history, API, engine switching
app/worker.py                       engine worker: loads the model and generates (runs in an engine venv)
app/qwen_compat.py                  RoPE fix for qwen-tts-hf on transformers 5 (fast engine)
app/static/index.html               web UI
.github/workflows/build-image.yml   builds the image and pushes it to ghcr.io
k8s/                                PVCs, Deployment, Service, optional IngressRoute (namespace tts)
scripts/smoke-test.sh               curl-based end-to-end check
```

## Pipeline

- **Triggers:** a push to `main` that changes the Dockerfile, `requirements.txt`, `app/` or the workflow, or a manual
  **Actions → Build image → Run workflow** (where you can optionally set a `faster-qwen3-tts` PyPI version).
- **Pushes** `ghcr.io/<owner>/qwen3-tts-server:latest` and `:<short sha>`. The run summary prints the exact tags.
- The first build takes a while (PyTorch + CUDA wheels are several GB).
- If the GHCR package is private, the cluster needs the same pull secret chatterbox uses (or make the package public).

## Deploy

Uses the existing `tts` namespace, `nvidia` RuntimeClass and GPU time-slicing.

```
kubectl apply -k k8s/
kubectl -n tts logs -f deploy/qwen3-tts          # first start downloads ~5 GB of weights
kubectl -n tts get pods -l app=qwen3-tts -w       # READY 1/1 as soon as the web server is up
kubectl -n tts exec deploy/qwen3-tts -- nvidia-smi
```

The pod is Ready as soon as the web server is up; the model keeps loading behind it, and the UI header shows
progress (`/v1/engine` and `/healthz` report it too). Each generation logs its speed, tagged with the engine
(`[fast] generated 17.0s audio in 3.1s (5.48x real time ...)`), and the UI's Status box shows the same.
The log also prints how much VRAM the model took once it's ready (about 5–7 GB in bf16 for the 1.7B model).

**VRAM budget on a shared card:** chatterbox, Qwen3-TTS and Jellyfin transcodes all draw from the same memory, and
time-slicing does not partition it. While you're comparing, it's fine to run both; if you hit CUDA out-of-memory
errors, scale chatterbox down (`kubectl -n tts scale deploy/chatterbox --replicas=0`) or switch `MODEL_ID` to
`Qwen/Qwen3-TTS-12Hz-0.6B-Base`.

**Pin the version:** set `newTag` in `k8s/kustomization.yaml` to the sha tag from the run summary, then
`kubectl apply -k k8s/`.

## Web UI

```
kubectl -n tts port-forward svc/qwen3-tts 8005:8005
```
Then open http://localhost:8005 (API docs at `/docs`). The **Record** button needs a secure context, which
`localhost` through port-forward counts as. For `https://qwen-tts.rdz.sh`, uncomment `ingressroute.yaml` in the
kustomization after creating the DNS record and the `qwen-tts-rdz-sh-tls` secret.

The server has **no authentication**. Keep it LAN-only.

## Comparing engines

Pick a voice, set a **seed**, generate, then click **Original** in the header, wait for it to load, and generate
again. History records which engine made each clip. The two engines won't produce the same audio from the same
seed, so judge them over a few takes each. Saved presets include the engine; loading one asks before switching.

## Voice cloning tips

1. **Always give the transcript.** With a transcript the model clones in in-context mode (timbre *and* speaking
   style). Without one it falls back to the speaker embedding only, which is noticeably weaker. The transcript must
   match the clip word for word.
2. Use 5–20 s of clean, dry speech from one speaker: no music, reverb, or noise. Clips over 30 s are rejected.
   The clip must start and end on a pause: the model continues the reference, so a clip cut mid-word puts the end of
   that word at the start of every output.
3. The style of the clip carries over. A calm read gives calm output; record the delivery you want.
4. Tuning: **temperature** (default 0.9) lower = steadier and closer to the reference, higher = more varied.
   Leave top-p at 1.0 and repetition penalty at ~1.05 unless you hear loops or stutters. Set a **seed** to compare
   settings on the same take.
5. **Language** defaults to the voice's language; set it explicitly for best results (Auto works but can drift).

Only clone voices you have permission to use.

## API

```
# save a voice
curl -F name=preston -F file=@me.wav -F "ref_text=Exactly what I said in the clip." -F language=English \
  http://localhost:8005/v1/voices

# speak (OpenAI-compatible)
curl -H 'Content-Type: application/json' -o out.mp3 http://localhost:8005/v1/audio/speech \
  -d '{"model":"qwen3-tts","voice":"preston","input":"Hello from the cluster.","response_format":"mp3"}'

# one-shot clone without saving anything
curl -F file=@me.wav -F "ref_text=..." -F "text=Hello." -o out.wav http://localhost:8005/v1/audio/clone
```

`/v1/audio/speech` also accepts `language`, `speed` (time-stretch), `temperature`, `top_p`, `top_k`,
`repetition_penalty`, `max_new_tokens`, `seed` and `save` (keep it in the UI's history). It uses whichever engine
is active; `GET /v1/engine` shows it and `POST /v1/engine {"engine":"original"}` switches.
History: `GET /v1/generations`, `GET /v1/generations/{id}/audio`, `DELETE /v1/generations/{id}` (or `all`).
Presets: `GET /v1/presets`, `PUT /v1/presets/{name}`, `DELETE /v1/presets/{name}`. Formats: wav, mp3, flac, opus, aac, pcm. Long input is split on
sentences and stitched together.

Or run `./scripts/smoke-test.sh my-clip.wav "transcript"` against the port-forward.

## Home Assistant

Same as chatterbox: in the **OpenAI TTS** integration (sfortis/openai_tts) add a custom endpoint at
`http://qwen3-tts.tts.svc.cluster.local:8005` (add `/v1/audio/speech` if it asks for the full URL) with no API key,
any model name, and one of your saved voice names as the voice. Set `DEFAULT_VOICE` in the Deployment if you want requests without a voice to use a
specific one.

## Config (Deployment env)

| Variable | Default | |
|---|---|---|
| `MODEL_ID` | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | `...-0.6B-Base` is smaller and faster |
| `ENGINE` | `fast` | engine on first start (`fast` or `original`); afterwards the UI's choice is remembered |
| `MAX_SEQ_LEN` | `2048` | fast engine's static KV cache: reference + text + output tokens per chunk |
| `REF_TAIL_SILENCE` | `0.5` | fast engine: silence appended to the reference (original engine passes it unchanged) |
| `MAX_GENERATIONS` | `200` | history size; the oldest clips are deleted beyond this |
| `DTYPE` | `bfloat16` | |
| `ATTN_IMPL` | `sdpa` | `flash_attention_2` needs flash-attn in the image |
| `DEFAULT_LANGUAGE` | `English` | used when neither the request nor the voice sets one |
| `DEFAULT_VOICE` | *(first saved voice)* | |
| `CHUNK_CHARS` | `350` | long-input split size |
| `MAX_REF_SECONDS` | `30` | |
