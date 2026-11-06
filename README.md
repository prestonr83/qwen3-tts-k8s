# qwen3-tts-k8s

[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) voice cloning on k3s, on the same GPU node as chatterbox
(`k3s-server-01`, time-sliced). It runs a small FastAPI server from this repo around the
`Qwen3-TTS-12Hz-1.7B-Base` model, with:

- a test web UI (save voices from a file or the mic, tune sampling, compare takes)
- an OpenAI-compatible `/v1/audio/speech` endpoint and `/v1/audio/voices`, so Home Assistant's
  OpenAI TTS integration works the same way it does with chatterbox
- saved voices on a PVC, with the clone prompt cached in memory after first use

GitHub Actions builds the image. Nothing is built on the node.

```
Dockerfile, requirements.txt, app/  the server (app/server.py) and web UI (app/static/index.html)
.github/workflows/build-image.yml   builds the image and pushes it to ghcr.io
k8s/                                PVCs, Deployment, Service, optional IngressRoute (namespace tts)
scripts/smoke-test.sh               curl-based end-to-end check
```

## Pipeline

- **Triggers:** a push to `main` that changes the Dockerfile, `requirements.txt`, `app/` or the workflow, or a manual
  **Actions → Build image → Run workflow** (where you can optionally set a `qwen-tts` PyPI version).
- **Pushes** `ghcr.io/<owner>/qwen3-tts-server:latest` and `:<short sha>`. The run summary prints the exact tags.
- The first build takes a while (PyTorch + CUDA wheels are several GB).
- If the GHCR package is private, the cluster needs the same pull secret chatterbox uses (or make the package public).

## Deploy

Uses the existing `tts` namespace, `nvidia` RuntimeClass and GPU time-slicing.

```
kubectl apply -k k8s/
kubectl -n tts logs -f deploy/qwen3-tts          # first start downloads ~5 GB of weights
kubectl -n tts get pods -l app=qwen3-tts -w       # READY 1/1 once the model is loaded
kubectl -n tts exec deploy/qwen3-tts -- nvidia-smi
```

The log prints how much VRAM the model took once it's ready (about 5–7 GB in bf16 for the 1.7B model; generation
adds a little on top). `/healthz` reports it too.

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

## Voice cloning tips

1. **Always give the transcript.** With a transcript the model clones in in-context mode (timbre *and* speaking
   style). Without one it falls back to the speaker embedding only, which is noticeably weaker. The transcript must
   match the clip word for word.
2. Use 5–20 s of clean, dry speech from one speaker: no music, reverb, or noise. Clips over 30 s are rejected.
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
`repetition_penalty`, `max_new_tokens` and `seed`. Formats: wav, mp3, flac, opus, aac, pcm. Long input is split on
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
| `DTYPE` | `bfloat16` | |
| `ATTN_IMPL` | `sdpa` | `flash_attention_2` needs flash-attn in the image |
| `DEFAULT_LANGUAGE` | `English` | used when neither the request nor the voice sets one |
| `DEFAULT_VOICE` | *(first saved voice)* | |
| `CHUNK_CHARS` | `350` | long-input split size |
| `MAX_REF_SECONDS` | `30` | |
