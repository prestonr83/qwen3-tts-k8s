# Qwen3-TTS voice-clone server (NVIDIA / CUDA 12.8).
# The PyTorch cu128 wheels bundle the CUDA runtime, so a slim Python base is enough;
# the node only needs the NVIDIA driver + container runtime.
FROM python:3.12-slim

ARG TORCH_VERSION=2.8.0
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128
# faster-qwen3-tts pulls in qwen-tts-hf (the qwen_tts module on transformers 5),
# which also serves BACKEND=stock, so one image covers both backends.
ARG FASTER_QWEN3_TTS_VERSION=0.5.4
# transformers 5.21.0 made MimiModel.upsample a read-only property; qwen-tts-hf's 12Hz tokenizer
# assigns it, so model load fails with "property 'upsample' ... has no setter". Stay on 5.20.x.
ARG TRANSFORMERS_VERSION=5.20.2

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/models \
    DATA_DIR=/data

# ffmpeg: decode uploads + encode mp3/opus/flac; sox: required by qwen-tts; libsndfile: soundfile
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg sox libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

# Torch first, from the CUDA index, so qwen-tts doesn't pull a CPU build
RUN pip install torch==${TORCH_VERSION} torchaudio==${TORCH_VERSION} --index-url ${TORCH_INDEX}

COPY requirements.txt /tmp/requirements.txt
RUN pip install faster-qwen3-tts==${FASTER_QWEN3_TTS_VERSION} transformers==${TRANSFORMERS_VERSION} \
        torch==${TORCH_VERSION} torchaudio==${TORCH_VERSION} -r /tmp/requirements.txt \
 && python -c "import faster_qwen3_tts, qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2, torch, transformers; \
from transformers import MimiModel; \
assert not isinstance(getattr(MimiModel, 'upsample', None), property), 'MimiModel.upsample is read-only in this transformers; qwen-tts-hf needs <5.21'; \
print('torch', torch.__version__, 'transformers', transformers.__version__, 'faster-qwen3-tts', faster_qwen3_tts.__version__)"

RUN useradd -m -u 1000 app && mkdir -p /models /data && chown -R app:app /models /data
WORKDIR /app
COPY --chown=app:app app/ /app/
USER app

EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]