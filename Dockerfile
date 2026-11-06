# Qwen3-TTS voice-clone server (NVIDIA / CUDA 12.8).
# The PyTorch cu128 wheels bundle the CUDA runtime, so a slim Python base is enough;
# the node only needs the NVIDIA driver + container runtime.
FROM python:3.12-slim

ARG TORCH_VERSION=2.8.0
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128
ARG QWEN_TTS_VERSION=0.1.1

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/models \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    DATA_DIR=/data

# ffmpeg: decode uploads + encode mp3/opus/flac; sox: required by qwen-tts; libsndfile: soundfile
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg sox libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

# Torch first, from the CUDA index, so qwen-tts doesn't pull a CPU build
RUN pip install torch==${TORCH_VERSION} torchaudio==${TORCH_VERSION} --index-url ${TORCH_INDEX}

COPY requirements.txt /tmp/requirements.txt
RUN pip install qwen-tts==${QWEN_TTS_VERSION} -r /tmp/requirements.txt

RUN useradd -m -u 1000 app && mkdir -p /models /data && chown -R app:app /models /data
WORKDIR /app
COPY --chown=app:app app/ /app/
USER app

EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
