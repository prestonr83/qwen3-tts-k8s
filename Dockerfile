# Qwen3-TTS voice-clone server (NVIDIA / CUDA 12.8) with two switchable engines:
#   /opt/engines/fast      faster-qwen3-tts 0.5.4 (qwen-tts-hf on transformers 5.20.2)
#   /opt/engines/original  qwen-tts 0.1.1 on transformers 4.57.3, exactly the first deployment
# Both are venvs layered on one system-wide PyTorch, so the CUDA wheels are only installed once.
# The web server runs on the system Python and starts one engine worker at a time.
FROM python:3.12-slim

ARG TORCH_VERSION=2.8.0
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128
ARG FASTER_QWEN3_TTS_VERSION=0.5.4
# transformers 5.21.0 made MimiModel.upsample read-only; qwen-tts-hf assigns it. Stay on 5.20.x.
# (app/qwen_compat.py fixes the other incompatibility: the RoPE shim reading config.rope_theta.)
ARG TRANSFORMERS_FAST_VERSION=5.20.2
ARG QWEN_TTS_ORIGINAL_VERSION=0.1.1

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/models \
    DATA_DIR=/data

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg sox libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

# Shared: PyTorch (CUDA 12.8) + the web server's own dependencies
RUN pip install torch==${TORCH_VERSION} torchaudio==${TORCH_VERSION} --index-url ${TORCH_INDEX}
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt \
 && printf 'torch==%s\ntorchaudio==%s\n' "${TORCH_VERSION}" "${TORCH_VERSION}" > /opt/torch-constraints.txt

# Engine: original (unchanged first deployment)
RUN python -m venv --system-site-packages /opt/engines/original \
 && /opt/engines/original/bin/pip install -c /opt/torch-constraints.txt qwen-tts==${QWEN_TTS_ORIGINAL_VERSION} \
 && /opt/engines/original/bin/python -c "import torch, transformers; from qwen_tts import Qwen3TTSModel; \
assert transformers.__version__.startswith('4.57'), transformers.__version__; \
print('original engine: torch', torch.__version__, 'transformers', transformers.__version__)"

# Engine: fast
RUN python -m venv --system-site-packages /opt/engines/fast \
 && /opt/engines/fast/bin/pip install -c /opt/torch-constraints.txt \
        faster-qwen3-tts==${FASTER_QWEN3_TTS_VERSION} transformers==${TRANSFORMERS_FAST_VERSION} \
 && /opt/engines/fast/bin/python -c "import faster_qwen3_tts, torch, transformers; \
from transformers import MimiModel; \
assert not isinstance(getattr(MimiModel, 'upsample', None), property), 'MimiModel.upsample is read-only; qwen-tts-hf needs transformers <5.21'; \
print('fast engine: torch', torch.__version__, 'transformers', transformers.__version__, 'faster-qwen3-tts', faster_qwen3_tts.__version__)"

RUN useradd -m -u 1000 app && mkdir -p /models /data && chown -R app:app /models /data
WORKDIR /app
COPY --chown=app:app app/ /app/
USER app

# Fail the build (not the pod) if the RoPE fix stops working in the fast engine.
RUN /opt/engines/fast/bin/python -c "import torch, qwen_compat; qwen_compat.apply(); \
from transformers import MimiConfig; from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS; \
inv, _ = ROPE_INIT_FUNCTIONS['default'](MimiConfig(hidden_size=512, num_attention_heads=8, rope_theta=10000.0)); \
assert torch.allclose(inv, 1.0 / (10000.0 ** (torch.arange(0, 64, 2).float() / 64))), 'RoPE fix broken'; print('rope fix ok')"

EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
