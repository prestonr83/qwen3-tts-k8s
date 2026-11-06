"""
Runtime fix for qwen-tts-hf (the Transformers-5 port of qwen_tts used by faster-qwen3-tts).

Its RoPE shim reads `config.rope_theta`. Transformers 5 moved that value into
`config.rope_parameters`, and newer 5.x releases re-run RoPE initialisation while
loading weights, so loading the 12Hz speech tokenizer fails with:

    AttributeError: 'MimiConfig' object has no attribute 'rope_theta'

The shim function is replaced in place (its __code__), so every existing reference to it
(the transformers RoPE registry, modules that stored it as rope_init_fn) gets the fix.
"""

from __future__ import annotations

import logging

log = logging.getLogger("qwen3-tts")


def _default_rope_parameters(config, device=None, seq_len=None, layer_type=None):
    del seq_len
    params = getattr(config, "rope_parameters", None) or {}
    if layer_type is not None and isinstance(params.get(layer_type), dict):
        params = params[layer_type]
    base = getattr(config, "rope_theta", None)
    if base is None:
        base = params.get("rope_theta", 10000.0)
    partial_rotary_factor = getattr(config, "partial_rotary_factor", None)
    if partial_rotary_factor is None:
        partial_rotary_factor = params.get("partial_rotary_factor", 1.0)
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    dim = int(head_dim * partial_rotary_factor)
    inv_freq = 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
    )
    return inv_freq, 1.0


def apply() -> None:
    try:
        from qwen_tts import _transformers_compat as compat
    except ImportError:
        return  # original qwen-tts on Transformers 4.x: nothing to fix
    target = getattr(compat, "_default_rope_parameters", None)
    if target is None or getattr(target, "_qwen3_tts_patched", False):
        return
    # the replacement body uses `torch` from the compat module's globals, which imports it
    target.__code__ = _default_rope_parameters.__code__
    target._qwen3_tts_patched = True
    log.info("patched qwen_tts RoPE shim for transformers 5 rope_parameters")
