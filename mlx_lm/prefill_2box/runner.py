# Copyright © 2026 Apple Inc.
"""Layer-slice runner shared by the two-box prefill server and orchestrator.

A ``LayerSlice`` drives a contiguous run of decoder layers with its own cache
list, reproducing exactly what ``Qwen3_5TextModel.__call__`` does for those
layers: same mask construction (attention mask from the slice's first full-attn
cache, ssm mask from its first linear cache), same layer loop.  For batch-1
chunked prefill the ssm mask is None and the attention mask is "causal" with
the KV cache's offset — identical to the single-box path, so a split run is
bitwise-reproducible against a single-box run with the same chunk schedule.
"""

from pathlib import Path

import mlx.core as mx

from ..models.base import create_attention_mask, create_ssm_mask
from ..models.cache import ArraysCache, KVCache


def set_wired_limit():
    """>100 GB-class models silently thrash without this (fleet rule #1)."""
    try:
        info = mx.device_info()
    except AttributeError:  # older mlx
        info = mx.metal.device_info()
    lim = info["max_recommended_working_set_size"]
    try:
        mx.set_wired_limit(lim)
    except AttributeError:
        mx.metal.set_wired_limit(lim)
    return lim


def load_model_only(path):
    """Load model weights without a tokenizer (server side needs no tokenizer)."""
    from ..utils import load_model

    model, _config = load_model(Path(path))
    return model


def make_schedule(n, chunk):
    """Uniform chunk schedule covering exactly n tokens (last chunk = remainder).

    Two-stage pipeline total is (C + c_max)/2 where C is the single-box time of
    the same schedule and c_max the largest chunk's time — uniform chunks
    minimize c_max at a given chunk count.
    """
    if n <= 0:
        raise ValueError(f"nothing to prefill (n={n})")
    if chunk <= 0:
        raise ValueError(f"bad chunk {chunk}")
    out = []
    while n > 0:
        out.append(min(chunk, n))
        n -= out[-1]
    return out


class LayerSlice:
    """Contiguous decoder-layer slice [lo, hi) of a loaded qwen3_5-class model."""

    def __init__(self, model, lo, hi):
        core = model.model  # Qwen3_5TextModel
        n = len(core.layers)
        if not (0 <= lo < hi <= n):
            raise ValueError(f"bad slice [{lo}, {hi}) of {n} layers")
        self.lo, self.hi, self.n_layers = lo, hi, n
        self.hidden_size = model.language_model.args.hidden_size
        self.embed = core.embed_tokens if lo == 0 else None
        self.layers = core.layers[lo:hi]
        self.final_norm = core.norm if hi == n else None
        self.lm_head = None
        if hi == n:
            lm = model.language_model
            self.lm_head = (
                core.embed_tokens.as_linear
                if lm.args.tie_word_embeddings
                else lm.lm_head
            )
        self._fa = next(
            (i for i, l in enumerate(self.layers) if not l.is_linear), None
        )
        self._ssm = next((i for i, l in enumerate(self.layers) if l.is_linear), None)
        self.reset()

    def reset(self):
        self.cache = [
            ArraysCache(size=2) if l.is_linear else KVCache() for l in self.layers
        ]

    def state_arrays(self):
        """Raw cache arrays for mx.eval after a chunk (bounds the lazy graph)."""
        out = []
        for c in self.cache:
            if isinstance(c, ArraysCache):
                out.extend(x for x in c.cache if x is not None)
            elif c.keys is not None:
                out.append(c.keys)
                out.append(c.values)
        return out

    def offset(self):
        for c in self.cache:
            if isinstance(c, KVCache):
                return c.offset
        return None

    def forward(self, x, is_tokens=False):
        """One chunk through the slice; updates the slice cache in place."""
        h = self.embed(x) if is_tokens else x
        fa_mask = (
            create_attention_mask(h, self.cache[self._fa])
            if self._fa is not None
            else None
        )
        ssm_mask = (
            create_ssm_mask(h, self.cache[self._ssm])
            if self._ssm is not None
            else None
        )
        for layer, c in zip(self.layers, self.cache):
            h = layer(h, mask=ssm_mask if layer.is_linear else fa_mask, cache=c)
        return h

    def logits_last(self, h):
        """Final norm + lm_head on the last position only (num_logits=1)."""
        if self.final_norm is None or self.lm_head is None:
            raise RuntimeError("slice does not own the model tail")
        return self.lm_head(self.final_norm(h[:, -1:, :]))


def iter_cache_tensors(sl, include_kv=True):
    """Yield (name, array) for every cache tensor of the slice, named by
    global layer index: conv:<i> / ssm:<i> for GDN, k:<i> / v:<i> for KV."""
    for i, (layer, c) in enumerate(zip(sl.layers, sl.cache)):
        gi = sl.lo + i
        if layer.is_linear:
            if c.cache[0] is None:
                continue
            yield f"conv:{gi}", c.cache[0]
            yield f"ssm:{gi}", c.cache[1]
        elif include_kv and c.keys is not None:
            k, v = c.state  # trimmed to offset
            yield f"k:{gi}", mx.contiguous(k)
            yield f"v:{gi}", mx.contiguous(v)


def build_full_cache(model, local_slice, remote):
    """Full 64-entry cache list: remote tensors for [0, lo), local slice's own
    cache objects for [lo, n).  ``remote`` maps name -> mx.array."""
    lo = local_slice.lo
    caches = []
    for gi in range(lo):
        layer = model.model.layers[gi]
        if layer.is_linear:
            ac = ArraysCache(size=2)
            ac[0] = remote[f"conv:{gi}"]
            ac[1] = remote[f"ssm:{gi}"]
            caches.append(ac)
        else:
            kv = KVCache()
            kv.state = (remote[f"k:{gi}"], remote[f"v:{gi}"])
            caches.append(kv)
    caches.extend(local_slice.cache)
    return caches
