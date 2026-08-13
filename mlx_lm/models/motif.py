# Copyright © 2023-2024 Apple Inc.

"""Motif-3 (model_type "Motif") — Motif Technologies' 314.8B MoE.

Architecture (from the HF reference modeling_motif.py, which this file
reproduces numerically):

  - GDLA attention: MLA-style low-rank q/kv projections (q_lora 1024,
    kv_lora 512) materialized into a standard 16-head GQA KV cache
    (k 192 = 128 nope + 64 rope, v 128), differential attention v2
    (80 heads = 16 groups x [4 signal + 1 noise], data-dependent lambda),
    and an elementwise sigmoid output gate from the q latent.
  - PolyNorm activations (w0*N(x^3)+w1*N(x^2)+w2*N(x)+b); routed experts
    carry per-expert coefficients (GroupedPolyNorm).
  - MoE: 384 routed experts, top-8, sigmoid scores + aux-loss-free
    expert_bias for selection only, route-normalized weights x2.0, plus one
    always-on shared expert. Layers 0-1 are dense.
  - mHC (manifold-constrained hyper-connections): the residual stream is
    4-wide (B, L, 4, D); each sublayer reads/writes it through learned
    gates with a Sinkhorn-projected doubly-stochastic 4x4 mixing matrix.
  - Interleaved sliding-window attention: window 129 (incl. self) on layers
    where (i+1) % 4 != 0; full attention on the other 13 layers.

RoPE note: the shipped HF code's yarn path is inoperative (dim mismatch;
_init_weights restores plain theta=10000 frequencies). By default we implement
plain RoPE over the 64 rope dims with the DeepSeek-style mscale^2 factor folded
into the softmax scale. `rope_cos_sin_scale` (default 1.0) optionally
multiplies the rotated q_pe/k_pe like the HF attention_scaling would have —
a runtime A/B knob until settled empirically; see ~/motif3 runbook.

RoPE variants (additive, default-off): `rope_variant` ("plain"|"yarn") and
`rope_rotation` ("half"|"interleaved") select the frequency schedule and the
pairing convention through the MotifRotary module. The defaults ("plain",
"half") are byte-identical to nn.RoPE(traditional=False, base=rope_theta) — the
plain path calls mx.fast.rope(base=...) unchanged; "yarn" uses transformers'
_compute_yarn_parameters interpolation via mx.fast.rope(freqs=1/inv_freq), and
"interleaved" matches nn.RoPE(traditional=True). `rope.inv_freq` and
`rope.rotation` are plain (non-parameter) attributes, hot-patchable per layer
for single-load A/B testing; see ~/motif3/scripts/rope_ab2.py.

The MTP block (model.mtp_layers.0.*) is not instantiated by the reference
model and is dropped in sanitize().
"""

import math
import os
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import KVCache, RotatingKVCache
from .switch_layers import SwitchGLU, _gather_sort, _scatter_unsort


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "Motif"
    vocab_size: int = 220160
    hidden_size: int = 4096
    intermediate_size: int = 12288
    moe_intermediate_size: int = 1280
    num_hidden_layers: int = 53
    num_attention_heads: int = 80
    num_key_value_heads: int = 16
    num_noise_heads: int = 16
    head_dim: int = 192
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    q_lora_rank: int = 1024
    kv_lora_rank: int = 512
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    max_position_embeddings: int = 262144
    original_seq_len: int = 4096
    rope_factor: float = 64.0
    mscale: float = 1.0
    swa_rope_theta: float = 10000.0
    rope_cos_sin_scale: float = 1.0
    rope_variant: str = "plain"  # "plain" | "yarn" (additive; default = plain theta=10000)
    rope_rotation: str = "half"  # "half" (NeoX) | "interleaved" (traditional); default = today's
    polynorm_sigmoid_weight: bool = True
    polynorm_output_scale: float = 0.5
    polynorm_bias_clamp: Optional[float] = 0.5
    num_experts: int = 384
    experts_top_k: int = 8
    num_shared_experts: int = 1
    n_dense_first_layers: int = 2
    interleave_moe_layer_step: int = 1
    route_scale: float = 2.0
    route_norm: bool = True
    score_func: str = "sigmoid"
    attention_cls: str = "gdla"
    use_sliding_window: bool = True
    sliding_window: int = 128
    sliding_window_pattern: str = "interleave"
    sliding_window_period: int = 4
    mhc_enabled: bool = True
    mhc_expansion_rate: int = 4
    mhc_sinkhorn_iters: int = 20
    mhc_h_post_alpha_end: float = 0.0  # Motif-3: h_post = (1 + this) * sigmoid = 1.0*sigmoid
    elementwise_attn_output_gate: bool = True
    tie_word_embeddings: bool = False

    def __post_init__(self):
        if self.attention_cls != "gdla":
            raise ValueError(f"attention_cls={self.attention_cls!r} not supported (gdla only)")
        if self.score_func != "sigmoid":
            raise ValueError(f"score_func={self.score_func!r} not supported (sigmoid only)")
        if not self.mhc_enabled:
            raise ValueError("mhc_enabled=False path is not implemented")
        if self.rope_variant not in ("plain", "yarn"):
            raise ValueError(f"rope_variant={self.rope_variant!r} not supported (plain|yarn)")
        if self.rope_rotation not in ("half", "interleaved"):
            raise ValueError(
                f"rope_rotation={self.rope_rotation!r} not supported (half|interleaved)"
            )


def _poly_norm_terms(x, eps=1e-6):
    def n(t):
        return t * mx.rsqrt(mx.mean(mx.square(t), axis=-1, keepdims=True) + eps)

    x2 = mx.square(x)
    return n(x * x2), n(x2), n(x)


class PolyNorm(nn.Module):
    """Trainable polynomial-of-norms activation; weight [3], bias [1]."""

    def __init__(self, sigmoid_weight: bool = True, output_scale: float = 1.0):
        super().__init__()
        self.sigmoid_weight = sigmoid_weight
        self.output_scale = output_scale
        self.weight = mx.ones((3,)) / 3
        self.bias = mx.zeros((1,))

    def __call__(self, x):
        a, b, c = _poly_norm_terms(x)
        w = mx.sigmoid(self.weight) if self.sigmoid_weight else self.weight
        return self.output_scale * (w[0] * a + w[1] * b + w[2] * c + self.bias)


class GroupedPolyNorm(nn.Module):
    """Per-expert PolyNorm; weight [E, 3], bias [E, 1], gathered by expert index."""

    def __init__(
        self,
        num_experts: int,
        sigmoid_weight: bool = True,
        bias_clamp: Optional[float] = None,
        output_scale: float = 1.0,
    ):
        super().__init__()
        self.sigmoid_weight = sigmoid_weight
        self.bias_clamp = bias_clamp
        self.output_scale = output_scale
        self.weight = mx.ones((num_experts, 3)) / 3
        self.bias = mx.zeros((num_experts, 1))

    def __call__(self, x, indices):
        # x: (..., 1, I) rows aligned with indices (...,); both the sorted-flat
        # prefill path and the (B, L, K) decode path broadcast identically.
        a, b, c = _poly_norm_terms(x)
        w = self.weight[indices]
        if self.sigmoid_weight:
            w = mx.sigmoid(w)
        bias = self.bias[indices]
        if self.bias_clamp is not None:
            bias = mx.clip(bias, -self.bias_clamp, self.bias_clamp)
        return self.output_scale * (
            w[..., 0:1, None] * a
            + w[..., 1:2, None] * b
            + w[..., 2:3, None] * c
            + bias[..., None]
        )


class MotifMLP(nn.Module):
    def __init__(self, args: ModelArgs, intermediate_size: Optional[int] = None):
        super().__init__()
        dim = args.hidden_size
        hidden = intermediate_size or args.intermediate_size
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)
        self.act_fn = PolyNorm(
            sigmoid_weight=args.polynorm_sigmoid_weight,
            output_scale=args.polynorm_output_scale,
        )

    def __call__(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class MotifSwitchGLU(SwitchGLU):
    """SwitchGLU with per-expert PolyNorm coefficients.

    Same parameter tree as SwitchGLU (gate_proj/up_proj/down_proj SwitchLinear
    stacks) plus act_fn; __call__ mirrors SwitchGLU but feeds the (sorted)
    expert indices to the grouped activation.
    """

    def __init__(
        self, input_dims: int, hidden_dims: int, num_experts: int, args: ModelArgs
    ):
        super().__init__(input_dims, hidden_dims, num_experts, bias=False)
        self.act_fn = GroupedPolyNorm(
            num_experts,
            sigmoid_weight=args.polynorm_sigmoid_weight,
            bias_clamp=args.polynorm_bias_clamp,
            output_scale=args.polynorm_output_scale,
        )

    def __call__(self, x, indices):
        x = mx.expand_dims(x, (-2, -3))

        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)
        if self.training:
            idx = mx.stop_gradient(idx)
        x_up = self.up_proj(x, idx, sorted_indices=do_sort)
        x_gate = self.gate_proj(x, idx, sorted_indices=do_sort)
        x = self.down_proj(
            self.act_fn(x_gate, idx) * x_up,
            idx,
            sorted_indices=do_sort,
        )

        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)

        return x.squeeze(-2)


class MoEGate(nn.Module):
    """Sigmoid router with aux-loss-free expert_bias (selection only).

    Raw parameters (no to_quantized) so quantization skips it; the module
    path ends in `gate` for the router-KD/traffic tooling regexes.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.top_k = args.experts_top_k
        self.route_scale = args.route_scale
        self.route_norm = args.route_norm
        self.weight = mx.zeros((args.num_experts, args.hidden_size))
        self.expert_bias = mx.zeros((args.num_experts,))

    def __call__(self, x):
        scores = mx.sigmoid((x @ self.weight.T).astype(mx.float32))
        biased = scores + self.expert_bias.astype(mx.float32)
        inds = mx.argpartition(-biased, kth=self.top_k - 1, axis=-1)[..., : self.top_k]
        w = mx.take_along_axis(scores, inds, axis=-1)
        if self.route_norm:
            w = w / (w.sum(axis=-1, keepdims=True) + 1e-20)
        return inds, w * self.route_scale


class MotifMoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.gate = MoEGate(args)
        self.switch_mlp = MotifSwitchGLU(
            args.hidden_size, args.moe_intermediate_size, args.num_experts, args
        )
        self.shared_experts = MotifMLP(
            args,
            intermediate_size=args.moe_intermediate_size * args.num_shared_experts,
        )

    def __call__(self, x):
        inds, scores = self.gate(x)
        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2).astype(y.dtype)
        return y + self.shared_experts(x)


@partial(mx.compile, shapeless=True)
def _mhc_gates(hp, hq, hr, a_pre, b_pre, a_post, b_post, a_res, b_res, iters: int = 20):
    """Sigmoid pre/post gates and the Sinkhorn doubly-stochastic 4x4 mix.

    fp32 throughout (the reference casts projection outputs to float32 and
    runs Sinkhorn in float32); row-normalize first, then column, as in the
    reference _sinkhorn_knopp_batch.
    """
    h_pre = mx.sigmoid(mx.clip(a_pre * hp + b_pre, -10.0, 10.0))
    # base coefficient 1.0 (Motif-3: h_post_coeff = 1 + mhc_h_post_alpha_end = 1.0,
    # NOT the paper's 2*sigmoid — confirmed by the Motif team). A non-1.0
    # coefficient is applied by MHCLayer.__call__ outside this compiled graph.
    h_post = mx.sigmoid(mx.clip(a_post * hq + b_post, -10.0, 10.0))
    m = mx.exp(mx.clip(a_res * hr + b_res, -20.0, 20.0))
    for _ in range(iters):
        m = m / mx.maximum(m.sum(axis=-1, keepdims=True), 1e-8)
        m = m / mx.maximum(m.sum(axis=-2, keepdims=True), 1e-8)
    return h_pre, h_post, m


_SINKHORN_SRC = """
    uint g = thread_position_in_grid.x;
    if (g >= (uint)n_mats[0]) return;
    float m[16];
    for (int i = 0; i < 16; ++i) m[i] = metal::exp(metal::clamp(inp[g*16+i], -20.0f, 20.0f));
    for (int it = 0; it < 20; ++it) {
        for (int r = 0; r < 4; ++r) {
            float s = (m[r*4] + m[r*4+1]) + (m[r*4+2] + m[r*4+3]);
            s = metal::max(s, 1e-8f);
            for (int c = 0; c < 4; ++c) m[r*4+c] /= s;
        }
        for (int c = 0; c < 4; ++c) {
            float s = (m[c] + m[4+c]) + (m[8+c] + m[12+c]);
            s = metal::max(s, 1e-8f);
            for (int r = 0; r < 4; ++r) m[r*4+c] /= s;
        }
    }
    for (int i = 0; i < 16; ++i) out[g*16+i] = m[i];
"""
_sinkhorn_k = None


def _sinkhorn_kernel(pre):
    """[I182] Sinkhorn 20-iter 단일 커널 — eager 대비 마이크로-디스패치 4240→106/tok,
    decode +40% (14.4→20.1 tok/s). KL vs eager 2.2e-3 / top-1 flip 0% (256pos).
    킬스위치: MOTIF_SINKHORN_KERNEL=0 → eager 경로."""
    global _sinkhorn_k
    if _sinkhorn_k is None:
        _sinkhorn_k = mx.fast.metal_kernel(
            name="motif_sinkhorn4", input_names=["inp", "n_mats"],
            output_names=["out"], source=_SINKHORN_SRC)
    B, L, _ = pre.shape
    n = B * L
    (out,) = _sinkhorn_k(
        inputs=[pre.reshape(n * 16), mx.array([n], dtype=mx.int32)],
        output_shapes=[(n * 16,)], output_dtypes=[mx.float32],
        grid=(max(n, 1), 1, 1), threadgroup=(min(max(n, 1), 64), 1, 1))
    return out.reshape(B, L, 4, 4)


class MHCLayer(nn.Module):
    """Manifold-constrained Hyper-Connections gate block (arXiv 2512.24880)."""

    def __init__(self, expansion_rate: int, num_dim: int, h_post_coeff: float = 1.0):
        super().__init__()
        self.h_post_coeff = float(h_post_coeff)
        E, D = expansion_rate, num_dim
        self.proj_pre = nn.Linear(E * D, E, bias=False)
        self.proj_post = nn.Linear(E * D, E, bias=False)
        self.proj_res = nn.Linear(E * D, E * E, bias=False)
        # eps hardcoded 1e-6 in the reference MHCLayer (not config rms_norm_eps)
        self.rms_norm = nn.RMSNorm(E * D, eps=1e-6)
        self.bias_pre = mx.zeros((E,))
        self.bias_post = mx.zeros((E,))
        self.bias_res = mx.zeros((E, E))
        self.alpha_pre = mx.zeros((1,))
        self.alpha_post = mx.zeros((1,))
        self.alpha_res = mx.zeros((1,))

    def __call__(self, x):
        B, L, E, D = x.shape
        xr = self.rms_norm(x.reshape(B, L, E * D))
        hp = self.proj_pre(xr).astype(mx.float32)
        hq = self.proj_post(xr).astype(mx.float32)
        if os.environ.get("MOTIF_SINKHORN_KERNEL", "1") == "1":
            hr = self.proj_res(xr).astype(mx.float32)
            h_pre = mx.sigmoid(mx.clip(self.alpha_pre * hp + self.bias_pre, -10.0, 10.0))
            h_post = mx.sigmoid(mx.clip(self.alpha_post * hq + self.bias_post, -10.0, 10.0))
            m = _sinkhorn_kernel(self.alpha_res * hr + self.bias_res.reshape(E * E))
        else:
            hr = self.proj_res(xr).astype(mx.float32).reshape(B, L, E, E)
            h_pre, h_post, m = _mhc_gates(
                hp, hq, hr,
                self.alpha_pre, self.bias_pre,
                self.alpha_post, self.bias_post,
                self.alpha_res, self.bias_res,
            )
        if self.h_post_coeff != 1.0:
            h_post = h_post * self.h_post_coeff
        return h_pre, h_post, m


def _rope_inv_freq_plain(dims: int, base: float) -> np.ndarray:
    """Plain RoPE inverse frequencies base**(-arange(0,dims,2)/dims); fp32, (dims//2,).

    Informational for the plain variant: MotifRotary's plain forward drives
    mx.fast.rope with `base=` directly (byte-identical to nn.RoPE), so it never
    reads these — they are exposed only as `rope.inv_freq` for A/B hot-patching.
    """
    return (
        np.float32(base) ** (-np.arange(0, dims, 2, dtype=np.float32) / np.float32(dims))
    ).astype(np.float32)


def _rope_inv_freq_yarn(
    dims: int,
    base: float,
    factor: float,
    original_max_position_embeddings: int,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
    truncate: bool = True,
) -> np.ndarray:
    """YaRN-interpolated inverse frequencies; fp32, (dims//2,).

    Ported line-for-line (float32 throughout → bit-identical) from transformers
    modeling_rope_utils._compute_yarn_parameters: the [low, high] correction
    band is set by beta_fast/beta_slow, then a clamped linear ramp blends the
    extrapolation branch (1/pos_freqs) and the interpolation branch
    (1/(factor*pos_freqs)). Only inv_freq is returned; the companion
    attention_scaling that function also emits is the DeepSeek mscale, applied
    elsewhere (softmax scale / cs_scale), not folded in here.
    """

    def find_correction_dim(num_rotations: float) -> float:
        return (dims * math.log(original_max_position_embeddings / (num_rotations * 2 * math.pi))) / (
            2 * math.log(base)
        )

    low = find_correction_dim(beta_fast)
    high = find_correction_dim(beta_slow)
    if truncate:
        low = math.floor(low)
        high = math.ceil(high)
    low = max(low, 0)
    high = min(high, dims - 1)
    if low == high:
        high += 0.001  # prevent singularity (matches the reference)

    pos_freqs = np.float32(base) ** (np.arange(0, dims, 2, dtype=np.float32) / np.float32(dims))
    inv_freq_extrapolation = (1.0 / pos_freqs).astype(np.float32)
    inv_freq_interpolation = (1.0 / (np.float32(factor) * pos_freqs)).astype(np.float32)

    linear = (np.arange(dims // 2, dtype=np.float32) - low) / np.float32(high - low)
    extrapolation_factor = (1.0 - np.clip(linear, 0.0, 1.0)).astype(np.float32)
    inv_freq = (
        inv_freq_interpolation * (1.0 - extrapolation_factor)
        + inv_freq_extrapolation * extrapolation_factor
    )
    return inv_freq.astype(np.float32)


class MotifRotary(nn.Module):
    """RoPE with a selectable frequency schedule (`variant`) and pairing (`rotation`).

    variant  "plain": inv_freq = base**(-arange(0,dims,2)/dims); the forward calls
                      mx.fast.rope(base=...) — BYTE-IDENTICAL to today's
                      nn.RoPE(traditional=..., base=rope_theta).
             "yarn":  inv_freq = transformers YaRN interpolation; the forward calls
                      mx.fast.rope(freqs=1/inv_freq).
    rotation "half":        NeoX half-split pairing   (== nn.RoPE traditional=False).
             "interleaved": adjacent-pair pairing      (== nn.RoPE traditional=True).

    `inv_freq` (fp32, (dims//2,)) and `rotation` are plain attributes, read fresh
    on every forward so an A/B harness can hot-patch either, per layer, on a
    single model load. `inv_freq` is deliberately kept OUT of the module
    parameter tree (assigned via object.__setattr__) so strict weight loading and
    the plain-default parity result are unaffected.

    Note the plain forward uses the `base=` path rather than `freqs=1/inv_freq`
    on purpose: the freqs= path is only equal to nn.RoPE up to ~1e-5 at large
    offsets (fp32 phase accumulation), which would break byte-identical parity;
    `base=` reproduces nn.RoPE exactly. yarn has no `base=` equivalent, so it
    uses freqs=. Both convention checks are in scripts (see the module tests).
    """

    def __init__(self, dims: int, base: float, variant: str, args: "ModelArgs"):
        super().__init__()
        self.dims = dims
        self.base = float(base)
        self.variant = variant
        self.rotation = args.rope_rotation
        if variant == "yarn":
            inv = _rope_inv_freq_yarn(
                dims,
                self.base,
                args.rope_factor,
                args.original_seq_len,
                beta_fast=32.0,
                beta_slow=1.0,
            )
        else:
            inv = _rope_inv_freq_plain(dims, self.base)
        # Plain attribute, NOT an mx parameter: keeps strict load + parity intact
        # while staying hot-patchable (Module.__setattr__ moves it into the param
        # dict on reassignment, which is fine post-load).
        object.__setattr__(self, "inv_freq", mx.array(inv))

    def __call__(self, x, offset=0):
        traditional = self.rotation == "interleaved"
        if self.variant == "plain":
            # byte-identical to nn.RoPE(traditional=traditional, base=rope_theta)
            return mx.fast.rope(
                x,
                self.dims,
                traditional=traditional,
                base=self.base,
                scale=1.0,
                offset=offset,
            )
        # yarn (or any hot-patched inv_freq): drive mx.fast.rope via freqs=1/inv_freq
        freqs = mx.reciprocal(self.inv_freq.astype(mx.float32))
        return mx.fast.rope(
            x,
            self.dims,
            traditional=traditional,
            base=None,
            scale=1.0,
            offset=offset,
            freqs=freqs,
        )

    def extra_repr(self):
        return (
            f"{self.dims}, variant={self.variant!r}, "
            f"rotation={self.rotation!r}, base={self.base}"
        )


class MotifAttention(nn.Module):
    """GDLA: grouped differential latent attention with elementwise output gate."""

    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.n_noise = args.num_noise_heads
        self.group = self.n_heads // self.n_kv_heads  # 5 = 4 signal + 1 noise
        self.n_signal = self.n_heads - self.n_noise  # 64
        self.head_dim = args.head_dim
        self.rope_dim = args.qk_rope_head_dim
        self.nope_dim = args.head_dim - args.qk_rope_head_dim
        self.v_dim = args.v_head_dim
        self.kv_lora_rank = args.kv_lora_rank

        self.is_sliding = args.use_sliding_window and (
            layer_idx % args.sliding_window_period != 0
        )
        self.scale = args.head_dim ** -0.5
        if (
            not self.is_sliding
            and args.max_position_embeddings > args.original_seq_len
        ):
            m = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
            self.scale = self.scale * m * m

        self.wq_a = nn.Linear(dim, args.q_lora_rank, bias=False)
        self.q_norm = nn.RMSNorm(args.q_lora_rank, eps=args.rms_norm_eps)
        self.wq_b = nn.Linear(args.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.wq_b_gate = nn.Linear(
            args.q_lora_rank, self.n_signal * self.v_dim, bias=False
        )
        self.wkv_a = nn.Linear(dim, args.kv_lora_rank + self.rope_dim, bias=False)
        self.kv_norm = nn.RMSNorm(args.kv_lora_rank, eps=args.rms_norm_eps)
        self.wkv_b = nn.Linear(
            args.kv_lora_rank, self.n_kv_heads * (self.nope_dim + self.v_dim), bias=False
        )
        self.lambda_proj = nn.Linear(dim, self.n_signal, bias=False)
        self.wo = nn.Linear(self.n_signal * self.v_dim, dim, bias=False)

        if self.is_sliding:
            self.rope = MotifRotary(
                self.rope_dim, args.swa_rope_theta, "plain", args
            )
        else:
            self.rope = MotifRotary(self.rope_dim, args.rope_theta, "yarn", args)
        self.cs_scale = args.rope_cos_sin_scale

    def __call__(self, x, mask=None, cache=None):
        B, L, _ = x.shape

        ql = self.q_norm(self.wq_a(x))
        q = self.wq_b(ql).reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        gate = self.wq_b_gate(ql).reshape(B, L, self.n_signal, self.v_dim)
        q_nope, q_pe = mx.split(q, [self.nope_dim], axis=-1)

        kv = self.wkv_a(x)
        kv_lat, k_pe = mx.split(kv, [self.kv_lora_rank], axis=-1)
        k_pe = k_pe[:, None]  # (B, 1, L, rope_dim)

        offset = cache.offset if cache is not None else 0
        q_pe = self.rope(q_pe, offset)
        k_pe = self.rope(k_pe, offset)
        if self.cs_scale != 1.0:
            q_pe = q_pe * self.cs_scale
            k_pe = k_pe * self.cs_scale

        kvp = (
            self.wkv_b(self.kv_norm(kv_lat))
            .reshape(B, L, self.n_kv_heads, self.nope_dim + self.v_dim)
            .transpose(0, 2, 1, 3)
        )
        k_nope, v = mx.split(kvp, [self.nope_dim], axis=-1)
        k = mx.concatenate(
            [k_nope, mx.broadcast_to(k_pe, (B, self.n_kv_heads, L, self.rope_dim))],
            axis=-1,
        )
        q = mx.concatenate([q_nope, q_pe], axis=-1)

        if cache is not None:
            k, v = cache.update_and_fetch(k, v)

        o = scaled_dot_product_attention(
            q, k, v, cache=cache, scale=self.scale, mask=mask
        )  # (B, n_heads, L, v_dim)

        # differential combine: group-major heads, noise head last in each group
        o = o.transpose(0, 2, 1, 3).reshape(B, L, self.n_kv_heads, self.group, self.v_dim)
        signal = o[..., : self.group - 1, :].reshape(B, L, self.n_signal, self.v_dim)
        noise = mx.broadcast_to(
            o[..., self.group - 1 :, :],
            (B, L, self.n_kv_heads, self.group - 1, self.v_dim),
        ).reshape(B, L, self.n_signal, self.v_dim)
        lam = mx.sigmoid(self.lambda_proj(x))
        out = signal - lam[..., None] * noise
        out = out * mx.sigmoid(gate)
        return self.wo(out.reshape(B, L, self.n_signal * self.v_dim))


class MotifDecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.self_attn = MotifAttention(args, layer_idx)
        is_moe = (
            layer_idx >= args.n_dense_first_layers
            and args.interleave_moe_layer_step != 0
            and (layer_idx + 1) % args.interleave_moe_layer_step == 0
        )
        self.mlp = MotifMoE(args) if is_moe else MotifMLP(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        _hpc = 1.0 + args.mhc_h_post_alpha_end
        self.mhc_attn = MHCLayer(args.mhc_expansion_rate, args.hidden_size, _hpc)
        self.mhc_ffn = MHCLayer(args.mhc_expansion_rate, args.hidden_size, _hpc)

    def __call__(self, x, mask=None, cache=None):
        # x: (B, L, E, D) — the 4-wide mHC residual stream
        dt = x.dtype

        h_pre, h_post, h_res = self.mhc_attn(x)
        xin = (x * h_pre[..., None]).sum(axis=2).astype(dt)
        a = self.self_attn(self.input_layernorm(xin), mask, cache)
        x = (
            mx.matmul(h_res, x.astype(mx.float32)).astype(dt)
            + (h_post[..., None] * a[:, :, None, :]).astype(dt)
        )

        h_pre, h_post, h_res = self.mhc_ffn(x)
        hin = (x * h_pre[..., None]).sum(axis=2).astype(dt)
        f = self.mlp(self.post_attention_layernorm(hin))
        return (
            mx.matmul(h_res, x.astype(mx.float32)).astype(dt)
            + (h_post[..., None] * f[:, :, None, :]).astype(dt)
        )


class MotifModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            MotifDecoderLayer(args, i) for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.window_size = args.sliding_window + 1  # attends self + sliding_window

    def __call__(self, inputs, cache=None):
        h = self.embed_tokens(inputs)
        B, L, D = h.shape

        if cache is None:
            cache = [None] * len(self.layers)

        # Build each mask from a cache of the MATCHING attention type: full-attn
        # layers use a growing KVCache, SWA layers a RotatingKVCache with a
        # different key length once the sequence exceeds the window. (Layer 0 is
        # full attention under is_sliding = i % period != 0, so cache[0] is NOT a
        # valid source for the windowed mask.)
        first_full = next(
            (i for i, l in enumerate(self.layers) if not l.self_attn.is_sliding), 0
        )
        first_swa = next(
            (i for i, l in enumerate(self.layers) if l.self_attn.is_sliding), 0
        )
        full_mask = create_attention_mask(h, cache[first_full])
        swa_mask = create_attention_mask(h, cache[first_swa], window_size=self.window_size)

        h = mx.broadcast_to(h[:, :, None, :], (B, L, self.args.mhc_expansion_rate, D))
        for layer, c in zip(self.layers, cache):
            h = layer(h, swa_mask if layer.self_attn.is_sliding else full_mask, c)
        return self.norm(h.mean(axis=2))


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = MotifModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs, cache=None):
        return self.lm_head(self.model(inputs, cache))

    def make_cache(self):
        # MOTIF_ROTATING_KV=0 forces plain KVCache on every layer (kv-probe /
        # debugging: rotating caches cannot be quantized).
        if os.environ.get("MOTIF_ROTATING_KV", "1") != "1":
            return [KVCache() for _ in self.layers]
        return [
            RotatingKVCache(max_size=self.model.window_size, keep=0)
            if l.self_attn.is_sliding
            else KVCache()
            for l in self.layers
        ]

    def sanitize(self, weights):
        if any(".mlp.switch_mlp." in k for k in weights):
            return weights  # already MLX layout (from-Q8 rebuild path)

        out = {}
        for k, v in weights.items():
            if k.startswith("model.mtp_layers."):
                continue  # MTP block: present in the checkpoint, never instantiated
            if ".moe." not in k:
                out[k] = v
                continue
            base, moe_part = k.split(".moe.", 1)
            if moe_part == "router.gate.weight":
                out[f"{base}.mlp.gate.weight"] = v
            elif moe_part == "expert_bias":
                out[f"{base}.mlp.gate.expert_bias"] = v
            elif moe_part == "experts.gate_up_proj":
                # gate first (reference chunk order). Deliberately sliced, NOT
                # mx.split: on this 8 GB (>2^31-element) tensor, mx.split
                # returns corrupted data past the 4 GiB offset (32-bit offset
                # overflow, verified empirically at expert 205 of 384); basic
                # strided slices read correctly.
                half = v.shape[1] // 2
                out[f"{base}.mlp.switch_mlp.gate_proj.weight"] = v[:, :half, :]
                out[f"{base}.mlp.switch_mlp.up_proj.weight"] = v[:, half:, :]
            elif moe_part == "experts.down_proj":
                out[f"{base}.mlp.switch_mlp.down_proj.weight"] = v
            elif moe_part.startswith("experts.act_fn."):
                out[f"{base}.mlp.switch_mlp.act_fn.{moe_part[len('experts.act_fn.'):]}"] = v
            elif moe_part.startswith("shared_experts."):
                out[f"{base}.mlp.{moe_part}"] = v
            else:
                raise ValueError(f"unexpected MoE tensor: {k}")
        return out

    @property
    def layers(self):
        return self.model.layers

    @property
    def quant_predicate(self):
        # Safety net for bare `mlx_lm convert -q` runs: the mHC gates and the
        # differential-attention lambda drive mixing decisions and are tiny —
        # keep them full precision. Recipe builds pass an explicit predicate
        # (see ~/motif3/scripts/motif_quant_predicate.py) which supersedes this.
        def predicate(path, module):
            if not hasattr(module, "to_quantized"):
                return False
            if "mhc_" in path or path.endswith("lambda_proj"):
                return False
            return True

        return predicate
