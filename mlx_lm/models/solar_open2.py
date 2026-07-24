# Copyright © 2025 Apple Inc.
#
# mlx-lm port of Upstage Solar Open 2 (250B-A15B hybrid MoE).
#
# Architecture (ground truth: upstageAI/transformers @ v5.14.1-solar-open2,
# src/transformers/models/solar_open2/modeling_solar_open2.py):
#   * 48 decoder layers; explicit ``gqa_layers`` (0,4,8,...,44) use NoPE
#     grouped-query full attention, the other 36 use Kimi-Delta-Attention
#     (KDA, gated delta rule with a per-(head, dim) vector decay).
#   * Every layer past ``first_k_dense_replace`` is a MoE layer:
#     320 routed experts (top-8, sigmoid router with grouped top-k and a
#     non-trainable e_score_correction_bias) + 1 shared expert.
#
# Reuse strategy: structure follows mlx_lm/models/kimi_linear.py (same KDA
# family) with the Solar-specific gate math (lower-bounded log-decay gate,
# optional beta x2 for negative eigenvalues) and a plain NoPE GQA layer
# instead of MLA. The recurrent metal kernel / ops fallbacks are imported
# from mlx_lm/models/gated_delta.py; the MoE feed-forward is
# mlx_lm/models/switch_layers.py:SwitchGLU so QuantizedSwitchGLU conversion
# and alis-dwq's SwitchGLU hooks keep working unchanged.

from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .activations import swiglu
from .base import (
    BaseModelArgs,
    create_attention_mask,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from .cache import ArraysCache, KVCache
from .gated_delta import gated_delta_kernel, gated_delta_ops
from .rope_utils import initialize_rope
from .switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    vocab_size: int = 196608
    hidden_size: int = 4096
    num_hidden_layers: int = 48
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 10240
    moe_intermediate_size: int = 1280
    rms_norm_eps: float = 1e-5
    # Kept for checkpoint-config compatibility; unused unless use_rope=True.
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict[str, Any]] = None
    partial_rotary_factor: float = 1.0  # was silently dropped by from_dict
    max_position_embeddings: int = 1048576
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    # Full-attention (GQA) layer options.
    use_rope: bool = False  # NoPE by default
    use_qk_norm: bool = False
    use_gqa_gate: bool = True
    use_gqa_gate_bias: bool = False
    gqa_interval: int = 4
    gqa_layers: Optional[List[int]] = None  # takes priority over gqa_interval
    # KDA (linear attention) options.
    linear_attn_config: Optional[Dict[str, Any]] = None
    kda_use_full_proj: bool = False
    kda_gate_lower_bound: Optional[float] = -5.0
    # HF SolarOpen2Config's *code* default is True (configuration:145; its
    # docstring wrongly says False); the shipped checkpoint config sets it
    # explicitly to true, so real-model behavior is unchanged either way.
    # Match the HF code default.
    kda_allow_neg_eigval: bool = True
    # MoE options.
    n_routed_experts: int = 320
    n_shared_experts: int = 1
    num_experts_per_tok: int = 8
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 1.0
    n_group: int = 1
    topk_group: int = 1
    first_k_dense_replace: int = 0
    hidden_act: str = "silu"  # SwitchGLU hard-codes SwiGLU; silu-only port

    def __post_init__(self):
        if self.hidden_act != "silu":
            raise ValueError(
                f"hidden_act={self.hidden_act!r} is not supported: SwitchGLU "
                "and SolarMLP hard-code SwiGLU (silu-only port)."
            )
        if self.linear_attn_config is None:
            self.linear_attn_config = {
                "short_conv_kernel_size": 4,
                "head_dim": self.head_dim,
                "num_heads": self.num_attention_heads,
                "num_kv_heads": None,
            }
        # Per-layer attention pattern, mirroring SolarOpen2Config.__post_init__.
        if self.gqa_layers is not None:
            full = set(self.gqa_layers)
            self.layer_types = [
                "full_attention" if i in full else "linear_attention"
                for i in range(self.num_hidden_layers)
            ]
        else:
            self.layer_types = [
                (
                    "full_attention"
                    if (i + 1) % self.gqa_interval == 0
                    else "linear_attention"
                )
                for i in range(self.num_hidden_layers)
            ]


# --------------------------------------------------------------------------
# KDA helpers
# --------------------------------------------------------------------------


@partial(mx.compile, shapeless=True)
def _kda_decay_gate(
    g_raw: mx.array,
    A_log: mx.array,
    dt_bias: mx.array,
    lower_bound: Optional[float],
) -> mx.array:
    """Solar Open 2 KDA log-decay gate (reference: ``torch_kda_gate`` +
    ``fla.ops.kda.gate.fused_kda_gate``).

    ``g = -exp(A_log) * softplus(g_raw + dt_bias)`` computed in float32,
    clamped from below at ``lower_bound`` (-5.0 by default), then
    exponentiated once so the result is the multiplicative per-(head, dim)
    decay consumed by ``gated_delta_ops`` / ``gated_delta_kernel``.

    Differs from kimi_linear's ``compute_g``: that variant has no lower-bound
    clamp because Kimi Linear leaves the gate unclamped.
    """
    g = g_raw.astype(mx.float32) + dt_bias.astype(mx.float32)
    g = -mx.exp(A_log.astype(mx.float32)) * nn.softplus(g)
    if lower_bound is not None:
        g = mx.maximum(g, mx.array(lower_bound, dtype=mx.float32))
    return mx.exp(g)


@partial(mx.compile, shapeless=True)
def _l2norm(x: mx.array, eps: float) -> mx.array:
    """x * rsqrt(sum(x^2) + eps) computed in float32 (HF l2norm reference)."""
    xf = x.astype(mx.float32)
    return xf * mx.rsqrt((xf * xf).sum(axis=-1, keepdims=True) + eps)


class SolarRMSNormGated(nn.Module):
    """Gated RMSNorm at the KDA output: ``RMSNorm(x) * sigmoid(gate)``,
    computed in float32 like ``SolarOpen2RMSNormGated`` /
    ``fla.modules.FusedRMSNormGated(activation="sigmoid")``."""

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((hidden_size,))
        self.eps = eps

    def __call__(self, x: mx.array, gate: mx.array) -> mx.array:
        return _gated_rms_norm(x, self.weight, gate, self.eps)


@partial(mx.compile, shapeless=True)
def _gated_rms_norm(x, weight, gate, eps):
    xf = x.astype(mx.float32)
    variance = (xf * xf).mean(axis=-1, keepdims=True)
    xf = xf * mx.rsqrt(variance + eps)
    xf = xf * weight.astype(mx.float32) * mx.sigmoid(gate.astype(mx.float32))
    return xf.astype(x.dtype)


class ShortConv1d(nn.Module):
    """Depthwise causal short conv (silu, no bias) with a (kernel - 1)-token
    recurrent state. Identical to kimi_linear.ShortConv1d; kept local so the
    port is self-contained."""

    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            bias=False,
            groups=channels,
            padding=0,
        )

    def __call__(
        self,
        x: mx.array,
        state: Optional[mx.array],
        mask: Optional[mx.array],
        lengths: Optional[mx.array],
    ) -> Tuple[mx.array, mx.array]:
        if mask is not None:
            x = mx.where(mask[..., None], x, 0)

        if state is None:
            state = mx.zeros(
                (x.shape[0], self.kernel_size - 1, x.shape[-1]), dtype=x.dtype
            )
        conv_input = mx.concatenate([state, x], axis=1)
        out = nn.silu(self.conv(conv_input))
        n_keep = self.kernel_size - 1
        if lengths is not None:
            ends = mx.clip(lengths, 0, x.shape[1])
            positions = (ends[:, None] + mx.arange(n_keep))[..., None]
            new_state = mx.take_along_axis(conv_input, positions, axis=1)
        else:
            new_state = mx.contiguous(conv_input[:, -n_keep:, :])

        return out, new_state


class SolarDeltaAttention(nn.Module):
    """Kimi-Delta-Attention layer as configured for Solar Open 2.

    Differences vs kimi_linear.KimiDeltaAttention:
      * optional GQA-style kv heads (``linear_attn_config.num_kv_heads``;
        Solar Open 2 sets it to null -> equal to num_heads),
      * ``kda_use_full_proj`` switches between full-rank ``f_proj``/``g_proj``
        and the factored low-rank ``f_a_proj``/``f_b_proj`` /
        ``g_a_proj``/``g_b_proj`` (Solar Open 2 uses the factored form),
      * the decay gate is lower-bounded (``kda_gate_lower_bound=-5.0``),
      * ``kda_allow_neg_eigval`` scales beta by 2 (beta in [0, 2]).
    """

    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        cfg = args.linear_attn_config

        self.layer_idx = layer_idx
        self.num_heads = cfg["num_heads"]
        self.head_dim = cfg["head_dim"]
        self.num_kv_heads = cfg.get("num_kv_heads") or self.num_heads
        self.n_rep = self.num_heads // self.num_kv_heads
        self.conv_kernel = cfg.get("short_conv_kernel_size", 4)
        self.use_full_proj = args.kda_use_full_proj
        self.gate_lower_bound = args.kda_gate_lower_bound
        self.allow_neg_eigval = args.kda_allow_neg_eigval

        self.projection_dim = self.num_heads * self.head_dim
        self.kv_projection_dim = self.num_kv_heads * self.head_dim
        hidden = args.hidden_size

        self.scale = float(self.head_dim) ** -0.5

        self.q_proj = nn.Linear(hidden, self.projection_dim, bias=False)
        self.k_proj = nn.Linear(hidden, self.kv_projection_dim, bias=False)
        self.v_proj = nn.Linear(hidden, self.kv_projection_dim, bias=False)
        self.b_proj = nn.Linear(hidden, self.num_heads, bias=False)

        self.q_conv = ShortConv1d(self.projection_dim, self.conv_kernel)
        self.k_conv = ShortConv1d(self.kv_projection_dim, self.conv_kernel)
        self.v_conv = ShortConv1d(self.kv_projection_dim, self.conv_kernel)

        if self.use_full_proj:
            self.f_proj = nn.Linear(hidden, self.projection_dim, bias=False)
            self.g_proj = nn.Linear(hidden, self.projection_dim, bias=False)
        else:
            self.f_a_proj = nn.Linear(hidden, self.head_dim, bias=False)
            self.f_b_proj = nn.Linear(self.head_dim, self.projection_dim, bias=False)
            self.g_a_proj = nn.Linear(hidden, self.head_dim, bias=False)
            self.g_b_proj = nn.Linear(self.head_dim, self.projection_dim, bias=False)

        # Shapes match the HF checkpoint: A_log (1, 1, H, 1), dt_bias (H*Dh,).
        self.A_log = mx.expand_dims(
            mx.log(mx.random.uniform(low=1.0, high=16.0, shape=(self.num_heads,))),
            (0, 1, 3),
        )
        self.dt_bias = mx.ones((self.projection_dim,))

        self.o_norm = SolarRMSNormGated(self.head_dim, eps=args.rms_norm_eps)
        self.o_proj = nn.Linear(self.projection_dim, hidden, bias=False)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, T, _ = x.shape
        dtype = x.dtype

        if cache is not None:
            q_state, k_state, v_state, ssm_state = cache
            lengths = cache.lengths
        else:
            q_state = k_state = v_state = ssm_state = None
            lengths = None

        q_conv, q_state = self.q_conv(self.q_proj(x), q_state, mask, lengths)
        k_conv, k_state = self.k_conv(self.k_proj(x), k_state, mask, lengths)
        v_conv, v_state = self.v_conv(self.v_proj(x), v_state, mask, lengths)

        if cache is not None:
            cache[0] = q_state
            cache[1] = k_state
            cache[2] = v_state

        q = q_conv.reshape(B, T, self.num_heads, self.head_dim)
        k = k_conv.reshape(B, T, self.num_kv_heads, self.head_dim)
        v = v_conv.reshape(B, T, self.num_kv_heads, self.head_dim)

        # l2-normalize q/k exactly like the reference
        # (use_qk_l2norm_in_kernel=True): x * rsqrt(sum(x^2) + eps) in f32.
        # Folding this through mx.fast.rms_norm is NOT equivalent: rms_norm
        # puts eps under the mean (i.e. D*eps on the sum), which measurably
        # shifts small-norm head vectors (validated on real weights, see
        # VALIDATION_RESULTS.md). q/k/v go into the recurrence in f32,
        # mirroring the f32 casts in torch_recurrent_kda.
        q = _l2norm(q, 1e-6) * self.scale
        k = _l2norm(k, 1e-6)
        v = v.astype(mx.float32)

        if self.use_full_proj:
            g_raw = self.f_proj(x)
        else:
            g_raw = self.f_b_proj(self.f_a_proj(x))
        g_raw = g_raw.reshape(B, T, self.num_heads, self.head_dim)

        beta = mx.sigmoid(self.b_proj(x))
        if self.allow_neg_eigval:
            beta = beta * 2.0

        g = _kda_decay_gate(
            g_raw,
            self.A_log.reshape(self.num_heads, 1),
            self.dt_bias.reshape(self.num_heads, self.head_dim),
            self.gate_lower_bound,
        )

        if ssm_state is None:
            ssm_state = mx.zeros(
                (B, self.num_kv_heads, self.head_dim, self.head_dim),
                dtype=mx.float32,
            )

        if (
            self.training
            or mx.default_device() != mx.gpu
            or not mx.metal.is_available()
        ):
            out, ssm_state = gated_delta_ops(q, k, v, g, beta, ssm_state, mask)
        else:
            out, ssm_state = gated_delta_kernel(q, k, v, g, beta, ssm_state, mask)

        if cache is not None:
            cache[3] = ssm_state
            cache.advance(T)

        if self.use_full_proj:
            g_out = self.g_proj(x)
        else:
            g_out = self.g_b_proj(self.g_a_proj(x))
        g_out = g_out.reshape(B, T, self.num_heads, self.head_dim)

        out = self.o_norm(out.reshape(B, T, self.num_heads, self.head_dim), g_out)
        out = out.reshape(B, T, -1).astype(dtype)
        return self.o_proj(out)


class SolarFullAttention(nn.Module):
    """Grouped-query full attention: NoPE by default (use_rope=False),
    optional per-head qk RMSNorm (off in Solar Open 2), and a sigmoid output
    gate (``use_gqa_gate=True``: ``attn_out * sigmoid(g_proj(x))``)."""

    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = args.num_attention_heads
        self.num_key_value_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = float(self.head_dim) ** -0.5
        self.use_gqa_gate = args.use_gqa_gate
        self.use_qk_norm = args.use_qk_norm
        self.use_rope = args.use_rope

        hidden = args.hidden_size
        self.q_proj = nn.Linear(
            hidden, self.num_heads * self.head_dim, bias=args.attention_bias
        )
        self.k_proj = nn.Linear(
            hidden, self.num_key_value_heads * self.head_dim, bias=args.attention_bias
        )
        self.v_proj = nn.Linear(
            hidden, self.num_key_value_heads * self.head_dim, bias=args.attention_bias
        )
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden, bias=False)

        if self.use_qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
            self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)

        if self.use_rope:
            self.rope = initialize_rope(
                int(self.head_dim * args.partial_rotary_factor),
                base=args.rope_theta,
                traditional=False,
                scaling_config=args.rope_scaling,
                max_position_embeddings=args.max_position_embeddings,
            )
        else:
            self.rope = None

        if self.use_gqa_gate:
            self.g_proj = nn.Linear(
                hidden,
                self.num_heads * self.head_dim,
                bias=args.use_gqa_gate_bias,
            )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        q = self.q_proj(x).reshape(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(B, L, self.num_key_value_heads, self.head_dim)
        v = self.v_proj(x).reshape(B, L, self.num_key_value_heads, self.head_dim)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        if self.rope is not None:
            offset = cache.offset if cache is not None else 0
            q = self.rope(q, offset=offset)
            k = self.rope(k, offset=offset)

        if cache is not None:
            k, v = cache.update_and_fetch(k, v)

        out = scaled_dot_product_attention(q, k, v, cache=cache, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)

        if self.use_gqa_gate:
            out = out * mx.sigmoid(self.g_proj(x))

        return self.o_proj(out)


# --------------------------------------------------------------------------
# MoE
# --------------------------------------------------------------------------


class SolarMLP(nn.Module):
    def __init__(
        self,
        args: ModelArgs,
        hidden_size: Optional[int] = None,
        intermediate_size: Optional[int] = None,
    ):
        super().__init__()
        dim = hidden_size or args.hidden_size
        hidden = intermediate_size or args.intermediate_size
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


@mx.compile
def _expert_select(
    gates: mx.array,
    bias: Optional[mx.array],
    top_k: int,
    n_group: int,
    topk_group: int,
    routed_scaling_factor: float,
    renormalize: bool,
) -> Tuple[mx.array, mx.array]:
    """Sigmoid router with optional grouped top-k (SolarOpen2TopkRouter).

    Selection uses bias-adjusted scores; the returned weights are the raw
    sigmoid scores at the selected experts (bias only affects *which* experts
    are picked), optionally renormalized (``norm_topk_prob``) and scaled by
    ``routed_scaling_factor``.
    """
    scores = mx.sigmoid(gates)
    orig_scores = scores
    if bias is not None:
        scores = scores + bias.astype(scores.dtype)

    if n_group > 1:
        scores = mx.unflatten(scores, axis=-1, shape=(n_group, -1))
        group_scores = mx.topk(scores, 2, axis=-1).sum(axis=-1, keepdims=True)
        k = n_group - topk_group
        group_idx = mx.argpartition(group_scores, kth=k - 1, axis=-2)[..., :k, :]
        # HF masks dropped groups with -inf (masked_fill), not 0.0: with a
        # negative e_score_correction_bias an unmasked adjusted score can be
        # < 0, and a 0.0 mask would then pick a MASKED expert HF never picks
        # (attacks/attack6 A6a).
        scores = mx.put_along_axis(
            scores,
            mx.stop_gradient(group_idx),
            mx.array(-float("inf"), dtype=scores.dtype),
            axis=-2,
        )
        scores = mx.flatten(scores, -2, -1)

    inds = mx.argpartition(-scores, kth=top_k - 1, axis=-1)[..., :top_k]
    weights = mx.take_along_axis(orig_scores, inds, axis=-1)

    if top_k > 1 and renormalize:
        weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-20)

    return inds, weights * routed_scaling_factor


class SolarSparseMoE(nn.Module):
    """Routed MoE + shared expert.

    Naming contract (kept identical to kimi_linear so alis-dwq's hooks match):
      * ``gate``        -- router Linear; module path ``...mlp.gate`` matches
                           alis-dwq's ``(?:^|\.)(?:gate|router)$`` router regex
      * ``switch_mlp``  -- SwitchGLU instance (expert_traffic hooks the class)
      * ``shared_experts``
      * ``e_score_correction_bias`` lives on this module (not on ``gate``) so
        an 8-bit quantization of ``gate`` cannot drop it; ``sanitize`` moves
        the HF key ``...mlp.gate.e_score_correction_bias`` here.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        hidden = args.hidden_size
        experts = args.n_routed_experts

        self.gate = nn.Linear(hidden, experts, bias=False)
        self.switch_mlp = SwitchGLU(hidden, args.moe_intermediate_size, experts)
        self.e_score_correction_bias = mx.zeros((experts,), dtype=mx.float32)

        if args.n_shared_experts:
            shared_hidden = args.moe_intermediate_size * args.n_shared_experts
            self.shared_experts = SolarMLP(args, intermediate_size=shared_hidden)
        else:
            self.shared_experts = None

    def __call__(self, x: mx.array) -> mx.array:
        # HF computes the router logits in float32:
        #   F.linear(hidden_states.float(), gate.weight.float())
        # Mirror that exactly — a bf16 matmul's reduction-order noise flips
        # razor-tie routing decisions (attacks/attack1b: 7/64 tokens flipped
        # vs HF f32 logits; both-sides-f32 gives 0/64). A quantized router
        # (packed uint32 weight) cannot upcast, so that path keeps the
        # activation-dtype matmul with an f32 cast of the logits.
        if hasattr(self.gate, "bits"):
            gates = self.gate(x).astype(mx.float32)
        else:
            gates = x.astype(mx.float32) @ self.gate.weight.astype(mx.float32).T
        inds, weights = _expert_select(
            gates,
            self.e_score_correction_bias,
            self.args.num_experts_per_tok,
            self.args.n_group,
            self.args.topk_group,
            self.args.routed_scaling_factor,
            self.args.norm_topk_prob,
        )
        out = self.switch_mlp(x, inds)
        # HF casts the weighted expert sum back to the activation dtype.
        out = (out * weights[..., None]).sum(axis=-2).astype(x.dtype)
        if self.shared_experts is not None:
            out = out + self.shared_experts(x)
        return out


class SolarDecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.is_linear = args.layer_types[layer_idx] == "linear_attention"

        if self.is_linear:
            self.self_attn = SolarDeltaAttention(args, layer_idx)
        else:
            self.self_attn = SolarFullAttention(args, layer_idx)

        if layer_idx >= args.first_k_dense_replace:
            self.mlp = SolarSparseMoE(args)
        else:
            self.mlp = SolarMLP(args)

        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        y = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + y
        z = self.mlp(self.post_attention_layernorm(h))
        return h + z


class SolarOpen2Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [SolarDecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.ssm_idx = None
        self.attn_idx = None
        for i, layer in enumerate(self.layers):
            if layer.is_linear:
                self.ssm_idx = i
                break
        for i, layer in enumerate(self.layers):
            if not layer.is_linear:
                self.attn_idx = i
                break

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[List[Any]] = None,
    ) -> mx.array:
        h = self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)

        ssm_mask = (
            create_ssm_mask(h, cache[self.ssm_idx])
            if self.ssm_idx is not None
            else None
        )
        attn_mask = (
            create_attention_mask(h, cache[self.attn_idx])
            if self.attn_idx is not None
            else None
        )

        for layer, layer_cache in zip(self.layers, cache):
            mask = ssm_mask if layer.is_linear else attn_mask
            h = layer(h, mask=mask, cache=layer_cache)

        return self.norm(h)


def sanitize_layer_weights(
    weights: Dict[str, mx.array],
    prefix: str,
    layer: "SolarDecoderLayer",
    num_experts: int,
) -> Dict[str, mx.array]:
    """In-place HF -> MLX key conversion for one decoder layer.

    Handles both on-disk expert layouts:
      * per-expert tensors (the actual upstage/Solar-Open2-250B checkpoint):
        ``{prefix}.mlp.experts.{i}.{gate,up,down}_proj.weight`` -> stacked
        ``switch_mlp.{gate,up,down}_proj.weight``;
      * fused 3D tensors (module layout of the HF modeling code):
        ``experts.gate_up_proj`` (E, 2I, H) split into gate/up, and
        ``experts.down_proj`` renamed.
    """
    if isinstance(layer.mlp, SolarSparseMoE):
        mlp_prefix = f"{prefix}.mlp"

        if f"{mlp_prefix}.experts.0.gate_proj.weight" in weights:
            # Per-expert layout: refuse silently-misconverted pre-quantized
            # tensors (scales/biases companions would be left behind and the
            # stacked uint32 "weights" would be nonsense).
            if any(
                k.startswith(f"{mlp_prefix}.experts.")
                and (k.endswith(".scales") or k.endswith(".biases"))
                for k in weights
            ):
                raise NotImplementedError(
                    "Pre-quantized HF expert tensors are not supported by "
                    "this port; load the BF16 checkpoint and quantize "
                    "with mlx-lm instead."
                )
            for name in ("gate_proj", "up_proj", "down_proj"):
                stacked = [
                    weights.pop(f"{mlp_prefix}.experts.{i}.{name}.weight")
                    for i in range(num_experts)
                ]
                weights[f"{mlp_prefix}.switch_mlp.{name}.weight"] = mx.stack(stacked)
        else:
            if f"{mlp_prefix}.experts.gate_up_proj.scales" in weights:
                raise NotImplementedError(
                    "Pre-quantized HF expert tensors are not supported by "
                    "this port; load the BF16 checkpoint and quantize "
                    "with mlx-lm instead."
                )
            gu_key = f"{mlp_prefix}.experts.gate_up_proj"
            if gu_key in weights:
                gu = weights.pop(gu_key)
                # mlx#3836: mx.split silently corrupts >2**31-element tensors;
                # a fused HF-layout gate_up_proj (E, 2I, H) is 1.56x that for
                # this model. Use strided slices instead.
                half = gu.shape[1] // 2
                gate_w, up_w = gu[:, :half], gu[:, half:]
                weights[f"{mlp_prefix}.switch_mlp.gate_proj.weight"] = mx.contiguous(
                    gate_w
                )
                weights[f"{mlp_prefix}.switch_mlp.up_proj.weight"] = mx.contiguous(
                    up_w
                )
            down_key = f"{mlp_prefix}.experts.down_proj"
            if down_key in weights:
                weights[f"{mlp_prefix}.switch_mlp.down_proj.weight"] = weights.pop(
                    down_key
                )

        # Router bias lives on the MoE block in MLX (see class note).
        bias_key = f"{mlp_prefix}.gate.e_score_correction_bias"
        if bias_key in weights:
            weights[f"{mlp_prefix}.e_score_correction_bias"] = weights.pop(bias_key)

    attn = layer.self_attn
    if isinstance(attn, SolarDeltaAttention):
        attn_prefix = f"{prefix}.self_attn"
        for src_name, dst_name in (
            ("q_conv1d", "q_conv"),
            ("k_conv1d", "k_conv"),
            ("v_conv1d", "v_conv"),
        ):
            src_key = f"{attn_prefix}.{src_name}.weight"
            if src_key in weights:
                w = weights.pop(src_key)
                if w.ndim == 3:
                    # HF depthwise conv weight (C, 1, K) ->
                    # MLX Conv1d weight (C, K, 1)
                    w = w.moveaxis(2, 1)
                weights[f"{attn_prefix}.{dst_name}.conv.weight"] = w
        dt_key = f"{attn_prefix}.dt_bias"
        if dt_key in weights and weights[dt_key].ndim > 1:
            weights[dt_key] = mx.reshape(weights[dt_key], (-1,))

    return weights


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = SolarOpen2Model(args)
        if args.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[List[Any]] = None,
    ) -> mx.array:
        out = self.model(inputs, cache)
        if self.lm_head is None:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        caches: List[Any] = []
        for layer in self.layers:
            if layer.is_linear:
                # [q_conv_state, k_conv_state, v_conv_state, recurrent_state]
                caches.append(ArraysCache(size=4))
            else:
                caches.append(KVCache())
        return caches

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        # Drop non-persistent HF rope buffers if a checkpoint carries them.
        weights = {k: v for k, v in weights.items() if "rotary_emb.inv_freq" not in k}

        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        for layer_idx, layer in enumerate(self.layers):
            sanitize_layer_weights(
                weights,
                f"model.layers.{layer_idx}",
                layer,
                self.args.n_routed_experts,
            )

        return weights

    @property
    def cast_predicate(self):
        def predicate(path: str):
            if "e_score_correction_bias" in path:
                return False
            if path.endswith("A_log") or path.endswith("dt_bias"):
                return False
            return True

        return predicate

    @property
    def quant_predicate(self):
        def predicate(path, _):
            # Keep the router at 8 bits: routing is selection-sensitive and
            # the tensor is tiny (320 x 4096 per layer).
            if path.endswith("mlp.gate"):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate
