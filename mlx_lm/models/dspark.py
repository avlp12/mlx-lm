# Copyright © 2026 Apple Inc.

"""DSpark / DFlash block-diffusion draft model.

Unlike EAGLE- or MTP-style drafters, this one does **not** produce its tokens
autoregressively: an entire block of `block_size` positions is drafted in a
single forward pass. The block starts as MASK-token embeddings and the drafter
denoises it in one shot, attending bidirectionally within the block. That makes
the draft cost one small-model forward per block instead of `k` sequential ones,
which is what makes it worth its parameters on a latency-bound machine.

Two sources feed every attention layer:
  * the block itself (queries, plus its own keys/values), and
  * a context stream projected from *intermediate* hidden states of the target
    model, taken at `dflash_config.target_layer_ids` and concatenated.

The drafter carries no embedding or output head; it borrows the target's
`embed_tokens` and `lm_head`, so its vocabulary is the target's by construction.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs
from .cache import KVCache
from .rope_utils import initialize_rope


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "dspark"
    hidden_size: int = 5120
    intermediate_size: int = 10240
    num_hidden_layers: int = 5
    num_attention_heads: int = 40
    num_key_value_heads: int = 8
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    vocab_size: int = 248320
    max_position_embeddings: int = 262144
    attention_bias: bool = False
    block_size: int = 7
    num_target_layers: int = 64
    markov_rank: int = 0
    enable_confidence_head: bool = False
    confidence_head_with_markov: bool = True
    dflash_config: Dict[str, Any] = field(default_factory=dict)
    rope_parameters: Optional[Dict[str, Any]] = None
    rope_theta: float = 10000000.0

    def __post_init__(self):
        if self.rope_parameters:
            self.rope_theta = self.rope_parameters.get("rope_theta", self.rope_theta)


class Attention(nn.Module):
    """Dual-source attention: queries from the block, keys/values from
    context ++ block."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5

        b = args.attention_bias
        self.q_proj = nn.Linear(args.hidden_size, self.n_heads * self.head_dim, bias=b)
        self.k_proj = nn.Linear(args.hidden_size, self.n_kv_heads * self.head_dim, bias=b)
        self.v_proj = nn.Linear(args.hidden_size, self.n_kv_heads * self.head_dim, bias=b)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, args.hidden_size, bias=b)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)

        rope_scaling = args.rope_parameters or {}
        self.rope = initialize_rope(
            self.head_dim,
            args.rope_theta,
            traditional=False,
            scaling_config=rope_scaling if rope_scaling.get("rope_type") else None,
            max_position_embeddings=args.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        target_hidden: mx.array,
        k_offset: int,
        q_offset: int,
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        B, L, _ = x.shape
        C = target_hidden.shape[1]

        q = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim)
        q = self.q_norm(q).transpose(0, 2, 1, 3)

        # The context tokens are keys/values only; they never produce queries.
        kv_in = mx.concatenate([target_hidden, x], axis=1)
        k = self.k_proj(kv_in).reshape(B, C + L, self.n_kv_heads, self.head_dim)
        k = self.k_norm(k).transpose(0, 2, 1, 3)
        v = self.v_proj(kv_in).reshape(B, C + L, self.n_kv_heads, self.head_dim)
        v = v.transpose(0, 2, 1, 3)

        # q sits at the tail of the same position range k spans, so both are
        # contiguous and a plain offset suffices for each.
        q = self.rope(q, offset=q_offset)
        k = self.rope(k, offset=k_offset)

        if cache is not None:
            k, v = cache.update_and_fetch(k, v)

        # No mask: context is entirely in the past, and the block is meant to
        # attend to itself in both directions (this is a diffusion step, not an
        # autoregressive one).
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out)


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.self_attn = Attention(args)
        self.mlp = MLP(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(self, x, target_hidden, k_offset, q_offset, cache=None):
        h = x + self.self_attn(
            self.input_layernorm(x), target_hidden, k_offset, q_offset, cache
        )
        return h + self.mlp(self.post_attention_layernorm(h))


class MarkovHead(nn.Module):
    """Low-rank learned bigram bias added to the draft logits."""

    def __init__(self, vocab_size: int, rank: int):
        super().__init__()
        self.markov_w1 = nn.Embedding(vocab_size, rank)
        self.markov_w2 = nn.Linear(rank, vocab_size, bias=False)

    def __call__(self, prev_tokens: mx.array) -> mx.array:
        return self.markov_w2(self.markov_w1(prev_tokens))


class ConfidenceHead(nn.Module):
    """Predicts a per-position acceptance probability, for adaptive blocks."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, 1)

    def __call__(self, features: mx.array) -> mx.array:
        return self.proj(features).squeeze(-1)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.layers = [DecoderLayer(args) for _ in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.hidden_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        self.target_layer_ids: List[int] = args.dflash_config.get("target_layer_ids", [])
        self.fc = nn.Linear(
            len(self.target_layer_ids) * args.hidden_size, args.hidden_size, bias=False
        )
        self.block_size = args.block_size
        self.mask_token_id = args.dflash_config.get("mask_token_id")

        if args.markov_rank:
            self.markov_head = MarkovHead(args.vocab_size, args.markov_rank)
        if args.enable_confidence_head:
            dim = args.hidden_size + (
                args.markov_rank if args.confidence_head_with_markov else 0
            )
            self.confidence_head = ConfidenceHead(dim)

    def make_cache(self):
        return [KVCache() for _ in self.layers]

    def __call__(
        self,
        noise_embedding: mx.array,
        target_hidden: mx.array,
        k_offset: int,
        q_offset: int,
        cache: Optional[List[KVCache]] = None,
    ) -> mx.array:
        """`target_hidden` is the raw concatenation of the tapped target layers."""
        ctx = self.hidden_norm(self.fc(target_hidden))
        h = noise_embedding
        cache = cache or [None] * len(self.layers)
        for layer, c in zip(self.layers, cache):
            h = layer(h, ctx, k_offset, q_offset, c)
        return self.norm(h)


def extract_context_feature(hidden_states: List[mx.array], layer_ids: List[int]):
    """Concatenate the tapped target hidden states.

    The +1 offset matches the reference: index 0 of a Transformers
    `hidden_states` tuple is the embedding output, so layer `i` lives at `i + 1`.
    Callers that collect post-layer outputs directly must pass the same
    convention.
    """
    return mx.concatenate([hidden_states[i + 1] for i in layer_ids], axis=-1)


def load_dspark(path: str):
    """Load a DSpark drafter from a local directory."""
    path = Path(path)
    with open(path / "config.json") as f:
        cfg = json.load(f)
    args = ModelArgs.from_dict(cfg)
    model = Model(args)
    weights = mx.load(str(path / "model.safetensors"))
    model.load_weights(list(weights.items()), strict=True)
    model.eval()
    return model, args
