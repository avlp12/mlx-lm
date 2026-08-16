# Copyright © 2026 Apple Inc.

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import shard_inplace, shard_linear, sum_gradients
from mlx.utils import tree_map

from .activations import swiglu
from .base import (
    BaseModelArgs,
    create_attention_mask,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from .cache import ArraysCache, KVCache
from .gated_delta import gated_delta_update
from .pipeline import PipelineMixin
from .qwen3_next import Qwen3NextAttention as _Qwen3NextAttention
from .qwen3_next import Qwen3NextMLP as _Qwen3NextMLP
from .qwen3_next import Qwen3NextRMSNormGated as RMSNormGated
from .qwen3_next import Qwen3NextSparseMoeBlock as SparseMoeBlock

# ── 글루-융합 1단계 (~/qwen38/glue_fusion_plan.md §2.1) ──────────────────────
# 디코드 스텝은 대역폭 바닥(21.7ms) 위에 ~1600개 launch 의 직렬 디스패치 갭이
# 얹힌 구조라, 소형-텐서 부기를 연접·컴파일로 줄이는 것이 유일한 회수 수단이다.
# 대형 GEMM 은 절대 융합체 안에 넣지 않는다([I49]/[I51]: MLP GEMM 은 이미 대역폭
# 상한. Motif v1 메가커널 침몰 원인).
#
# 킬스위치는 import 시 1회 판독(Motif 관례). A/B 는 반드시 **별도 프로세스** —
# 같은 프로세스 토글은 compile 캐시·fast_qmm 클래스 패치 잔존으로 무효(§5-R2).
#   QWEN35_FUSED_PROJ=0   … §2.1(a) 가중치-연접(GDN in_proj 4→1, ATTN qkv 3→1,
#                          MLP gate/up 2→1) 비활성. launch −240/스텝, S=1~4 비트 동일.
#   QWEN35_COMPILE_GLUE=0 … §2.1(b) 중 생존 항목 = q/k 스칼라 접기(launch −96/스텝,
#                          비트 동일 실측 — fast.rms_norm 이 정규화값을 출력 dtype 으로
#                          라운딩한 뒤 weight 를 곱해 eager 이중-라운딩과 일치).
#
# §2.1(b)의 βg 융합·ATTN 에필로그는 **실측 no-op 으로 판명되어 제거**(2026-08-16
# census): mx.compile 은 연결되지 않은 출력(sigmoid(b) ↔ g-사슬)을 한 커널로
# 묶지 않고, compute 연산이 1개뿐인 서브그래프(transpose/reshape+mul)는 융합
# 커널 없이 원 프리미티브로 방출한다. exp5_fusion 원장 [J#] 참조.
_FUSED_PROJ = os.environ.get("QWEN35_FUSED_PROJ", "0") == "1"
_COMPILE_GLUE = os.environ.get("QWEN35_COMPILE_GLUE", "0") == "1"


class _ConcatQuantizedLinear(nn.QuantizedLinear):
    """N-축으로 연접한 QuantizedLinear.

    반드시 nn.QuantizedLinear 의 서브클래스여야 한다: fast_qmm.enable() 이
    `nn.QuantizedLinear.__call__` 을 클래스 패치하므로, __call__ 을 정의하지
    않는 서브클래스는 검증 폭 6~8 의 split-K 커널을 자동으로 탄다(§5-R2).
    4bit/g64 affine 의 양자화 파라미터(scales/biases)는 출력-행 단위라 N-축
    연접은 행별 연산을 바꾸지 않는다 — 수치 항등(오라클 T1 로 확인).
    인스턴스는 `_concat_quantized` 로만 만든다.
    """


def _concat_quantized(mods):
    """같은 입력을 받는 QuantizedLinear 들을 N-축 연접한 모듈로. 부적격이면 None.

    엄격한 타입 검사(서브클래스 불허)로 샤딩 래퍼(Quantized*ShardedLinear)를
    걸러낸다 — TP 경로는 레거시 유지(§5-R8).
    """
    if not all(type(m) is nn.QuantizedLinear for m in mods):
        return None
    m0 = mods[0]
    for m in mods:
        if (
            "bias" in m
            or m.get("biases") is None
            or m.group_size != m0.group_size
            or m.bits != m0.bits
            or m.mode != m0.mode
        ):
            return None
    out = _ConcatQuantizedLinear.__new__(_ConcatQuantizedLinear)
    nn.Module.__init__(out)
    out.group_size, out.bits, out.mode = m0.group_size, m0.bits, m0.mode
    out.weight = mx.concatenate([m["weight"] for m in mods], axis=0)
    out.scales = mx.concatenate([m["scales"] for m in mods], axis=0)
    out.biases = mx.concatenate([m["biases"] for m in mods], axis=0)
    mx.eval(out.weight, out.scales, out.biases)
    out.freeze()
    return out


def _alias_rows(fused, mods):
    """원본 모듈들의 어레이를 연접 버퍼의 행-슬라이스 뷰로 교체한다.

    연접 직후 원본 버퍼가 해제되므로 순 메모리 증가가 0 이고, 레거시 경로
    (프리필 S>8, 킬스위치 밖 형상)는 뷰 위에서 비트 동일하게 동작한다
    (행-슬라이스는 연속 뷰 — 같은 값·같은 레이아웃·같은 커널).
    """
    o = 0
    for m in mods:
        n = m["weight"].shape[0]
        m.weight = fused["weight"][o : o + n]
        m.scales = fused["scales"][o : o + n]
        m.biases = fused["biases"][o : o + n]
        o += n


class Attention(_Qwen3NextAttention):
    """qwen3_5 전용 확장: §2.1(a) qkv 연접(3→1 GEMM).

    게이트 밖(프리필 L>8, 배치 B>1, 샤딩, 훈련)은 부모 경로 그대로 —
    MTP 블록도 이 클래스를 재사용하므로 융합이 자동 적용된다(§3 오라클 포함).
    (§2.1b 의 compile 에필로그는 실측 no-op 으로 제거 — 모듈 헤더 주석 참조.)
    """

    def __init__(self, args):
        super().__init__(args)
        self._fused_qkv = None
        self._fused_tried = False

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape
        in_window = (
            B == 1
            and L <= 8
            and not self.training
            and (cache is None or getattr(cache, "lengths", None) is None)
        )
        if not (_FUSED_PROJ and in_window):
            return super().__call__(x, mask, cache)

        if self._fused_qkv is None and not self._fused_tried:
            self._fused_tried = True
            mods = [self.q_proj, self.k_proj, self.v_proj]
            self._fused_qkv = _concat_quantized(mods)
            if self._fused_qkv is not None:
                _alias_rows(self._fused_qkv, mods)
        if self._fused_qkv is None:
            return super().__call__(x, mask, cache)
        nq = self.num_attention_heads * self.head_dim * 2
        nkv = self.num_key_value_heads * self.head_dim
        qkv = self._fused_qkv(x)
        q_proj_output = qkv[..., :nq]
        keys = qkv[..., nq : nq + nkv]
        values = qkv[..., nq + nkv :]

        queries, gate = mx.split(
            q_proj_output.reshape(B, L, self.num_attention_heads, -1), 2, axis=-1
        )
        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        keys = self.k_norm(keys.reshape(B, L, self.num_key_value_heads, -1)).transpose(
            0, 2, 1, 3
        )
        values = values.reshape(B, L, self.num_key_value_heads, -1).transpose(
            0, 2, 1, 3
        )

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        gate = gate.reshape(B, L, -1)
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output * mx.sigmoid(gate))


class MLP(_Qwen3NextMLP):
    """qwen3_5 전용 확장: §2.1(a) gate/up 연접(2→1 GEMM). swiglu 는 기존
    shapeless-compile 그대로."""

    def __init__(self, dim, hidden_dim):
        super().__init__(dim, hidden_dim)
        self._hidden_dim = hidden_dim
        self._fused_gateup = None
        self._fused_tried = False

    def __call__(self, x) -> mx.array:
        if not (
            _FUSED_PROJ
            and x.ndim == 3
            and x.shape[0] == 1
            and x.shape[1] <= 8
            and not self.training
        ):
            return super().__call__(x)
        if self._fused_gateup is None and not self._fused_tried:
            self._fused_tried = True
            mods = [self.gate_proj, self.up_proj]
            self._fused_gateup = _concat_quantized(mods)
            if self._fused_gateup is not None:
                _alias_rows(self._fused_gateup, mods)
        if self._fused_gateup is None:
            return super().__call__(x)
        gu = self._fused_gateup(x)
        h = self._hidden_dim
        return self.down_proj(swiglu(gu[..., :h], gu[..., h:]))


@dataclass
class TextModelArgs(BaseModelArgs):
    model_type: str = ""
    hidden_size: int = 4096
    intermediate_size: int = 14336
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    rms_norm_eps: float = 1e-6
    vocab_size: int = 151936
    num_key_value_heads: int = 8
    max_position_embeddings: int = 131072
    linear_num_value_heads: int = 64
    linear_num_key_heads: int = 16
    linear_key_head_dim: int = 192
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    head_dim: Optional[int] = None
    full_attention_interval: int = 4

    # MoE fields (optional, for Qwen3_5MoeForConditionalGeneration)
    num_experts: int = 0
    num_experts_per_tok: int = 0
    decoder_sparse_step: int = 1
    shared_expert_intermediate_size: int = 0
    moe_intermediate_size: int = 0
    norm_topk_prob: bool = True

    # Rope parameters
    rope_parameters: Optional[Dict[str, Union[float, str, bool, List[int]]]] = field(
        default_factory=lambda: {
            "type": "default",
            "mrope_section": [11, 11, 10],
            "rope_theta": 100000,
            "partial_rotary_factor": 0.25,
        }
    )

    # Derived from rope_parameters (set in __post_init__)
    partial_rotary_factor: float = 0.25
    rope_theta: float = 100000.0
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads

        if self.rope_parameters:
            if (
                "type" not in self.rope_parameters
                and "rope_type" in self.rope_parameters
            ):
                self.rope_parameters["type"] = self.rope_parameters.pop("rope_type")

            self.partial_rotary_factor = self.rope_parameters.get(
                "partial_rotary_factor", 0.25
            )
            self.rope_theta = self.rope_parameters.get("rope_theta", 100000.0)
            self.rope_scaling = self.rope_parameters


class GatedDeltaNet(nn.Module):
    def __init__(self, config: TextModelArgs):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        if self.num_v_heads % self.num_k_heads != 0:
            raise ValueError(
                f"num_v_heads ({self.num_v_heads}) must be divisible by num_k_heads ({self.num_k_heads})"
            )

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_norm_epsilon = config.rms_norm_eps

        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=0,
        )

        self.in_proj_qkv = nn.Linear(
            self.hidden_size, self.key_dim * 2 + self.value_dim, bias=False
        )
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)

        self.dt_bias = mx.ones(self.num_v_heads)

        A = mx.random.uniform(low=0, high=16, shape=(self.num_v_heads,))
        self.A_log = mx.log(A)

        self.norm = RMSNormGated(self.head_v_dim, eps=self.layer_norm_epsilon)

        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

        self.sharding_group = None

        # 글루-융합 상태(§2.1) — 밑줄 접두라 파라미터 트리 밖(저장/양자화 불가시).
        self._fused_in_proj = None
        self._fused_tried = False
        self._fold_w = None

        # capture-and-rerun 롤백(DSpark [PA23]①, mlx-dspark 계보 [I65]) —
        # 투기 루프가 True 로 켜면 매 forward 의 스캔 입력과 pre-round 상태를
        # 보관한다. 부분 수락 시 dspark_rerun() 이 수락 접두만으로 recurrence 를
        # 재실행해 상태를 복원한다 — pending-carry 의 재공급 세금이 사라진다.
        # 기본 False: 미사용 시 분기 1회 외 비용 0.
        self._dspark_capture = False
        self._dspark_scan = None

    def __call__(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, S, _ = inputs.shape

        # The recurrent path takes a (B, S) validity mask, as built by
        # create_ssm_mask. Callers that drive decoder layers directly -- AWQ
        # calibration builds one attention mask for the whole model -- can hand
        # us an attention mask ("causal", or a (S, S) array) instead. That mask
        # carries nothing the scan can use, so drop it here. Forwarding it is
        # not merely useless: the fused scan kernel indexes the mask flat, so a
        # (S, S) array is silently consumed as garbage instead of erroring.
        if mask is not None and (
            not isinstance(mask, mx.array) or mask.shape != (B, S)
        ):
            mask = None

        if self.sharding_group is not None:
            inputs = sum_gradients(self.sharding_group)(inputs)

        # §2.1 융합 게이트: 디코드/검증 폭 전용. 프리필(S>8)·배치·샤딩·훈련·
        # ragged(lengths) 는 레거시 경로(§5-R3/R7/R8).
        in_window = (
            B == 1
            and S <= 8
            and self.sharding_group is None
            and not self.training
            and (cache is None or cache.lengths is None)
        )
        fused_ok = _FUSED_PROJ and in_window
        glue_ok = _COMPILE_GLUE and in_window

        if fused_ok and self._fused_in_proj is None and not self._fused_tried:
            self._fused_tried = True
            mods = [self.in_proj_qkv, self.in_proj_z, self.in_proj_b, self.in_proj_a]
            self._fused_in_proj = _concat_quantized(mods)
            if self._fused_in_proj is not None:
                _alias_rows(self._fused_in_proj, mods)
        if fused_ok and self._fused_in_proj is not None:
            # 4 qmm → 1 qmm(N=16480). N=48 GEMV 2개(그리드 미달, 지연-바운드)가
            # 대형 GEMV 패스에 흡수된다. S=1 에서 슬라이스는 전부 연속-오프셋 뷰.
            cd, vd, nh = self.conv_dim, self.value_dim, self.num_v_heads
            proj = self._fused_in_proj(inputs)
            qkv = proj[..., :cd]
            z = proj[..., cd : cd + vd].reshape(B, S, nh, self.head_v_dim)
            b = proj[..., cd + vd : cd + vd + nh]
            a = proj[..., cd + vd + nh :]
        else:
            qkv = self.in_proj_qkv(inputs)
            z = self.in_proj_z(inputs).reshape(B, S, self.num_v_heads, self.head_v_dim)
            b = self.in_proj_b(inputs)
            a = self.in_proj_a(inputs)

        if cache is not None and cache[0] is not None:
            conv_state = cache[0]
        else:
            conv_state = mx.zeros(
                (B, self.conv_kernel_size - 1, self.conv_dim),
                dtype=inputs.dtype,
            )

        if mask is not None:
            qkv = mx.where(mask[..., None], qkv, 0)
        conv_input = mx.concatenate([conv_state, qkv], axis=1)
        if cache is not None:
            n_keep = self.conv_kernel_size - 1
            if cache.lengths is not None:
                ends = mx.clip(cache.lengths, 0, S)
                positions = (ends[:, None] + mx.arange(n_keep))[..., None]
                cache[0] = mx.take_along_axis(conv_input, positions, axis=1)
            else:
                cache[0] = mx.contiguous(conv_input[:, -n_keep:, :])
        conv_out = nn.silu(self.conv1d(conv_input))

        q, k, v = [
            t.reshape(B, S, h, d)
            for t, h, d in zip(
                mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                [self.head_k_dim, self.head_k_dim, self.head_v_dim],
            )
        ]

        state = cache[1] if cache else None
        inv_scale = k.shape[-1] ** -0.5
        if glue_ok:
            # 스칼라곱을 rms_norm 의 weight 벡터로 접는다 — launch −2/층.
            # 비트 동일 실측(오라클 T1): fast.rms_norm 은 정규화값을 출력 dtype 으로
            # 라운딩한 뒤 weight 를 곱하므로 eager 의 이중-라운딩과 일치하고,
            # 스칼라의 bf16 캐스트도 eager 의 weak-promotion 과 같다.
            if self._fold_w is None:
                self._fold_w = (
                    mx.full((self.head_k_dim,), inv_scale**2, dtype=q.dtype),
                    mx.full((self.head_k_dim,), inv_scale, dtype=k.dtype),
                )
                mx.eval(self._fold_w)
            q = mx.fast.rms_norm(q, self._fold_w[0], 1e-6)
            k = mx.fast.rms_norm(k, self._fold_w[1], 1e-6)
        else:
            q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
            k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)

        if self._dspark_capture:
            # 스캔 입력은 conv·norm 통과 후의 값이라 위치별(row-wise) 연산만
            # 남았다 — 접두 슬라이스가 곧 "그 접두만 처리했을 때의 입력"이다.
            # conv_input 은 pre-round conv 상태의 재구성용([:, :K-1] = 구 상태,
            # 이후 행 = 이번 배치의 qkv). state 는 아직 갱신 전 참조(mx.array
            # 불변이라 아래 cache[1] 덮어쓰기와 무관하게 유효).
            self._dspark_scan = (q, k, v, a, b, conv_input, state)

        out, state = gated_delta_update(
            q,
            k,
            v,
            a,
            b,
            self.A_log,
            self.dt_bias,
            state,
            mask,
            # [I233] 융합 커널은 VJP 미구현 — 민감도 추정/미분이 필요한 경로에서는
            # MLX_QWEN35_NO_KERNEL=1 로 미분 가능 경로 강제(K3 _DETACH_KDA 교훈).
            use_kernel=(not self.training
                        and os.environ.get("MLX_QWEN35_NO_KERNEL") != "1"),
        )

        if cache is not None:
            cache[1] = state
            cache.advance(S)

        out = self.norm(out, z)
        out = self.out_proj(out.reshape(B, S, -1))

        if self.sharding_group is not None:
            out = mx.distributed.all_sum(out, group=self.sharding_group)

        return out

    def dspark_rerun(self, cache, n_keep: int) -> None:
        """부분 수락 롤백: 직전 캡처의 수락 접두 `n_keep` 행만으로 recurrence 를
        재실행해 캐시 상태를 복원한다(capture-and-rerun, [I65]).

        pending-carry 대체 경로: 기각 시 상태를 통째로 되감고 수락 토큰을 다음
        검증에 끌고 가는 대신, pre-round 상태에서 접두만 다시 스캔한다.
        비용은 층당 소형 커널 1회(S=n_keep ≤ 8) — 풀모델 재공급 세금이 없다.
        conv 상태는 [구 상태 ‖ 이번 qkv] 연접의 슬라이스로 산술 복원된다.
        """
        q, k, v, a, b, conv_input, state0 = self._dspark_scan
        _, state = gated_delta_update(
            q[:, :n_keep],
            k[:, :n_keep],
            v[:, :n_keep],
            a[:, :n_keep],
            b[:, :n_keep],
            self.A_log,
            self.dt_bias,
            state0,
            None,
            use_kernel=(not self.training
                        and os.environ.get("MLX_QWEN35_NO_KERNEL") != "1"),
        )
        n_ctx = self.conv_kernel_size - 1
        cache[0] = mx.contiguous(conv_input[:, n_keep : n_keep + n_ctx, :])
        cache[1] = state


class DecoderLayer(nn.Module):
    def __init__(self, args: TextModelArgs, layer_idx: int):
        super().__init__()
        self.is_linear = (layer_idx + 1) % args.full_attention_interval != 0
        if self.is_linear:
            self.linear_attn = GatedDeltaNet(args)
        else:
            self.self_attn = Attention(args)

        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

        if args.num_experts > 0:
            self.mlp = SparseMoeBlock(args)
        else:
            self.mlp = MLP(args.hidden_size, args.intermediate_size)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        if self.is_linear:
            r = self.linear_attn(self.input_layernorm(x), mask, cache)
        else:
            r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out


class Qwen3_5TextModel(PipelineMixin, nn.Module):
    def __init__(self, args: TextModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            DecoderLayer(args=args, layer_idx=i) for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.ssm_idx = 0
        self.fa_idx = args.full_attention_interval - 1

    def pipeline(self, group):
        super().pipeline(group)
        self.ssm_idx = None
        self.fa_idx = None
        for e, l in enumerate(self.pipeline_layers):
            if self.ssm_idx is None and l.is_linear:
                self.ssm_idx = e
            elif self.fa_idx is None and not l.is_linear:
                self.fa_idx = e
            if self.ssm_idx is not None and self.fa_idx is not None:
                break

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        input_embeddings: Optional[mx.array] = None,
        tap_layers: Optional[List[int]] = None,
    ) -> mx.array:
        if input_embeddings is not None:
            hidden_states = input_embeddings
        else:
            hidden_states = self.embed_tokens(inputs)

        pipeline_rank = self.pipeline_rank
        pipeline_size = self.pipeline_size

        if cache is None:
            cache = [None] * len(self.pipeline_layers)

        fa_mask = None
        ssm_mask = None
        if self.fa_idx is not None:
            fa_mask = create_attention_mask(hidden_states, cache[self.fa_idx])
        if self.ssm_idx is not None:
            ssm_mask = create_ssm_mask(hidden_states, cache[self.ssm_idx])

        # Receive from the previous process in the pipeline
        if pipeline_rank < pipeline_size - 1:
            hidden_states = mx.distributed.recv_like(hidden_states, (pipeline_rank + 1))

        # A speculative drafter (DSpark/DFlash) is fed intermediate hidden
        # states, not just the final one. Collecting them costs nothing when
        # unused, and the alternative — re-running the target — is the whole
        # cost the drafter exists to avoid.
        taps = {} if tap_layers is None else {i: None for i in tap_layers}

        for e, (layer, c) in enumerate(zip(self.pipeline_layers, cache)):
            mask = ssm_mask if layer.is_linear else fa_mask
            hidden_states = layer(hidden_states, mask=mask, cache=c)
            if e in taps:
                taps[e] = hidden_states

        # Send to the next process in the pipeline
        if pipeline_rank != 0:
            hidden_states = mx.distributed.send(
                hidden_states, (pipeline_rank - 1) % pipeline_size
            )
            if cache[-1] is not None:
                if hasattr(cache[-1], "keys"):
                    cache[-1].keys = mx.depends(cache[-1].keys, hidden_states)
                else:
                    cache[-1][0] = mx.depends(cache[-1][0], hidden_states)

        # Broadcast h while keeping it in the graph
        if pipeline_size > 1:
            hidden_states = mx.distributed.all_gather(hidden_states)[
                : hidden_states.shape[0]
            ]

        hidden_states = self.norm(hidden_states)
        self._h_prenorm = hidden_states        # [I231] MTP 앵커 = final-norm 이후(vLLM 계약)
        # 탭은 별도 속성으로 넘긴다 — 반환 시그니처를 바꾸면 이 모델을 부르는
        # 모든 경로(생성·서버·MTP·투기 루프)가 함께 깨진다.
        self._taps = taps
        return hidden_states


class Qwen3_5MtpBlock(nn.Module):
    """MTP 내부 트랜스포머 1층(풀 어텐션 + MLP) — 키 mtp.layers.0.*"""

    def __init__(self, args: TextModelArgs):
        super().__init__()
        h = args.hidden_size
        self.self_attn = Attention(args)
        self.mlp = MLP(h, args.intermediate_size)
        self.input_layernorm = nn.RMSNorm(h, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(h, eps=args.rms_norm_eps)


class Qwen3_5Mtp(nn.Module):
    """벤더 동봉 MTP(nextn) 헤드 — DeepSeek-V3 계열 배선([I231]).

    h' = fc([pre_fc_norm_hidden(h) ; pre_fc_norm_embedding(emb(t+1))])
       → 트랜스포머 1층(풀 어텐션) → norm → 공유 lm_head
    체크포인트 키: mtp.fc / mtp.pre_fc_norm_{hidden,embedding} / mtp.layers.0.* / mtp.norm
    """

    def __init__(self, args: TextModelArgs):
        super().__init__()
        h = args.hidden_size
        self.pre_fc_norm_hidden = nn.RMSNorm(h, eps=args.rms_norm_eps)
        self.pre_fc_norm_embedding = nn.RMSNorm(h, eps=args.rms_norm_eps)
        self.fc = nn.Linear(2 * h, h, bias=False)
        self.layers = [Qwen3_5MtpBlock(args)]      # 체크포인트 키 mtp.layers.0.*
        self.norm = nn.RMSNorm(h, eps=args.rms_norm_eps)

    @property
    def shared_head(self):
        return self.norm


class TextModel(nn.Module):
    def __init__(self, args: TextModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Qwen3_5TextModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        self.mtp = Qwen3_5Mtp(args) if getattr(args, "with_mtp", True) else None

    @property
    def has_mtp(self):
        return self.mtp is not None and self.mtp.fc.weight.size > 0

    def make_mtp_cache(self):
        return KVCache()

    def mtp_forward(self, h_prev, tokens, cache=None, return_hidden=False):
        """드래프트 로짓: 백본 pre-norm hidden + 커밋 토큰(t+1 페어링).
        체인은 mtp.norm 출력을 pre_normed=True로 직결(재-norm 금지)."""
        emb = self.model.embed_tokens(tokens)
        h_ = h_prev.astype(emb.dtype)
        hn = self.mtp.pre_fc_norm_hidden(h_)   # 체인 스텝에서도 항상 적용
        x = self.mtp.fc(mx.concatenate(
            [self.mtp.pre_fc_norm_embedding(emb), hn], axis=-1))
        mask = None
        if x.shape[1] > 1:
            mask = create_attention_mask(x, cache, return_array=True)
        blk = self.mtp.layers[0]
        h1 = x + blk.self_attn(blk.input_layernorm(x), mask, cache)
        h2 = h1 + blk.mlp(blk.post_attention_layernorm(h1))
        logits = self.lm_head(self.mtp.norm(h2))
        return (logits, h2) if return_hidden else logits

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        input_embeddings: Optional[mx.array] = None,
        tap_layers: Optional[List[int]] = None,
        num_logits: Optional[int] = None,
    ) -> mx.array:
        out = self.model(
            inputs, cache, input_embeddings=input_embeddings, tap_layers=tap_layers
        )
        self._taps = self.model._taps
        # Prefill chunks are consumed for their cache, not their logits, yet the
        # output head runs over every position: 79.4 ms for 512 rows against
        # 4.25 ms for one, on a 248k vocabulary. `num_logits` keeps only the tail
        # the caller will actually read.
        if num_logits is not None and num_logits < out.shape[1]:
            out = out[:, -num_logits:, :]
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    @property
    def layers(self):
        return self.model.pipeline_layers

    def make_cache(self):
        return [ArraysCache(size=2) if l.is_linear else KVCache() for l in self.layers]

    def sanitize(self, weights):
        # [CA79] mtp 키 존재를 raw-HF 신호로 쓰면 **MTP 보존 빌드에서 영구 참**이 되어
        # 변환 때 이미 시프트된 norm에 로드 때 또 +1.0이 붙는다(감마 0.944→1.944→2.944).
        # conv1d 형상만이 raw-HF의 신뢰 가능한 판별자다.
        has_unsanitized_conv1d = any(
            "conv1d.weight" in k and v.shape[-1] != 1 for k, v in weights.items()
        )
        should_shift_norm_weights = has_unsanitized_conv1d
        # [I231] MTP 보존(종전에는 폐기). language_model.mtp.* 로 매핑되어 들어옴.

        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        norm_keys = (
            ".input_layernorm.weight",
            ".post_attention_layernorm.weight",
            "model.norm.weight",
            ".q_norm.weight",
            ".k_norm.weight",
            ".pre_fc_norm_hidden.weight",
            ".pre_fc_norm_embedding.weight",
            "mtp.norm.weight",
        )
        for k, v in weights.items():
            if "conv1d.weight" in k and v.shape[-1] != 1:
                weights[k] = v.moveaxis(2, 1)
            if should_shift_norm_weights and any(k.endswith(sfx) for sfx in norm_keys):
                if v.ndim == 1:
                    weights[k] = v + 1.0
        return weights

    @property
    def quant_predicate(self):
        if self.args.num_experts <= 0:
            return None

        def predicate(path, _):
            if path.endswith("mlp.gate") or path.endswith("shared_expert_gate"):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate

    @property
    def cast_predicate(self):
        def predicate(path: str):
            if path.endswith("A_log"):
                return False
            return True

        return predicate


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    text_config: dict

    @classmethod
    def from_dict(cls, params):
        if "text_config" not in params:
            return cls(model_type=params["model_type"], text_config=params)
        return super().from_dict(params)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.language_model = TextModel(TextModelArgs.from_dict(args.text_config))

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
        tap_layers: Optional[List[int]] = None,
        num_logits: Optional[int] = None,
    ):
        out = self.language_model(
            inputs,
            cache=cache,
            input_embeddings=input_embeddings,
            tap_layers=tap_layers,
            num_logits=num_logits,
        )
        self._taps = self.language_model._taps
        return out

    @property
    def model(self):
        return self.language_model.model

    # ── MTP 자기-투기 계약 위임([I231])
    @property
    def has_mtp(self):
        return self.language_model.has_mtp

    @property
    def mtp(self):
        return self.language_model.mtp

    def make_mtp_cache(self):
        return self.language_model.make_mtp_cache()

    def mtp_forward(self, *a, **kw):
        return self.language_model.mtp_forward(*a, **kw)

    # Towers this text-only port has no modules for, but whose weights must
    # survive conversion (see utils.save_passthrough_weights). Dropping them
    # turns a multimodal checkpoint into a text-only one with no diagnostic;
    # `mtp.` is listed for stock mlx-lm, and is a no-op here because this port
    # implements the MTP head and therefore already saves those tensors.
    passthrough_patterns = ("model.visual.", "vision_tower.", "mtp.")

    def sanitize(self, weights):
        sanitized = {}
        for key, value in weights.items():
            # Vision weights are not consumed by this text-only port; they are
            # dropped here (nothing to load them into) and preserved at save
            # time from the source checkpoint instead.
            if key.startswith("vision_tower") or key.startswith("model.visual"):
                continue
            if key.startswith("model.language_model"):
                key = key.replace("model.language_model", "language_model.model")
            elif key.startswith("language_model."):
                pass
            else:
                key = "language_model." + key
            sanitized[key] = value
        return self.language_model.sanitize(sanitized)

    def shard(self, group=None):
        group = group or mx.distributed.init()
        N = group.size()
        rank = group.rank()

        # A sharding factory for the convolution in gated delta net
        def conv_sharding(key_dim):
            return lambda p, w: (0, [key_dim, 2 * key_dim])

        def repeat_kv_layer_inplace(layer, h):
            # No repeat needed cause we have more heads than nodes
            if N <= h:
                return

            # Repeat function to apply to the layer weights
            def _repeat(p):
                s = p.shape
                p = p.reshape(h, s[0] // h, *s[1:])
                p = mx.repeat(p, N // h, axis=0)
                p = p.reshape(-1, *s[1:])
                return p

            layer.update(tree_map(_repeat, layer.parameters()))

        for layer in self.layers:
            # Linear attention
            if layer.is_linear:
                kd = layer.linear_attn.key_dim
                layer.linear_attn.sharding_group = group
                shard_inplace(layer.linear_attn.conv1d, conv_sharding(kd), group=group)
                layer.linear_attn.conv1d.groups //= N
                shard_inplace(
                    layer.linear_attn.in_proj_qkv,
                    "all-to-sharded",
                    segments=[kd, 2 * kd],
                    group=group,
                )
                shard_inplace(
                    layer.linear_attn.in_proj_z, "all-to-sharded", group=group
                )
                shard_inplace(
                    layer.linear_attn.in_proj_b, "all-to-sharded", group=group
                )
                shard_inplace(
                    layer.linear_attn.in_proj_a, "all-to-sharded", group=group
                )
                layer.linear_attn.dt_bias = mx.contiguous(
                    mx.split(layer.linear_attn.dt_bias, N)[rank]
                )
                layer.linear_attn.A_log = mx.contiguous(
                    mx.split(layer.linear_attn.A_log, N)[rank]
                )
                shard_inplace(layer.linear_attn.out_proj, "sharded-to-all", group=group)
                layer.linear_attn.num_k_heads //= N
                layer.linear_attn.num_v_heads //= N
                layer.linear_attn.key_dim //= N
                layer.linear_attn.value_dim //= N
                layer.linear_attn.conv_dim //= N

            # Softmax attention
            else:
                layer.self_attn.o_proj = shard_linear(
                    layer.self_attn.o_proj, "sharded-to-all", group=group
                )
                layer.self_attn.q_proj = shard_linear(
                    layer.self_attn.q_proj, "all-to-sharded", group=group
                )
                repeat_kv_layer_inplace(
                    layer.self_attn.k_proj, layer.self_attn.num_key_value_heads
                )
                repeat_kv_layer_inplace(
                    layer.self_attn.v_proj, layer.self_attn.num_key_value_heads
                )
                layer.self_attn.k_proj = shard_linear(
                    layer.self_attn.k_proj, "all-to-sharded", group=group
                )
                layer.self_attn.v_proj = shard_linear(
                    layer.self_attn.v_proj, "all-to-sharded", group=group
                )
                layer.self_attn.num_attention_heads //= N
                layer.self_attn.num_key_value_heads = max(
                    1, layer.self_attn.num_key_value_heads // N
                )

            # MLP
            if isinstance(layer.mlp, MLP):
                layer.mlp.gate_proj = shard_linear(
                    layer.mlp.gate_proj, "all-to-sharded", group=group
                )
                layer.mlp.down_proj = shard_linear(
                    layer.mlp.down_proj, "sharded-to-all", group=group
                )
                layer.mlp.up_proj = shard_linear(
                    layer.mlp.up_proj, "all-to-sharded", group=group
                )

            # MoE
            else:
                layer.mlp.sharding_group = group
                shard_inplace(
                    layer.mlp.shared_expert.gate_proj, "all-to-sharded", group=group
                )
                shard_inplace(
                    layer.mlp.shared_expert.down_proj, "sharded-to-all", group=group
                )
                shard_inplace(
                    layer.mlp.shared_expert.up_proj, "all-to-sharded", group=group
                )
                shard_inplace(
                    layer.mlp.switch_mlp.gate_proj, "all-to-sharded", group=group
                )
                shard_inplace(
                    layer.mlp.switch_mlp.down_proj, "sharded-to-all", group=group
                )
                shard_inplace(
                    layer.mlp.switch_mlp.up_proj, "all-to-sharded", group=group
                )

    @property
    def layers(self):
        return self.language_model.model.pipeline_layers

    def make_cache(self):
        return self.language_model.make_cache()

    @property
    def quant_predicate(self):
        return self.language_model.quant_predicate

    @property
    def cast_predicate(self):
        return self.language_model.cast_predicate
