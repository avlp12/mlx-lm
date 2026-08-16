# Copyright © 2026 Apple Inc.

"""Small-M quantized matmul that amortizes the weight read.

`mx.quantized_matmul` grows almost linearly in M for M in 2..8 while the dense
bf16 path is flat — the weight read is not amortized across rows until roughly
M >= 16 (ml-explore/mlx#4265). Everything that verifies several tokens at once
lives in that window: speculative decoding, MTP, small-batch serving.

This kernel puts the multiply on `simdgroup_matrix`. An 8x8 MMA tile covers
M <= 8 exactly, so each quantized weight group is read once and reused by every
row. Dequantization happens per quantization group (64 elements) rather than per
MMA tile (8), which cuts the barrier count eightfold — that single change took
the kernel from 0.29 ms to 0.15 ms at M=8.

Measured on M3 Ultra, (M,5120) @ (17408,5120)^T, 4-bit / group 64:

    M      MLX      ours    ratio
    1   0.042     0.136    0.31x
    4   0.122     0.141    0.86x
    7   0.204     0.149    1.37x
    8   0.232     0.152    1.52x

So MLX wins below M=4 (its GEMV path is at roofline) and loses above it. The
dispatch below reflects that: this is a supplement to `quantized_matmul`, not a
replacement.
"""

import os
from typing import Any

import mlx.core as mx
import mlx.nn as nn

KC = 128           # x staging chunk (4KB) — total threadgroup use 12KB
NPT = 64           # output columns per threadgroup — amortizes the x staging
TGT = 256          # 8 simdgroups
M_MIN = 6          # measured crossover on a *dependent* chain (see below)
# M=4 measures 0.98-1.10x — inside the noise, and it made MTP k=3 slower in the
# model. M=6 is 1.16-1.52x and M=8 is 1.57-1.92x. Latency, not throughput, is
# what decides here: benchmarked as independent calls this kernel looks like a
# win from M=4, because 32 queued copies overlap and hide its longer critical
# path. Decode is a dependent chain and cannot.
M_MAX = 8          # one MMA tile
# Below this the grid is ceil(N/64) threadgroups and the GPU sits idle; the
# model has 96 layers with N=48, which would each get a single threadgroup.
N_MIN = 4096

_SRC = r"""
    const int K = KD, N = ND, M = MD;
    const int KPS = KD / 8;                 // 심드그룹당 K 구간

    uint tid  = thread_position_in_threadgroup.x;
    uint tgid = threadgroup_position_in_grid.x;
    uint sg   = tid >> 5;
    uint lane = tid & 31;

    int n0 = (int)tgid * 8;                 // threadgroup 하나가 출력열 8개

    // x 는 device 에서 직접 MMA 로 읽는다(bf16 입력 + float 누산). 스테이징이 없어져
    // threadgroup 메모리가 10KB 로 내려가고, 무엇보다 임계경로가 짧아진다.
    threadgroup bfloat16_t bs[8 * 512];     // 심드그룹별 64k x 8n, 8KB
    threadgroup float red[8 * 64];          // 심드그룹 간 합산, 2KB

    simdgroup_matrix<float, 8, 8> C = simdgroup_matrix<float, 8, 8>(0);
    threadgroup bfloat16_t* bt = bs + sg * 512;

    // ── split-K: 8개 심드그룹이 K 를 8등분해 각자 짧은 직렬 루프를 돈다.
    //    (종전 판본은 한 threadgroup 이 K 전체를 순회해 배리어 160쌍이 임계경로에 놓였다)
    int kbeg = (int)sg * KPS;
    for (int kk = 0; kk < KPS; kk += 64) {
        int ka = kbeg + kk;
        int j  = (int)(lane & 7);
        int kq = (int)(lane >> 3);
        int n  = n0 + j;
        if (n < N) {
            int g = ka >> 6;
            float s  = (float)sc[(size_t)n * (K / 64) + g];
            float bb = (float)bi[(size_t)n * (K / 64) + g];
            const device uint* wr = w + (size_t)n * (K / 8) + (ka >> 3) + kq * 2;
            uint p0 = wr[0], p1 = wr[1];
            for (int t = 0; t < 8; ++t)
                bt[(kq * 16 + t) * 8 + j] = (bfloat16_t)((float)((p0 >> (4 * t)) & 15u) * s + bb);
            for (int t = 0; t < 8; ++t)
                bt[(kq * 16 + 8 + t) * 8 + j] = (bfloat16_t)((float)((p1 >> (4 * t)) & 15u) * s + bb);
        } else {
            for (int t = 0; t < 16; ++t) bt[(kq * 16 + t) * 8 + j] = (bfloat16_t)0;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_matrix<bfloat16_t, 8, 8> A, B;
        for (int kt = 0; kt < 8; ++kt) {
            simdgroup_load(A, x + ka + kt * 8, K);   // x[0:8, ka+8kt ..] — 패딩된 8행
            simdgroup_load(B, bt + kt * 64, 8);
            simdgroup_multiply_accumulate(C, A, B, C);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    simdgroup_store(C, red + sg * 64, 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 8개 부분합을 더해 쓴다.
    for (int i = (int)tid; i < 64; i += 256) {
        int m = i >> 3, j = i & 7;
        int n = n0 + j;
        if (m < M && n < N) {
            float v = 0.0f;
            for (int q = 0; q < 8; ++q) v += red[q * 64 + i];
            out[(size_t)m * N + n] = (bfloat16_t)v;
        }
    }
"""

_KERNEL = mx.fast.metal_kernel(
    name="qmm_mma4",
    input_names=["x", "w", "sc", "bi"],
    output_names=["out"],
    source=_SRC,
)


def _eligible(x: mx.array, group_size: int, bits: int, K: int, N: int) -> bool:
    return (
        bits == 4
        and group_size == 64
        and K % KC == 0
        and N >= N_MIN
        and x.dtype == mx.bfloat16
        and mx.default_device() == mx.gpu
    )


def fast_qmm(x, w, scales, biases, *, group_size: int, bits: int):
    """Drop-in for `mx.quantized_matmul(..., transpose=True)` in the small-M window.

    Falls back whenever the shape or dtype is outside what the kernel handles, so
    callers never have to check.
    """
    K = x.shape[-1]
    M = 1
    for d in x.shape[:-1]:
        M *= d
    N = w.shape[0]
    if not (M_MIN <= M <= M_MAX and _eligible(x, group_size, bits, K, N)):
        return mx.quantized_matmul(
            x, w, scales, biases, transpose=True, group_size=group_size, bits=bits
        )
    flat = x.reshape(M, K)
    if M < 8:  # 커널이 8행 MMA 타일을 device 에서 직접 읽는다 — 경계 밖을 막는다
        flat = mx.concatenate([flat, mx.zeros((8 - M, K), dtype=flat.dtype)], axis=0)
    (out,) = _KERNEL(
        inputs=[flat, w, scales, biases],
        template=[("KD", K), ("ND", N), ("MD", M)],
        output_shapes=[(M, N)],
        output_dtypes=[mx.bfloat16],
        grid=(((N + 7) // 8) * TGT, 1, 1),
        threadgroup=(TGT, 1, 1),
    )
    return out.reshape(*x.shape[:-1], N)


_ORIGINAL_CALL = None


def enable(model: Any = None) -> None:
    """Route `nn.QuantizedLinear` through `fast_qmm`.

    Patching the class rather than each instance keeps this reversible and keeps
    quantized layers created later (draft models, adapters) on the same path.
    Set `MLXLM_NO_FAST_QMM=1` to opt out without touching call sites.
    """
    global _ORIGINAL_CALL
    if _ORIGINAL_CALL is not None or os.environ.get("MLXLM_NO_FAST_QMM") == "1":
        return
    _ORIGINAL_CALL = nn.QuantizedLinear.__call__

    def __call__(self, x):
        # 커널은 affine(scales+biases) 전용 — nvfp4/mxfp4 등 다른 모드의 층은
        # biases 가 없어 KeyError 로 죽는다(community 빌드 KL 측정에서 실증).
        if "biases" not in self or getattr(self, "mode", "affine") != "affine":
            return _ORIGINAL_CALL(self, x)
        y = fast_qmm(
            x,
            self["weight"],
            self["scales"],
            self["biases"],
            group_size=self.group_size,
            bits=self.bits,
        )
        if "bias" in self:
            y = y + self["bias"]
        return y

    nn.QuantizedLinear.__call__ = __call__


def disable() -> None:
    global _ORIGINAL_CALL
    if _ORIGINAL_CALL is not None:
        nn.QuantizedLinear.__call__ = _ORIGINAL_CALL
        _ORIGINAL_CALL = None
