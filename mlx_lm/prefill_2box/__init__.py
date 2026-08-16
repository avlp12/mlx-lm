# Copyright © 2026 Apple Inc.
"""Opt-in two-box (TB5) layer-split prefill for qwen3_5-class hybrid models.

Design (fleet two-box lane, 2026-08-16): layer-major sequence split.  A remote
box (epsilon) runs layers [0, split) including the embedding; the local box
(gesicht) runs layers [split, n) plus final norm / lm_head, and owns decode.
The prompt is cut into chunks; while the local box computes chunk i through its
half, the remote box computes chunk i+1 through the other half.  Boundary
activations ([1, T, hidden] bf16, ~21 MB at T=2048) move over a raw TCP socket
on the TB5 link (4.35 GB/s measured), which hides entirely under per-chunk
compute.  After the last chunk the remote half's cache (GDN conv+ssm states,
KV slabs) is pulled over and installed so decode continues single-box.

Nothing here is imported by any existing mlx_lm path; use explicitly:

    from mlx_lm.prefill_2box import TwoBoxPrefill

Server side (remote box):

    python -m mlx_lm.prefill_2box.server --model ~/qwen38/q4v --port 39919
"""

from .orchestrator import TwoBoxPrefill
from .runner import LayerSlice, make_schedule

__all__ = ["TwoBoxPrefill", "LayerSlice", "make_schedule"]
