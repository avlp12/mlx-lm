# Copyright © 2026 Apple Inc.
"""Minimal framed TCP wire for the two-box prefill split.

Frames: 13-byte header ``<IBQ`` (magic, type, payload length) + payload.
  T_JSON:   payload is a UTF-8 JSON object (control messages).
  T_TENSOR: payload is ``<I json_len | json meta | raw bytes``.  Meta carries
            ``n`` (name), ``d`` (dtype tag), ``s`` (shape) plus free extras.

bf16/f16 tensors travel as their exact uint16 bytes (bitwise round trip,
verified); fp32 as-is.  numpy is only a byte-mover here — no math.
"""

import json
import socket
import struct

import numpy as np

import mlx.core as mx

MAGIC = 0x2B0C5EED
_HDR = struct.Struct("<IBQ")
T_JSON = 1
T_TENSOR = 2


class WireError(RuntimeError):
    pass


def tune(sock):
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    for opt in (socket.SO_SNDBUF, socket.SO_RCVBUF):
        try:
            sock.setsockopt(socket.SOL_SOCKET, opt, 16 << 20)
        except OSError:
            pass


def recv_exact(sock, n):
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        r = sock.recv_into(view[got:n], n - got)
        if r == 0:
            raise WireError("peer closed")
        got += r
    return buf


def send_json(sock, obj):
    b = json.dumps(obj).encode()
    sock.sendall(_HDR.pack(MAGIC, T_JSON, len(b)))
    sock.sendall(b)


_TAG_OF_MX = {
    mx.bfloat16: "bf16",
    mx.float16: "f16",
    mx.float32: "f32",
    mx.uint16: "u16",
    mx.uint32: "u32",
    mx.int32: "i32",
}
_NP_OF_TAG = {
    "bf16": np.uint16,
    "f16": np.uint16,
    "f32": np.float32,
    "u16": np.uint16,
    "u32": np.uint32,
    "i32": np.int32,
}


def _npbytes(a):
    """mx.array -> (contiguous numpy array of exact bytes, dtype tag)."""
    tag = _TAG_OF_MX.get(a.dtype)
    if tag is None:
        raise WireError(f"unsupported dtype {a.dtype}")
    if a.dtype in (mx.bfloat16, mx.float16):
        a = a.view(mx.uint16)
    try:
        n = np.array(a, copy=False)
    except Exception:
        n = np.array(a)
    if not n.flags["C_CONTIGUOUS"]:
        n = np.ascontiguousarray(n)
    return n, tag


def send_tensor(sock, name, a, **meta):
    """Send one tensor frame; returns raw byte count (excl. framing)."""
    n, tag = _npbytes(a)
    hdr = {"n": name, "d": tag, "s": list(a.shape)}
    hdr.update(meta)
    j = json.dumps(hdr).encode()
    sock.sendall(_HDR.pack(MAGIC, T_TENSOR, 4 + len(j) + n.nbytes))
    sock.sendall(struct.pack("<I", len(j)))
    sock.sendall(j)
    sock.sendall(n)
    return n.nbytes


def recv_msg(sock):
    """-> ("json", obj) or ("tensor", meta, memoryview_of_raw_bytes)."""
    hdr = recv_exact(sock, _HDR.size)
    magic, ftype, plen = _HDR.unpack(bytes(hdr))
    if magic != MAGIC:
        raise WireError(f"bad magic {magic:#x}")
    payload = recv_exact(sock, plen)
    if ftype == T_JSON:
        return ("json", json.loads(bytes(payload).decode()))
    if ftype == T_TENSOR:
        (jlen,) = struct.unpack_from("<I", payload, 0)
        meta = json.loads(bytes(payload[4 : 4 + jlen]).decode())
        return ("tensor", meta, memoryview(payload)[4 + jlen :])
    raise WireError(f"bad frame type {ftype}")


def to_mx(meta, raw):
    """Tensor frame -> mx.array with the original dtype/shape (exact bytes)."""
    arr = np.frombuffer(raw, dtype=_NP_OF_TAG[meta["d"]]).reshape(meta["s"])
    out = mx.array(arr)
    if meta["d"] == "bf16":
        out = out.view(mx.bfloat16)
    elif meta["d"] == "f16":
        out = out.view(mx.float16)
    return out
