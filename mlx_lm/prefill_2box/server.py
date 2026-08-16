# Copyright © 2026 Apple Inc.
"""Two-box prefill server: runs layers [lo, hi) (default the bottom half incl.
embedding) and streams per-chunk boundary activations to the orchestrator.

    python -m mlx_lm.prefill_2box.server --model /path/to/model --port 39919

Single-threaded compute; a sender thread drains a queue so the socket write of
chunk i overlaps the compute of chunk i+1.  Serves one connection at a time and
returns to accept() when the client disconnects, so a crashed client never
requires a relaunch.  SIGTERM exits cleanly (no-KILL discipline: this process
should never need SIGKILL).
"""

import argparse
import os
import queue
import signal
import socket
import sys
import threading
import time
import traceback

import numpy as np

import mlx.core as mx

from . import wire
from ..models.cache import ArraysCache
from .runner import (
    LayerSlice,
    iter_cache_tensors,
    load_model_only,
    set_wired_limit,
)


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


class SessionStore:
    """Recent layer-[lo,hi) caches kept resident between connections.

    Entries are (tokens, cache_list) with the cache state at offset
    len(tokens).  Lookup returns the longest stored entry whose token list is
    an exact prefix of the request and no longer than ``max_resume`` (the
    local box cannot consume activations before its own cached offset).  A
    matched entry is *popped*: resuming mutates the cache in place, so it is
    re-stored under its new key after the prefill completes.
    """

    def __init__(self, max_sessions=4):
        self.max_sessions = max_sessions
        self._entries = []  # ordered oldest -> newest

    def pop_best(self, tokens, max_resume):
        best_i = -1
        best_len = 0
        for i, (toks, _) in enumerate(self._entries):
            n = len(toks)
            if n == 0 or n > max_resume or n > len(tokens) or n <= best_len:
                continue
            if toks == tokens[:n]:
                best_i, best_len = i, n
        if best_i < 0:
            return None, 0
        toks, cache = self._entries.pop(best_i)
        return cache, len(toks)

    def put(self, tokens, cache):
        # Replace any entry that is a prefix of the new one (superseded).
        self._entries = [
            (t, c)
            for (t, c) in self._entries
            if not (len(t) <= len(tokens) and t == tokens[: len(t)])
        ]
        self._entries.append((tokens, cache))
        while len(self._entries) > self.max_sessions:
            self._entries.pop(0)

    def nbytes(self):
        total = 0
        for _, cache in self._entries:
            for c in cache:
                if isinstance(c, ArraysCache):
                    total += sum(x.nbytes for x in c.cache if x is not None)
                elif c.keys is not None:
                    total += c.keys.nbytes + c.values.nbytes
        return total


def _do_prefill2(sock, sl, sessions, msg):
    """Incremental prefill: resume from a resident session cache when its
    stored tokens are an exact prefix of the request, and stream back only
    activations for positions >= send_from (what the local box still needs)."""
    tokens = [int(t) for t in msg["tokens"]]
    chunk = int(msg.get("chunk", 1024))
    send_from = int(msg["send_from"])
    P = len(tokens)
    if not (0 <= send_from < P):
        raise ValueError(f"bad send_from {send_from} for {P} tokens")

    cache, e0 = sessions.pop_best(tokens, send_from)
    if cache is not None:
        sl.cache = cache
    else:
        sl.reset()
        e0 = 0
    mx.clear_cache()

    schedule = []
    n = P - e0
    while n > 0:
        schedule.append(min(chunk, n))
        n -= schedule[-1]

    sendq = queue.Queue(maxsize=8)
    send_err = []

    def sender():
        try:
            while True:
                item = sendq.get()
                if item is None:
                    return
                name, arr, meta = item
                wire.send_tensor(sock, name, arr, **meta)
        except Exception as e:
            send_err.append(e)

    st = threading.Thread(target=sender, daemon=True)
    st.start()

    toks = mx.array(np.asarray(tokens, dtype=np.int32))[None]
    pos = e0
    t_chunks = []
    n_sent = 0
    t0 = time.perf_counter()
    try:
        for n in schedule:
            tc0 = time.perf_counter()
            h = sl.forward(toks[:, pos : pos + n], is_tokens=(sl.embed is not None))
            # Anything handed to the sender thread must be fully evaluated
            # here: lazy graphs must not be evaluated from another thread
            # (thread-local stream trap -> hard abort).
            s = max(pos, send_from)
            piece = None
            if s < pos + n:
                piece = h if s == pos else mx.contiguous(h[:, s - pos :, :])
            extra = [piece] if piece is not None else []
            mx.eval(h, *sl.state_arrays(), *extra)
            t_chunks.append(time.perf_counter() - tc0)
            if send_err:
                raise send_err[0]
            if piece is not None:
                sendq.put(("act", piece, {"p": s}))
                n_sent += 1
            pos += n
    finally:
        sendq.put(None)  # always unblock the sender thread
        st.join()
    if send_err:
        raise send_err[0]
    sessions.put(tokens, sl.cache)
    wire.send_json(
        sock,
        {
            "op": "prefill_done",
            "t_compute": time.perf_counter() - t0,
            "t_chunks": t_chunks,
            "offset": sl.offset(),
            "e0": e0,
            "n_sent": n_sent,
        },
    )
    log(
        f"prefill2: P={P} resume e0={e0} send_from={send_from} "
        f"computed {P - e0} sent {P - send_from} in {time.perf_counter() - t0:.3f}s "
        f"(sessions {len(sessions._entries)}, {sessions.nbytes() / 1e9:.2f} GB)"
    )


def _do_prefill(sock, sl, msg):
    tokens = msg["tokens"]
    schedule = msg["schedule"]
    stream_kv = bool(msg.get("stream_kv", False))
    if sum(schedule) != len(tokens):
        raise ValueError(f"schedule {sum(schedule)} != tokens {len(tokens)}")
    sl.reset()
    mx.clear_cache()

    sendq = queue.Queue(maxsize=8)
    send_err = []

    def sender():
        try:
            while True:
                item = sendq.get()
                if item is None:
                    return
                name, arr, meta = item
                wire.send_tensor(sock, name, arr, **meta)
        except Exception as e:  # surfaced in the main loop
            send_err.append(e)

    st = threading.Thread(target=sender, daemon=True)
    st.start()

    toks = mx.array(np.asarray(tokens, dtype=np.int32))[None]
    pos = 0
    t_chunks = []
    t0 = time.perf_counter()
    for ci, n in enumerate(schedule):
        tc0 = time.perf_counter()
        h = sl.forward(toks[:, pos : pos + n], is_tokens=(sl.embed is not None))
        if stream_kv:
            slabs = []
            for i, (layer, c) in enumerate(zip(sl.layers, sl.cache)):
                if not layer.is_linear:
                    gi = sl.lo + i
                    slabs.append(
                        (f"k:{gi}", mx.contiguous(c.keys[..., pos : pos + n, :]))
                    )
                    slabs.append(
                        (f"v:{gi}", mx.contiguous(c.values[..., pos : pos + n, :]))
                    )
            mx.eval(h, *sl.state_arrays(), *(s for _, s in slabs))
        else:
            mx.eval(h, *sl.state_arrays())
        t_chunks.append(time.perf_counter() - tc0)
        if send_err:
            raise send_err[0]
        sendq.put(("act", h, {"c": ci}))
        if stream_kv:
            for name, s in slabs:
                sendq.put((name, s, {"c": ci}))
        pos += n
    sendq.put(None)
    st.join()
    if send_err:
        raise send_err[0]
    wire.send_json(
        sock,
        {
            "op": "prefill_done",
            "t_compute": time.perf_counter() - t0,
            "t_chunks": t_chunks,
            "offset": sl.offset(),
        },
    )


def _handle_conn(sock, sl, sessions):
    """Returns True if the whole server should shut down."""
    wire.tune(sock)
    sock.settimeout(1800)
    while True:
        try:
            kind, *rest = wire.recv_msg(sock)
        except wire.WireError:
            log("client disconnected")
            return False
        if kind != "json":
            log("protocol error: unexpected tensor frame from client")
            return False
        msg = rest[0]
        op = msg.get("op")
        try:
            if op == "hello":
                wire.send_json(
                    sock,
                    {
                        "op": "hello_ack",
                        "mlx": mx.__version__,
                        "lo": sl.lo,
                        "hi": sl.hi,
                        "n_layers": sl.n_layers,
                        "hidden": sl.hidden_size,
                        "pid": os.getpid(),
                        "model": getattr(sl, "model_path", None),
                    },
                )
            elif op == "warmup":
                t0 = time.perf_counter()
                n = int(msg.get("n", 64))
                if sl.embed is not None:
                    x = mx.random.randint(0, 100000, (1, n))
                    h = sl.forward(x, is_tokens=True)
                else:
                    x = mx.random.normal((1, n, sl.hidden_size)).astype(mx.bfloat16)
                    h = sl.forward(x)
                mx.eval(h, *sl.state_arrays())
                sl.reset()
                mx.clear_cache()
                wire.send_json(
                    sock, {"op": "warmup_done", "t": time.perf_counter() - t0}
                )
            elif op == "prefill":
                log(
                    f"prefill: {len(msg['tokens'])} tokens, "
                    f"schedule {msg['schedule']}, stream_kv={msg.get('stream_kv', False)}"
                )
                _do_prefill(sock, sl, msg)
            elif op == "prefill2":
                _do_prefill2(sock, sl, sessions, msg)
            elif op == "fetch_cache":
                t0 = time.perf_counter()
                nbytes = 0
                for name, arr in iter_cache_tensors(
                    sl, include_kv=not msg.get("skip_kv", False)
                ):
                    nbytes += wire.send_tensor(sock, name, arr)
                wire.send_json(
                    sock,
                    {
                        "op": "cache_done",
                        "t": time.perf_counter() - t0,
                        "bytes": nbytes,
                        "offset": sl.offset(),
                    },
                )
                log(f"cache sent: {nbytes / 1e6:.1f} MB in {time.perf_counter()-t0:.3f}s")
            elif op == "reset":
                sl.reset()
                mx.clear_cache()
                wire.send_json(sock, {"op": "reset_done"})
            elif op == "quit":
                log("client quit")
                return False
            elif op == "shutdown":
                wire.send_json(sock, {"op": "bye"})
                return True
            else:
                wire.send_json(sock, {"op": "error", "err": f"unknown op {op!r}"})
        except Exception:
            tb = traceback.format_exc()
            log("op failed:\n" + tb)
            try:
                wire.send_json(sock, {"op": "error", "err": tb})
            except Exception:
                return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=39919)
    ap.add_argument("--lo", type=int, default=0)
    ap.add_argument("--hi", type=int, default=32)
    ap.add_argument(
        "--sessions",
        type=int,
        default=4,
        help="Resident layer-slice caches kept across connections for "
        "incremental (multi-turn) prefill via the prefill2 op.",
    )
    args = ap.parse_args()

    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    log(f"loading model {args.model} (layers [{args.lo}, {args.hi})) ...")
    t0 = time.perf_counter()
    model = load_model_only(args.model)
    lim = set_wired_limit()
    sl = LayerSlice(model, args.lo, args.hi)
    sl.model_path = args.model
    sessions = SessionStore(max_sessions=args.sessions)
    log(
        f"model loaded in {time.perf_counter()-t0:.1f}s, "
        f"wired limit {lim/2**30:.0f} GiB, mlx {mx.__version__}, "
        f"sessions {args.sessions}"
    )

    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port))
    srv.listen(1)
    log(f"listening on {args.host}:{args.port}")
    while True:
        conn, addr = srv.accept()
        log(f"client {addr}")
        try:
            shutdown = _handle_conn(conn, sl, sessions)
        finally:
            conn.close()
        sl.reset()
        mx.clear_cache()
        if shutdown:
            log("shutdown requested")
            return


if __name__ == "__main__":
    main()
