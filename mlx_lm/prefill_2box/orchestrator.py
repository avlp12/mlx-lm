# Copyright © 2026 Apple Inc.
"""Two-box prefill orchestrator (local box: layers [split, n) + decode).

Usage:

    model, tokenizer = mlx_lm.load(path)
    tb = TwoBoxPrefill(model, host="10.0.0.2", port=39919, split=32)
    tb.warmup()
    cache, stats = tb.prefill(tokens[:-1])          # tokens: list[int]
    logits = model(mx.array([tokens[-1:]]), cache=cache, num_logits=1)
    ... decode continues on this box with `cache` ...

``prefill`` processes exactly the tokens it is given (callers follow the
generate_step convention: prefill tokens[:-1], then step the last token for the
first-token logits).  A receiver thread drains activation frames while the GPU
computes, so the remote box, the link, and the local GPU overlap.
"""

import queue
import socket
import threading
import time

import mlx.core as mx

from . import wire
from .runner import LayerSlice, build_full_cache, make_schedule


class TwoBoxError(RuntimeError):
    pass


class TwoBoxPrefill:
    def __init__(self, model, host, port=39919, split=32, connect_timeout=15):
        self.model = model
        core = model.model
        self.split = split
        self.n_layers = len(core.layers)
        self.local = LayerSlice(model, split, self.n_layers)
        self.hidden = self.local.hidden_size
        self.sock = socket.create_connection((host, port), timeout=connect_timeout)
        wire.tune(self.sock)
        self.sock.settimeout(1800)
        wire.send_json(self.sock, {"op": "hello"})
        kind, ack = wire.recv_msg(self.sock)
        if kind != "json" or ack.get("op") != "hello_ack":
            raise TwoBoxError(f"bad hello ack: {ack}")
        if ack["lo"] != 0 or ack["hi"] != split or ack["n_layers"] != self.n_layers:
            raise TwoBoxError(
                f"server slice [{ack['lo']},{ack['hi']}) of {ack['n_layers']} "
                f"!= expected [0,{split}) of {self.n_layers}"
            )
        self.remote_meta = ack
        if ack["mlx"] != mx.__version__:
            print(
                f"[prefill_2box] WARNING: mlx mismatch local {mx.__version__} "
                f"remote {ack['mlx']} — bitwise parity not guaranteed",
                flush=True,
            )

    # ------------------------------------------------------------------ utils
    def _ctrl(self, expect):
        while True:
            kind, *rest = wire.recv_msg(self.sock)
            if kind != "json":
                raise TwoBoxError(f"unexpected tensor frame while waiting {expect}")
            msg = rest[0]
            if msg.get("op") == "error":
                raise TwoBoxError(f"server error: {msg['err']}")
            if msg.get("op") == expect:
                return msg

    def warmup(self, n=64):
        wire.send_json(self.sock, {"op": "warmup", "n": n})
        x = mx.random.normal((1, n, self.hidden)).astype(mx.bfloat16)
        h = self.local.forward(x)
        mx.eval(self.local.logits_last(h), *self.local.state_arrays())
        self.local.reset()
        mx.clear_cache()
        return self._ctrl("warmup_done")

    def close(self):
        try:
            wire.send_json(self.sock, {"op": "quit"})
        except Exception:
            pass
        self.sock.close()

    def shutdown_server(self):
        wire.send_json(self.sock, {"op": "shutdown"})
        self._ctrl("bye")
        self.sock.close()

    # ---------------------------------------------------------------- prefill
    def prefill(self, tokens, chunk=2048, schedule=None, stream_kv=False):
        """Run two-box prefill over `tokens`; returns (full_cache, stats).

        full_cache is a 64-entry cache list ready for single-box decode on the
        local model.  stats carries the timing/accounting breakdown.
        """
        tokens = [int(t) for t in tokens]
        n_pre = len(tokens)
        schedule = schedule or make_schedule(n_pre, chunk)
        if sum(schedule) != n_pre:
            raise TwoBoxError(f"schedule sums {sum(schedule)} != {n_pre}")

        self.local.reset()
        mx.clear_cache()

        acts_q = queue.Queue(maxsize=8)
        store = {}  # name -> (meta, raw) or list[(chunk_idx, meta, raw)]
        ctrl = {}
        rx_err = []
        done_evt = threading.Event()

        def rx():
            try:
                while not done_evt.is_set():
                    m = wire.recv_msg(self.sock)
                    if m[0] == "json":
                        msg = m[1]
                        op = msg.get("op")
                        if op == "error":
                            raise TwoBoxError(f"server error: {msg['err']}")
                        ctrl[op] = msg
                        if op == "cache_done":
                            return
                    else:
                        meta, raw = m[1], m[2]
                        name = meta["n"]
                        if name == "act":
                            acts_q.put((meta, raw))
                        elif name.startswith(("k:", "v:")) and "c" in meta:
                            store.setdefault(name, []).append((meta["c"], meta, raw))
                        else:
                            store[name] = (meta, raw)
            except Exception as e:
                rx_err.append(e)
                acts_q.put(None)  # unblock main thread

        rx_thread = threading.Thread(target=rx, daemon=True)

        t0 = time.perf_counter()
        wire.send_json(
            self.sock,
            {
                "op": "prefill",
                "tokens": tokens,
                "schedule": schedule,
                "stream_kv": stream_kv,
            },
        )
        rx_thread.start()

        waits = []  # per-chunk act wait (pipeline bubble accounting)
        t_local = []  # per-chunk local compute
        pos = 0
        for ci, n in enumerate(schedule):
            tw = time.perf_counter()
            item = acts_q.get()
            if item is None:
                raise rx_err[0] if rx_err else TwoBoxError("rx died")
            waits.append(time.perf_counter() - tw)
            meta, raw = item
            if meta.get("c") != ci or list(meta["s"]) != [1, n, self.hidden]:
                raise TwoBoxError(f"act mismatch: chunk {ci} n {n} got {meta}")
            tc = time.perf_counter()
            h = wire.to_mx(meta, raw)
            h = self.local.forward(h)
            mx.eval(h, *self.local.state_arrays())
            t_local.append(time.perf_counter() - tc)
            pos += n
        t_pipeline = time.perf_counter() - t0

        # ---- pull the remote half's cache and install it
        t1 = time.perf_counter()
        wire.send_json(self.sock, {"op": "fetch_cache", "skip_kv": stream_kv})
        rx_thread.join()
        if rx_err:
            raise rx_err[0]
        if "cache_done" not in ctrl:
            raise TwoBoxError("no cache_done")
        remote = {}
        kv_bytes = 0
        for name, val in store.items():
            if isinstance(val, list):  # streamed KV slabs, ordered by chunk
                val.sort(key=lambda x: x[0])
                parts = [wire.to_mx(m, r) for _, m, r in val]
                kv_bytes += sum(len(r) for _, _, r in val)
                remote[name] = (
                    parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=2)
                )
            else:
                meta, raw = val
                kv_bytes += len(raw)
                remote[name] = wire.to_mx(meta, raw)
        full_cache = build_full_cache(self.model, self.local, remote)
        mx.eval(*(a for c in full_cache[: self.split] for a in _cache_arrays(c)))
        t_cache = time.perf_counter() - t1

        # ---- invariants
        from ..models.cache import KVCache

        for i, c in enumerate(full_cache):
            if isinstance(c, KVCache) and c.offset != n_pre:
                raise TwoBoxError(f"layer {i} offset {c.offset} != {n_pre}")
        srv = ctrl.get("prefill_done", {})
        if srv.get("offset") != n_pre:
            raise TwoBoxError(f"server offset {srv.get('offset')} != {n_pre}")

        stats = {
            "n_tokens": n_pre,
            "schedule": schedule,
            "stream_kv": stream_kv,
            "t_pipeline": t_pipeline,
            "t_cache_install": t_cache,
            "t_local_chunks": t_local,
            "t_act_waits": waits,
            "server_t_chunks": srv.get("t_chunks"),
            "server_t_compute": srv.get("t_compute"),
            "cache_wire_bytes": kv_bytes,
            "cache_wire_s": ctrl["cache_done"].get("t"),
        }
        return full_cache, stats


def _cache_arrays(c):
    from ..models.cache import ArraysCache

    if isinstance(c, ArraysCache):
        return [x for x in c.cache if x is not None]
    if c.keys is not None:
        return [c.keys, c.values]
    return []
