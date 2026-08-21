# Copyright © 2026 Apple Inc.
"""Serving glue for two-box prefill: incremental (prompt-cache aware) prefill
for ``mlx_lm.server --prefill-2box HOST:PORT``.

Differences from the benchmark orchestrator (``TwoBoxPrefill.prefill``):

  * ``prefill_into`` advances an *existing* 64-entry prompt cache in place
    (entries at offset ``start``) instead of always starting from zero, so the
    server's LRU prompt-cache reuse (multi-turn incremental prefill) works.
  * The remote runner keeps recent layer-[0,split) caches resident between
    requests (session store, ``prefill2`` op).  The orchestrator sends the
    *full* token prefix each time; the runner resumes from its longest stored
    exact-prefix ≤ ``send_from`` and recomputes only its own delta, streaming
    back only activations the local box actually needs (positions ≥
    ``send_from``).
  * Connection management: fail-fast probe at server startup, lazy reconnect
    per request afterwards.
  * The chunk schedule is chosen per request from the length of the un-cached
    suffix (``chunk_for``).  A wider chunk amortises fixed per-chunk cost and
    is what an accelerator offload needs to engage at all, but it deepens the
    pipeline bubble; which one wins depends on how many chunks the prompt has.
    See ``--prefill-2box-chunk-long`` / ``--prefill-2box-long-tokens``.

Single-turn requests (no reusable prefix) reduce exactly to the verified
benchmark path: same chunk schedule, same slice code, bitwise-identical.
"""

import socket
import threading
import time
import queue

import mlx.core as mx

from . import wire
from .orchestrator import TwoBoxError, TwoBoxPrefill, _cache_arrays
from ..models.cache import ArraysCache, KVCache


def parse_hostport(s):
    host, sep, port = s.rpartition(":")
    if not sep or not host:
        raise ValueError(f"--prefill-2box expects HOST:PORT, got {s!r}")
    return host, int(port)


def probe_runner(host, port, timeout=5):
    """Startup fail-fast probe: connect + hello + basic slice sanity.

    Returns the hello ack dict.  Raises TwoBoxError with a clear message if
    the runner is unreachable or misconfigured.  The probe disconnects
    afterwards (the runner serves one connection at a time).
    """
    try:
        sock = socket.create_connection((host, port), timeout=timeout)
    except OSError as e:
        raise TwoBoxError(
            f"prefill-2box runner unreachable at {host}:{port} ({e}). "
            "Start it on the remote box: "
            "python -m mlx_lm.prefill_2box.server --model <same model> "
            f"--port {port}"
        ) from None
    try:
        sock.settimeout(timeout)
        wire.send_json(sock, {"op": "hello"})
        kind, ack = wire.recv_msg(sock)
        if kind != "json" or ack.get("op") != "hello_ack":
            raise TwoBoxError(f"prefill-2box runner bad hello ack: {ack}")
        if ack.get("mlx") != mx.__version__:
            raise TwoBoxError(
                f"prefill-2box mlx mismatch: local {mx.__version__} vs "
                f"runner {ack.get('mlx')} — bitwise parity not guaranteed; "
                "align both boxes before serving"
            )
        try:
            wire.send_json(sock, {"op": "quit"})
        except OSError:
            pass
        return ack
    finally:
        sock.close()


class ServingPrefill:
    """Persistent two-box prefill engine for one loaded server model."""

    def __init__(
        self,
        model,
        host,
        port,
        split=32,
        chunk=1024,
        min_tokens=4096,
        chunk_long=None,
        long_tokens=11264,
    ):
        self.model = model
        self.host = host
        self.port = port
        self.split = split
        self.chunk = chunk
        self.min_tokens = min_tokens
        # Optional second chunk schedule for long prompts.  ``chunk_long`` is
        # what turns the branch on; ``long_tokens`` defaults to the measured
        # crossover (see ``chunk_for``).  Asking for the branch without a
        # threshold is a configuration error rather than a silent
        # single-schedule fallback.
        if chunk_long is not None and long_tokens is None:
            raise TwoBoxError("chunk_long needs a long_tokens threshold")
        self.chunk_long = chunk_long
        self.long_tokens = long_tokens
        self._tb = None
        self.ensure()  # fail fast at load time

    # ------------------------------------------------------------- connection
    def ensure(self):
        """(Re)establish the runner connection; raises TwoBoxError if down."""
        if self._tb is not None:
            return self._tb
        try:
            tb = TwoBoxPrefill(self.model, self.host, port=self.port, split=self.split)
        except OSError as e:
            raise TwoBoxError(
                f"prefill-2box runner unreachable at {self.host}:{self.port} ({e})"
            ) from None
        try:
            tb.warmup()
        except Exception:
            tb.close()
            raise
        self._tb = tb
        return tb

    def _drop(self):
        if self._tb is not None:
            try:
                self._tb.sock.close()
            except OSError:
                pass
            self._tb = None

    def close(self):
        if self._tb is not None:
            try:
                self._tb.close()
            except Exception:
                pass
            self._tb = None

    # ---------------------------------------------------------------- gating
    def applicable(self, n_new_tokens):
        """Use two-box only when the un-cached suffix is long enough to win."""
        return n_new_tokens - 1 >= self.min_tokens

    def chunk_for(self, n_new_tokens):
        """Chunk schedule for a prefill of ``n_new_tokens`` new tokens.

        The branch exists because the two schedules are not ordered: the wider
        one is faster only once the prompt carries enough chunks to amortise
        the deeper pipeline bubble.  Below ``long_tokens`` the narrow schedule
        wins outright, so the choice is made per request rather than fixed at
        startup.

        The 11264 default is the measured crossover for 1024 -> 2048 on a 27B
        hybrid-attention model split across two M3 Ultras with an ANE/CPU
        offload attached (2048/1024 throughput ratio 0.95 at 8K, 1.00 at 9-10K,
        1.03 at 11K, 1.11 at 32K).  It is a property of *that* pairing, not a
        universal one: with the offload off the wide schedule never won
        anywhere in 8K-32K, which is why ``chunk_long`` stays opt-in.  Re-measure
        before trusting the default on a different model, split or accelerator.
        """
        if self.chunk_long is not None and n_new_tokens >= self.long_tokens:
            return self.chunk_long
        return self.chunk

    # --------------------------------------------------------------- prefill
    def prefill_into(self, full_cache, tokens, start, progress=None):
        """Advance ``full_cache`` in place from offset ``start`` to len(tokens).

        ``tokens``: the full prompt token list *minus the final token* (the
        caller steps the last token itself, per the generate_step convention).
        ``full_cache``: per-layer cache list for the whole model whose entries
        are at offset ``start`` (0 for a fresh cache).  Layers [0, split) are
        replaced with the runner's state at the end; layers [split, n) are
        advanced locally chunk by chunk as boundary activations arrive.

        Returns a stats dict.  On any failure the connection is dropped (next
        request reconnects) and TwoBoxError propagates: no silent fallback.
        """
        tb = self.ensure()
        tokens = [int(t) for t in tokens]
        P = len(tokens)
        need = P - start
        if need < 1:
            raise TwoBoxError(f"nothing to prefill (P={P}, start={start})")
        chunk = self.chunk_for(need)
        n_layers = tb.n_layers
        if len(full_cache) != n_layers:
            raise TwoBoxError(
                f"cache has {len(full_cache)} entries, model has {n_layers}"
            )

        # Adopt the local half of the supplied cache and sanity-check offsets.
        local = tb.local
        local.cache = full_cache[self.split :]
        for i, c in enumerate(local.cache):
            if isinstance(c, KVCache) and c.offset != start:
                raise TwoBoxError(
                    f"layer {self.split + i} cache offset {c.offset} != start {start}"
                )

        try:
            return self._prefill_into(
                tb, full_cache, tokens, P, start, chunk, progress
            )
        except Exception:
            # Connection state is unknown mid-protocol: drop and reconnect on
            # the next request.  The supplied cache may be partially advanced
            # and must be discarded by the caller.
            self._drop()
            raise
        finally:
            local.cache = []

    def _prefill_into(self, tb, full_cache, tokens, P, start, chunk, progress):
        local = tb.local
        sock = tb.sock

        acts_q = queue.Queue(maxsize=8)
        store = {}
        ctrl = {}
        rx_err = []

        def rx():
            try:
                while True:
                    m = wire.recv_msg(sock)
                    if m[0] == "json":
                        msg = m[1]
                        op = msg.get("op")
                        if op == "error":
                            raise TwoBoxError(f"runner error: {msg['err']}")
                        ctrl[op] = msg
                        if op == "cache_done":
                            return
                    else:
                        meta, raw = m[1], m[2]
                        if meta["n"] == "act":
                            acts_q.put((meta, raw))
                        else:
                            store[meta["n"]] = (meta, raw)
            except Exception as e:
                rx_err.append(e)
                acts_q.put(None)

        t0 = time.perf_counter()
        wire.send_json(
            sock,
            {
                "op": "prefill2",
                "tokens": tokens,
                "chunk": chunk,
                "send_from": start,
            },
        )
        rx_thread = threading.Thread(target=rx, daemon=True)
        rx_thread.start()

        waits = []
        t_local = []
        consumed = start
        while consumed < P:
            tw = time.perf_counter()
            item = acts_q.get()
            if item is None:
                raise rx_err[0] if rx_err else TwoBoxError("rx died")
            waits.append(time.perf_counter() - tw)
            meta, raw = item
            n = meta["s"][1]
            if meta.get("p") != consumed or meta["s"][0] != 1 or meta["s"][2] != tb.hidden:
                raise TwoBoxError(
                    f"act mismatch: expected pos {consumed}, got {meta}"
                )
            tc = time.perf_counter()
            h = wire.to_mx(meta, raw)
            h = local.forward(h)
            mx.eval(h, *local.state_arrays())
            t_local.append(time.perf_counter() - tc)
            consumed += n
            if progress is not None:
                progress(consumed - start, P - start)
        t_pipeline = time.perf_counter() - t0

        # Pull the remote half's cache and install it into full_cache[0:split].
        t1 = time.perf_counter()
        wire.send_json(sock, {"op": "fetch_cache"})
        rx_thread.join()
        if rx_err:
            raise rx_err[0]
        if "cache_done" not in ctrl:
            raise TwoBoxError("no cache_done from runner")
        srv = ctrl.get("prefill_done", {})
        if srv.get("offset") != P:
            raise TwoBoxError(f"runner offset {srv.get('offset')} != {P}")

        kv_bytes = 0
        layers = self.model.model.layers
        for gi in range(self.split):
            entry = full_cache[gi]
            if layers[gi].is_linear:
                if not isinstance(entry, ArraysCache):
                    raise TwoBoxError(f"layer {gi}: expected ArraysCache")
                for slot, name in ((0, f"conv:{gi}"), (1, f"ssm:{gi}")):
                    if name not in store:
                        raise TwoBoxError(f"runner cache missing {name}")
                    meta, raw = store[name]
                    kv_bytes += len(raw)
                    entry[slot] = wire.to_mx(meta, raw)
            else:
                if not isinstance(entry, KVCache):
                    raise TwoBoxError(f"layer {gi}: expected KVCache")
                parts = []
                for name in (f"k:{gi}", f"v:{gi}"):
                    if name not in store:
                        raise TwoBoxError(f"runner cache missing {name}")
                    meta, raw = store[name]
                    kv_bytes += len(raw)
                    parts.append(wire.to_mx(meta, raw))
                entry.state = tuple(parts)
                if entry.offset != P:
                    raise TwoBoxError(
                        f"layer {gi} remote KV offset {entry.offset} != {P}"
                    )
        mx.eval(*(a for c in full_cache[: self.split] for a in _cache_arrays(c)))
        t_cache = time.perf_counter() - t1

        return {
            "n_tokens": P,
            "start": start,
            "chunk": chunk,
            "resumed_at": srv.get("e0"),
            "t_pipeline": t_pipeline,
            "t_cache_install": t_cache,
            "t_act_waits": waits,
            "t_local_chunks": t_local,
            "server_t_compute": srv.get("t_compute"),
            "cache_wire_bytes": kv_bytes,
            "cache_wire_s": ctrl["cache_done"].get("t"),
        }
