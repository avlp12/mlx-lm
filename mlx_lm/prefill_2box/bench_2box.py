# Copyright © 2026 Apple Inc.
"""Correctness + benchmark harness for the two-box prefill split.

    python -m mlx_lm.prefill_2box.bench_2box --model ~/qwen38/q4v \
        --host 10.0.0.2 --mode verify
    python -m mlx_lm.prefill_2box.bench_2box --model ~/qwen38/q4v \
        --host 10.0.0.2 --mode bench --out /Users/gesicht/qwen38/bench_2box

Measurement discipline (fleet [PA20]): interleaved arm order, cooldown gaps,
length asserts, end-to-end prefill only (no isolated microbenches).  Results go
to a permanent path, never /tmp.
"""

import argparse
import json
import os
import time

import numpy as np

import mlx.core as mx

from ..models.cache import ArraysCache, KVCache, make_prompt_cache
from .orchestrator import TwoBoxPrefill
from .runner import LayerSlice, load_model_only, make_schedule, set_wired_limit

PC = time.perf_counter


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def rng_tokens(n, seed=1234, hi=200000):
    r = np.random.default_rng(seed)
    return r.integers(0, hi, size=n, dtype=np.int64).tolist()


def _cache_arrays(c):
    if isinstance(c, ArraysCache):
        return [x for x in c.cache if x is not None]
    if getattr(c, "keys", None) is not None:
        return [c.keys, c.values]
    return []


def _np_bytes(a):
    if a.dtype in (mx.bfloat16, mx.float16):
        a = a.view(mx.uint16)
    return np.array(mx.contiguous(a))


def snapshot_cache(cache, lo, hi):
    """np snapshots (exact bytes) of cache entries [lo, hi) for comparison."""
    out = {}
    for i in range(lo, hi):
        c = cache[i]
        if isinstance(c, ArraysCache):
            out[f"conv:{i}"] = _np_bytes(c.cache[0])
            out[f"ssm:{i}"] = _np_bytes(c.cache[1])
        else:
            k, v = c.state
            out[f"k:{i}"] = _np_bytes(k)
            out[f"v:{i}"] = _np_bytes(v)
    return out


def compare_snapshots(a, b):
    """-> (n_equal, n_total, worst) where worst=(name, frac_mismatch)."""
    assert a.keys() == b.keys(), (sorted(a)[:4], sorted(b)[:4])
    n_eq, worst = 0, ("", 0.0)
    for k in a:
        assert a[k].shape == b[k].shape, (k, a[k].shape, b[k].shape)
        if np.array_equal(a[k], b[k]):
            n_eq += 1
        else:
            frac = float(np.mean(a[k] != b[k]))
            if frac > worst[1]:
                worst = (k, frac)
    return n_eq, len(a), worst


def single_box_prefill(model, tokens, chunk):
    """Canonical chunked prefill (generate_step semantics, num_logits=1)."""
    cache = make_prompt_cache(model)
    toks = mx.array(np.asarray(tokens, dtype=np.int32))[None]
    schedule = make_schedule(len(tokens), chunk)
    pos, t_chunks = 0, []
    t0 = PC()
    for n in schedule:
        tc = PC()
        model(toks[:, pos : pos + n], cache=cache, num_logits=1)
        mx.eval(*(a for c in cache for a in _cache_arrays(c)))
        t_chunks.append(PC() - tc)
        pos += n
    t = PC() - t0
    assert pos == len(tokens)
    return cache, t, t_chunks


def step_last(model, cache, tok):
    t0 = PC()
    logits = model(mx.array([[tok]], dtype=mx.int32), cache=cache, num_logits=1)
    logits = logits.astype(mx.float32)
    mx.eval(logits)
    return logits, PC() - t0


def greedy_decode(model, cache, logits, n):
    seq = [int(mx.argmax(logits[0, -1]).item())]
    t0 = PC()
    for _ in range(n - 1):
        logits = model(
            mx.array([[seq[-1]]], dtype=mx.int32), cache=cache, num_logits=1
        )
        seq.append(int(mx.argmax(logits[0, -1]).item()))
    return seq, (n - 1) / (PC() - t0)


def cooldown(sec, why=""):
    log(f"cooldown {sec}s {why}")
    time.sleep(sec)


# ---------------------------------------------------------------------- verify
def run_verify(model, client, n_total=2049, chunk=1024, n_decode=48):
    split = client.split
    tokens = rng_tokens(n_total)
    pre = tokens[:-1]
    report = {"n_total": n_total, "chunk": chunk, "split": split}

    log(f"verify: N={n_total} chunk={chunk} split={split}")

    # Reference: single-box chunked, same schedule
    cache_ref, t_ref, _ = single_box_prefill(model, pre, chunk)
    snap_ref = snapshot_cache(cache_ref, 0, len(cache_ref))
    logits_ref, _ = step_last(model, cache_ref, tokens[-1])
    seq_ref, _ = greedy_decode(model, cache_ref, logits_ref, n_decode)
    del cache_ref
    mx.clear_cache()

    # Single-shot (one chunk) reference for fp context
    cache_ss, _, _ = single_box_prefill(model, pre, len(pre))
    logits_ss, _ = step_last(model, cache_ss, tokens[-1])
    del cache_ss
    mx.clear_cache()

    # Two-box, same schedule
    cache_2b, stats = client.prefill(pre, chunk=chunk)
    snap_2b = snapshot_cache(cache_2b, 0, len(cache_2b))
    logits_2b, _ = step_last(model, cache_2b, tokens[-1])
    seq_2b, _ = greedy_decode(model, cache_2b, logits_2b, n_decode)
    del cache_2b
    mx.clear_cache()

    # Cache equality (all 64 layers: remote-installed half + local half)
    n_eq, n_tot, worst = compare_snapshots(snap_ref, snap_2b)
    report["cache_equal"] = f"{n_eq}/{n_tot}"
    report["cache_worst_mismatch"] = worst

    a = np.array(logits_ref)
    b = np.array(logits_2b)
    report["logits_maxabs_2box_vs_chunked"] = float(np.abs(a - b).max())
    report["logits_bitwise_2box_vs_chunked"] = bool(np.array_equal(a, b))
    c = np.array(logits_ss)
    report["logits_maxabs_chunked_vs_singleshot"] = float(np.abs(a - c).max())
    report["greedy_match"] = seq_ref == seq_2b
    report["greedy_first_mismatch"] = next(
        (i for i, (x, y) in enumerate(zip(seq_ref, seq_2b)) if x != y), None
    )
    report["seq_ref_head"] = seq_ref[:8]
    report["seq_2box_head"] = seq_2b[:8]
    report["t_ref_prefill"] = t_ref
    report["t_2box_pipeline"] = stats["t_pipeline"]
    report["stats"] = {
        k: v for k, v in stats.items() if k not in ("t_local_chunks", "t_act_waits")
    }

    ok = (
        report["greedy_match"]
        and n_eq == n_tot
        and report["logits_bitwise_2box_vs_chunked"]
    )
    report["strict_pass"] = ok
    report["pass"] = report["greedy_match"] and (
        report["logits_maxabs_2box_vs_chunked"]
        <= max(1e-3, 2 * report["logits_maxabs_chunked_vs_singleshot"])
    )
    return report


# ----------------------------------------------------------------------- bench
def bench_one_1box(model, tokens, chunk):
    cache, t_prefill, t_chunks = single_box_prefill(model, tokens[:-1], chunk)
    logits, t_step = step_last(model, cache, tokens[-1])
    seq, dec_tps = greedy_decode(model, cache, logits, 17)
    del cache
    mx.clear_cache()
    return {
        "arm": "1box",
        "chunk": chunk,
        "t_prefill": t_prefill,
        "t_chunks": t_chunks,
        "t_step": t_step,
        "ttft": t_prefill + t_step,
        "decode_tps": dec_tps,
        "tok_s": (len(tokens) - 1) / t_prefill,
        "first_tok": seq[0],
    }


def bench_one_2box(model, client, tokens, chunk, stream_kv=False):
    cache, stats = client.prefill(tokens[:-1], chunk=chunk, stream_kv=stream_kv)
    logits, t_step = step_last(model, cache, tokens[-1])
    seq, dec_tps = greedy_decode(model, cache, logits, 17)
    del cache
    mx.clear_cache()
    t_prefill = stats["t_pipeline"]
    return {
        "arm": "2box" + ("+skv" if stream_kv else ""),
        "chunk": chunk,
        "t_prefill": t_prefill,
        "t_cache_install": stats["t_cache_install"],
        "t_step": t_step,
        "ttft": t_prefill + stats["t_cache_install"] + t_step,
        "decode_tps": dec_tps,
        "tok_s": (len(tokens) - 1) / t_prefill,
        "first_tok": seq[0],
        "server_t_chunks": stats["server_t_chunks"],
        "t_local_chunks": stats["t_local_chunks"],
        "t_act_waits": stats["t_act_waits"],
        "cache_wire_bytes": stats["cache_wire_bytes"],
        "cache_wire_s": stats["cache_wire_s"],
    }


def run_bench(model, client, out_dir, ns, reps):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "results_2box.json")
    R = {"meta": {
        "mlx": mx.__version__,
        "remote": client.remote_meta,
        "date": time.strftime("%F %T"),
    }}

    def save():
        with open(out_path, "w") as f:
            json.dump(R, f, indent=1)

    plans = {
        2048: {"sweep_2box": [512, 1024], "sweep_1box": [512, 1024, 2048], "cd": 15},
        8192: {"sweep_2box": [1024, 1536, 2048], "sweep_1box": [1024, 1536, 2048], "cd": 40},
        32768: {"sweep_2box": [2048], "sweep_1box": [2048], "cd": 75},
    }

    for n in ns:
        plan = plans[n]
        cd = plan["cd"]
        tokens = rng_tokens(n)
        cell = {"runs": []}
        R[str(n)] = cell

        # phase 1: chunk sweeps (one run each, interleaved 1box/2box)
        for c1, c2 in zip_longest_cycle(plan["sweep_1box"], plan["sweep_2box"]):
            if c1 is not None:
                r = bench_one_1box(model, tokens, c1)
                log(f"N={n} 1box@{c1}: {r['t_prefill']:.3f}s {r['tok_s']:.1f} tok/s")
                cell["runs"].append(r)
                save()
                cooldown(cd)
            if c2 is not None:
                r = bench_one_2box(model, client, tokens, c2)
                log(
                    f"N={n} 2box@{c2}: {r['t_prefill']:.3f}s {r['tok_s']:.1f} tok/s "
                    f"(cache {r['t_cache_install']:.3f}s ttft {r['ttft']:.3f}s)"
                )
                cell["runs"].append(r)
                save()
                cooldown(cd)

        # pick best 2box chunk by t_prefill
        two = [r for r in cell["runs"] if r["arm"] == "2box"]
        best_chunk = min(two, key=lambda r: r["t_prefill"])["chunk"]
        cell["best_2box_chunk"] = best_chunk

        # phase 2: interleaved reps 1box@2048 vs 2box@best
        for rep in range(reps):
            r = bench_one_1box(model, tokens, 2048)
            cell["runs"].append(r)
            log(f"N={n} rep{rep} 1box@2048: {r['t_prefill']:.3f}s {r['tok_s']:.1f}")
            save()
            cooldown(cd)
            r = bench_one_2box(model, client, tokens, best_chunk)
            cell["runs"].append(r)
            log(f"N={n} rep{rep} 2box@{best_chunk}: {r['t_prefill']:.3f}s {r['tok_s']:.1f}")
            save()
            cooldown(cd)

        # phase 3: one streamed-KV run at best chunk (TTFT variant)
        r = bench_one_2box(model, client, tokens, best_chunk, stream_kv=True)
        cell["runs"].append(r)
        log(
            f"N={n} 2box+skv@{best_chunk}: {r['t_prefill']:.3f}s "
            f"(cache {r['t_cache_install']:.3f}s ttft {r['ttft']:.3f}s)"
        )
        save()
        if n != ns[-1]:
            cooldown(cd, "before next N")

    # summary
    for n in ns:
        cell = R[str(n)]
        best = {}
        for r in cell["runs"]:
            key = (r["arm"], r["chunk"])
            if key not in best or r["t_prefill"] < best[key]["t_prefill"]:
                best[key] = r
        one = min(
            (r for r in cell["runs"] if r["arm"] == "1box" and r["chunk"] == 2048),
            key=lambda r: r["t_prefill"],
        )
        two = min(
            (r for r in cell["runs"] if r["arm"] == "2box"),
            key=lambda r: r["t_prefill"],
        )
        cell["summary"] = {
            "1box_2048_best_s": one["t_prefill"],
            "1box_2048_tok_s": one["tok_s"],
            "2box_best_s": two["t_prefill"],
            "2box_best_chunk": two["chunk"],
            "2box_tok_s": two["tok_s"],
            "speedup_prefill": one["t_prefill"] / two["t_prefill"],
            "ttft_1box": one["ttft"],
            "ttft_2box": two["ttft"],
            "speedup_ttft": one["ttft"] / two["ttft"],
        }
        log(f"N={n} summary: {json.dumps(cell['summary'], indent=1)}")
    save()
    log(f"results -> {out_path}")
    return R


def zip_longest_cycle(a, b):
    m = max(len(a), len(b))
    for i in range(m):
        yield (a[i] if i < len(a) else None, b[i] if i < len(b) else None)


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, default=39919)
    ap.add_argument("--split", type=int, default=32)
    ap.add_argument("--mode", choices=["verify", "bench", "smoke"], default="verify")
    ap.add_argument("--out", default="/Users/gesicht/qwen38/bench_2box")
    ap.add_argument("--ns", default="2048,8192,32768")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--verify-n", type=int, default=2049)
    ap.add_argument("--verify-chunk", type=int, default=1024)
    args = ap.parse_args()

    log(f"loading model {args.model} ...")
    t0 = PC()
    model = load_model_only(args.model)
    lim = set_wired_limit()
    log(f"loaded in {PC()-t0:.1f}s, wired {lim/2**30:.0f} GiB, mlx {mx.__version__}")

    client = TwoBoxPrefill(model, args.host, args.port, split=args.split)
    log(f"connected: {client.remote_meta}")
    log("warmup ...")
    client.warmup(64)
    client.warmup(64)

    if args.mode == "smoke":
        tokens = rng_tokens(257)
        cache, stats = client.prefill(tokens[:-1], chunk=64)
        logits, t = step_last(model, cache, tokens[-1])
        seq, tps = greedy_decode(model, cache, logits, 8)
        log(f"smoke OK: pipeline {stats['t_pipeline']:.3f}s seq {seq} decode {tps:.1f} t/s")
    elif args.mode == "verify":
        rep = run_verify(model, client, args.verify_n, args.verify_chunk)
        print(json.dumps(rep, indent=1, default=str))
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "verify_2box.json"), "w") as f:
            json.dump(rep, f, indent=1, default=str)
        if not rep["pass"]:
            raise SystemExit("VERIFY FAILED")
        log("VERIFY PASS" + (" (strict bitwise)" if rep["strict_pass"] else " (fp tolerance)"))
    else:
        ns = [int(x) for x in args.ns.split(",")]
        run_bench(model, client, args.out, ns, args.reps)

    client.close()


if __name__ == "__main__":
    main()
