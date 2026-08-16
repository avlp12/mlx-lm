# Copyright © 2026 Apple Inc.

"""DSpark block-diffusion speculative decoding.

One draft forward yields a whole block of candidate tokens rather than `k`
sequential forwards, so the drafting overhead is a single 1.36B pass regardless
of block width. On a latency-bound machine that is the point: widening the block
is nearly free, while widening an autoregressive drafter is not.

The awkward part on a hybrid target is the rewind. Its linear-attention layers
carry a recurrent state with no per-token structure to cut, so a partially
accepted block cannot be trimmed back — the state has to be restored wholesale
and the accepted tokens replayed. Done naively that is a second full target pass
per step, which costs more than the speculation wins (measured: 0.4x plain).

So the replay is not performed. Accepted-but-unreplayed tokens are carried in
`pending` and prepended to the *next* step's verification, which the target has
to run anyway. That is one target pass per step at any acceptance rate.
"""

from typing import Any, Callable, Generator, List, Optional, Tuple

import mlx.core as mx

from .models.cache import make_prompt_cache
from .sample_utils import apply_top_k, apply_top_p


def _snapshot(cache: List[Any]) -> List[Any]:
    """Capture enough state to rewind `cache` after a rejected speculation.

    Trimming only what is trimmable fails *silently* here: the KV layers rewind,
    the recurrent layers keep the rejected tokens, and generation degenerates
    into repetition several tokens later, far from the cause.
    """
    snap = []
    for c in cache:
        if hasattr(c, "cache"):        # ArraysCache: recurrent/conv state
            snap.append(("a", list(c.cache), getattr(c, "lengths", None)))
        else:                          # KVCache: an offset is enough
            snap.append(("k", c.offset))
    return snap


def _rewind(cache: List[Any], snap: List[Any]) -> None:
    for c, st in zip(cache, snap):
        if st[0] == "a":
            c.cache = list(st[1])
            if st[2] is not None:
                c.lengths = st[2]
        else:
            c.trim(c.offset - st[1])


def dspark_generate_step(
    prompt: mx.array,
    model: Any,
    draft: Any,
    *,
    max_tokens: int = 256,
    sampler: Optional[Callable[[mx.array], mx.array]] = None,
    temp: float = 0.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    draft_temp: Optional[float] = None,
    use_heads: bool = True,
    conf_tau: float = 0.25,
    block_size: Optional[int] = None,
    max_pending: Optional[int] = None,
    prompt_cache: Optional[List[Any]] = None,
    prefill_step_size: int = 2048,
    # Defaults are the measured operating point, not the permissive setting.
    # max_width: the verification batch must stay inside the split-K kernel's
    #   M <= 8 window. Left unbounded the width drifted to a mean of 9.5 and a
    #   max of 16 — straight into the expensive region the kernel exists to
    #   avoid — and that alone cost 43% of throughput.
    # pad_lm: round the lm_head batch up into the same window. Worth +5% with
    #   the kernel on, and a ~2% loss with it off, so the two travel together.
    # use_conf: the confidence head gates the block down when it doubts a slot.
    #   That trade only pays when a wider block costs more, and with the kernel
    #   the cost is flat to M=8 — measured a loss at every threshold.
    max_width: Optional[int] = 8,
    pad_lm: bool = True,
    use_conf: bool = False,
    defer_sync: bool = True,
    stats: Optional[dict] = None,
    prof: Optional[dict] = None,
) -> Generator[Tuple[mx.array, int], None, None]:
    """Yield `(token, n_accepted)` pairs.

    `n_accepted` is the acceptance length of the step that produced the token,
    so a caller can report acceptance without instrumenting this loop.
    """
    tap_layers = draft.target_layer_ids
    # One slot past the width the drafter was trained at, capped by the kernel
    # window. Measured: block 8 = 71.0 tok/s vs block 7 = 63.6 on the same
    # prompts. Block 9 accepts *more* (3.55 vs 3.46) and still loses, because
    # the wider draft costs more than the extra acceptance returns.
    block_size = block_size or min(draft.block_size + 1, max_width or 8)
    n_spec = block_size - 1
    mask_id = draft.mask_token_id
    sampler = sampler or (lambda lg: mx.argmax(lg, axis=-1))
    # temp > 0 switches acceptance from "the argmax matched" to Leviathan
    # rejection sampling: accept x with prob min(1, p(x)/q(x)), and on rejection
    # resample from (p-q)+. That preserves the temperature-T target distribution
    # exactly while accepting tokens the equality rule throws away — which is the
    # setting the drafter's own acceptance figures were measured in.
    #
    # top_k / top_p extend the guarantee to truncated sampling: BOTH the target
    # and the draft logits go through the client's truncation chain — the exact
    # make_sampler order (top-p mask, then top-k mask, on temp-1 logprobs;
    # temperature last, renormalized) — giving p', q'. Acceptance, residual and
    # bonus all use p'/q'. The load-bearing detail is that drafts are sampled
    # from q' and accepted with that same q': truncating one side only, or
    # accepting with the pre-truncation q, silently breaks the distribution.
    # draft_temp (default: temp) reshapes q' only — rejection sampling
    # preserves p' for ANY proposal, so the draft temperature is a free knob
    # that trades acceptance, never the output law. All three default to None,
    # which is byte-identical to the pre-truncation code path.
    rejection = temp > 0.0
    truncated = top_k is not None or top_p is not None or draft_temp is not None

    def _trunc_logsoftmax(lg, t):
        z = lg.astype(mx.float32)
        z = z - mx.logsumexp(z, axis=-1, keepdims=True)
        if top_p is not None and 0.0 < top_p < 1.0:
            z = apply_top_p(z, top_p)
        if top_k is not None and top_k > 0:
            z = apply_top_k(z, top_k)
        z = z * (1 / max(t, 1e-6))
        return z - mx.logsumexp(z, axis=-1, keepdims=True)

    def _p_logsoftmax(lg):
        if truncated:
            return _trunc_logsoftmax(lg, temp)
        z = (lg / temp).astype(mx.float32)
        return z - mx.logsumexp(z, axis=-1, keepdims=True)

    def _q_logsoftmax(lg):
        if truncated:
            return _trunc_logsoftmax(
                lg, draft_temp if draft_temp is not None else temp
            )
        z = (lg / temp).astype(mx.float32)
        return z - mx.logsumexp(z, axis=-1, keepdims=True)
    # Carrying accepted tokens forward is free only while the run is short; a
    # long rejection streak would otherwise grow every verification without bound.
    max_pending = max_pending or 2 * block_size

    # DSpark's two heads over the DFlash backbone. The bundled reference loop is
    # DFlash's and never touches them, so a faithful port of it runs a 1.23B
    # drafter and leaves 127M of trained bigram correction on the floor.
    markov = getattr(draft, "markov_head", None) if use_heads else None
    conf = getattr(draft, "confidence_head", None) if (use_heads and use_conf) else None

    embed = model.model.embed_tokens
    lm_head = model.language_model.lm_head
    tcache = prompt_cache if prompt_cache is not None else make_prompt_cache(model)
    dcache = draft.make_cache()

    def tapped(taps):
        return mx.concatenate([taps[i] for i in tap_layers], axis=-1)

    # ── Prefill. The prompt's tapped states are the drafter's first context.
    y, chunks = prompt, []
    while y.size > prefill_step_size:
        # num_logits=1: the chunk is here for its cache and its taps, not its
        # logits. See _supports_num_logits in generate.py.
        model(
            y[:prefill_step_size][None], cache=tcache,
            tap_layers=tap_layers, num_logits=1,
        )
        chunks.append(model._taps)
        mx.eval([c.state for c in tcache])
        y = y[prefill_step_size:]
    # num_logits=1: only the last position's logits are sampled below, and taps
    # are collected before lm_head, so slicing the head input is loss-free. The
    # full-width lm_head GEMM here (up to prefill_step_size x 248k vocab) was
    # the one real full-logits materialization left in any prefill path.
    logits = model(y[None], cache=tcache, tap_layers=tap_layers, num_logits=1)
    chunks.append(model._taps)
    # Chunked prefill splits the taps too; keeping only the last chunk would hand
    # the drafter a context that starts mid-prompt.
    taps_cat = mx.concatenate(
        [mx.concatenate([c[i] for c in chunks], axis=1) for i in tap_layers], axis=-1
    )
    taps_base = 0

    if rejection and truncated:
        # In truncated-rejection mode every committed token is drawn internally
        # from p'; the first one included, so the caller needs no sampler.
        first = int(mx.random.categorical(_p_logsoftmax(logits[0, -1, :])).item())
    else:
        first = int(sampler(logits[:, -1, :]).item())
    mx.eval(taps_cat)
    yield mx.array([first]), 1

    n = 1
    base = int(prompt.size)   # the target cache covers [0, base)
    pending = [first]         # committed; positions [base, base + len(pending))
    dend = 0                  # the draft cache holds context for [0, dend)

    import time as _time

    _mark = [0.0]

    def _P(name, *arrs):
        """Region timer. Only active when `prof` is passed; inserts a barrier."""
        if prof is None:
            return
        if arrs:
            mx.eval(*arrs)
        t = _time.perf_counter()
        prof.setdefault(name, []).append(t - _mark[0])
        _mark[0] = t

    while n < max_tokens:
        _t0 = _time.perf_counter()
        _mark[0] = _t0
        L = len(pending)
        # Clamp the verification width so `S = L + n_sub` never leaves the
        # split-K kernel's M <= max_width window.
        n_avail = n_spec if max_width is None else max(0, min(n_spec, max_width - L))
        p_last = base + L - 1          # position the draft block starts at

        if n_avail == 0:
            # No room under the clamp: this step is a plain (unspeculated) pass.
            drafted, qlp, n_sub = [], None, 0
        else:
            # ── Draft. Context is every committed position the draft cache has
            # not seen yet; the block itself starts at the last committed token.
            ctx = taps_cat[:, dend - taps_base : p_last - taps_base, :]
            block = mx.concatenate(
                [mx.array([[pending[-1]]]),
                 mx.full((1, n_spec), mask_id, dtype=mx.int32)],
                axis=1,
            )
            h = draft(embed(block), ctx, k_offset=dend, q_offset=p_last, cache=dcache)
            _P("draft", h)
            # OPEN QUESTION. The reference slices `[:, -block_size+1:]` — see the
            # original loop for the full note; the shifted read is kept because
            # acceptance is scored against the target.
            if pad_lm:
                # Round the row count up into the split-K kernel's M in [6, 8]
                # window (and never past it); the extra rows are discarded.
                _w = min(max(n_avail, 6), h.shape[1], 8)
                base_logits = lm_head(h[:, :_w, :])[0][:n_avail]
            else:
                base_logits = lm_head(h[:, :n_avail, :])[0]
            _P("lm_head", base_logits)
            if markov is None:
                if rejection:
                    qlp = _q_logsoftmax(base_logits)
                    drafted = mx.random.categorical(qlp).tolist()
                else:
                    qlp = None
                    drafted = mx.argmax(base_logits, axis=-1).tolist()
                n_sub = n_avail
            elif defer_sync and conf is None and (not rejection or truncated):
                # Same serial bigram chain, but `prev` never leaves the GPU: the
                # whole chain stays queued and the host reads it once, together
                # with the target's posterior, after the verify pass. With
                # truncated rejection the chain samples each slot from q'
                # instead of taking the argmax; the RNG call order matches the
                # synchronous branch below, so defer_sync moves only timing,
                # never tokens. (Plain temp-only rejection keeps the
                # synchronous branch it was measured on.)
                prev = mx.array([pending[-1]])
                dq, rows = [], []
                for j in range(n_avail):
                    z = markov.markov_w1(prev)
                    row = base_logits[j] + markov.markov_w2(z)[0]
                    if rejection:
                        row = _q_logsoftmax(row)
                        prev = mx.random.categorical(row).reshape(1)
                        rows.append(row)
                    else:
                        prev = mx.argmax(row, keepdims=True)
                    dq.append(prev)
                drafted_arr = mx.concatenate(dq)
                qlp = mx.stack(rows) if rows else None
                drafted, n_sub = None, n_avail
            else:
                rows, drafted, zs, prev = [], [], [], pending[-1]
                for j in range(n_avail):
                    z = markov.markov_w1(mx.array([prev]))
                    lg = base_logits[j] + markov.markov_w2(z)[0]
                    row = _q_logsoftmax(lg) if rejection else lg
                    prev = int(
                        mx.random.categorical(row).item() if rejection
                        else mx.argmax(row).item()
                    )
                    rows.append(row)
                    zs.append(z[0])
                    drafted.append(prev)

                n_sub = n_avail
                if conf is not None:
                    probs = mx.sigmoid(
                        conf(mx.concatenate([h[0, :n_avail], mx.stack(zs)], axis=-1))
                    ).tolist()
                    cum = 1.0
                    for j, pj in enumerate(probs):
                        cum *= pj
                        if cum < conf_tau:
                            n_sub = j + 1
                            break

                drafted = drafted[:n_sub]
                qlp = mx.stack(rows[:n_sub]) if rejection else None
            _P("markov")

            # The block's keys/values are speculative; the context's are not.
            for c in dcache:
                c.trim(block_size)
            dend = p_last
            _P("dtrim")

        # ── Verify. `pending` rides at the front of the batch: the target has to
        # run anyway, and carrying it here is what removes the replay pass.
        snap = _snapshot(tcache)
        vin = (
            mx.concatenate([mx.array(pending), drafted_arr.astype(mx.int32)])[None]
            if drafted is None
            else mx.array([pending + drafted])
        )
        logits = model(vin, cache=tcache, tap_layers=tap_layers)
        _P("verify", logits)
        taps_cat = tapped(model._taps)
        taps_base = base
        if rejection and n_sub == 0:
            # The width clamp left no room to speculate, so this is a plain
            # sampled step: one token from the truncated (or plain-temp) target
            # law. The general path below cannot serve it — there is no q'.
            # (Pre-truncation code crashed here on qlp=None; the clamp that
            # creates n_sub == 0 postdates the last temp>0 measurement.)
            plp = _p_logsoftmax(logits[0, L - 1 : L, :])
            posterior = [int(mx.random.categorical(plp[0]).item())]
        elif rejection:
            # Rows L-1 .. L+n_spec-1: one target distribution per draft slot,
            # plus the bonus that follows a fully accepted block.
            plp = _p_logsoftmax(logits[0, L - 1 : L + n_sub, :])
            xs = drafted_arr if drafted is None else mx.array(drafted)
            p_at = mx.take_along_axis(plp[:n_sub], xs[:, None], axis=-1).squeeze(-1)
            q_at = mx.take_along_axis(qlp, xs[:, None], axis=-1).squeeze(-1)
            accept = mx.log(mx.random.uniform(shape=(n_sub,))) < (p_at - q_at)
            resid = mx.maximum(mx.exp(plp[:n_sub]) - mx.exp(qlp), 0.0)
            res_tok = mx.random.categorical(mx.log(resid + 1e-30), axis=-1)
            # On rejection p(x) < q(x), so the residual has no mass at x — the
            # equality test below then reads acceptance off `posterior` for free.
            head = mx.where(accept, xs, res_tok)
            bonus = mx.random.categorical(plp[n_sub])
            if drafted is None:
                # Deferred markov chain: the drafts and the verdicts leave the
                # GPU in the same single transfer, like the greedy defer path.
                both = mx.concatenate(
                    [xs.astype(mx.int32), head.astype(mx.int32),
                     bonus.reshape(1).astype(mx.int32)]
                ).tolist()
                drafted, posterior = both[:n_sub], both[n_sub:]
            else:
                posterior = mx.concatenate([head, bonus.reshape(1)]).tolist()
        elif drafted is None:
            # One host sync for the whole step: the queued draft chain and the
            # target's posterior are read out of the same transfer.
            both = mx.concatenate(
                [drafted_arr.astype(mx.int32),
                 sampler(logits)[0][L - 1 : L + n_sub].astype(mx.int32)]
            ).tolist()
            drafted, posterior = both[:n_sub], both[n_sub:]
        else:
            posterior = sampler(logits)[0].tolist()[L - 1 : L + n_sub]
        mx.eval(taps_cat)
        _P("posterior")

        # posterior[i] is the target's token for the position draft i occupies,
        # so a draft survives exactly while it still equals it.
        n_acc = 0
        for i, d in enumerate(drafted):
            if d != posterior[i]:
                break
            n_acc += 1
        emitted = posterior[: n_acc + 1]

        for t in emitted:
            n += 1
            yield mx.array([t]), n_acc + 1
            if n >= max_tokens:
                break

        if n_acc == n_sub:
            # Nothing was rejected, so the verification stands as written and the
            # cache is already correct — a good drafter pays off twice here.
            base += L + n_sub
            pending = [emitted[-1]]
        else:
            _rewind(tcache, snap)
            pending = pending + emitted
            if len(pending) > max_pending:
                # Rejection streak. Stop carrying and commit, at the cost of one
                # unspeculated pass.
                model(
                    mx.array([pending[:-1]]), cache=tcache, tap_layers=tap_layers
                )
                taps_cat = tapped(model._taps)
                taps_base = base
                base += len(pending) - 1
                pending = [pending[-1]]
                if stats is not None:
                    stats["flush"] = stats.get("flush", 0) + 1

        mx.eval([c.state for c in tcache])
        _P("commit")
        if stats is not None:
            stats.setdefault("steps", []).append(
                (L, n_sub, L + n_sub, n_acc, _time.perf_counter() - _t0)
            )
