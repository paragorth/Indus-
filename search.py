"""Simulated-annealing key search.

A key maps each of the top-N signs to a value (a syllable, a consonant, or a
whole word).  The objective is the number of 2-5-sign training windows whose
rendered string is in the target form set (or, for logographic keys, the
number of adjacent sign pairs whose word pair is an attested compound).
"""
import math
import random


class Problem:
    """Pre-indexed windows over sign indices 0..N-1 for fast delta scoring."""

    def __init__(self, windows, n_signs):
        self.windows = windows            # list of tuples of sign indices
        self.n = n_signs
        by = [set() for _ in range(n_signs)]
        for wi, w in enumerate(windows):
            for s in w:
                by[s].add(wi)
        self.by_sign = [sorted(b) for b in by]


def anneal(prob, values, hit_fn, seed, iters=60_000, t0=2.0, t1=0.02, fixed=None,
           values_for=None):
    """values: list of candidate values (default for every sign).
    values_for: optional per-sign candidate lists (mixed keys).
    hit_fn(window_values_tuple) -> bool.
    fixed: optional {sign_index: value} held constant."""
    rng = random.Random(seed)
    cand = values_for or [values] * prob.n
    key = [rng.choice(cand[s]) for s in range(prob.n)]
    if fixed:
        for s, v in fixed.items():
            key[s] = v
    free = [s for s in range(prob.n) if not fixed or s not in fixed]
    W = prob.windows
    hit = [hit_fn(tuple(key[x] for x in w)) for w in W]
    score = sum(hit)
    best, best_key = score, list(key)
    # weight sign choice by sqrt(#windows) so frequent signs are revisited often
    weights = [math.sqrt(len(prob.by_sign[s])) + 0.5 for s in free]
    for it in range(iters):
        T = t0 * (t1 / t0) ** (it / iters)
        if rng.random() < 0.8 or len(free) < 2:
            (s,) = rng.choices(free, weights)
            old = key[s]
            new = rng.choice(cand[s])
            if new == old:
                continue
            key[s] = new
            aff = prob.by_sign[s]
            newhits = [hit_fn(tuple(key[x] for x in W[wi])) for wi in aff]
            delta = sum(newhits) - sum(hit[wi] for wi in aff)
            if delta >= 0 or rng.random() < math.exp(delta / T):
                for wi, h in zip(aff, newhits):
                    hit[wi] = h
                score += delta
            else:
                key[s] = old
        else:
            a, b = rng.sample(free, 2)
            if key[a] == key[b] or (values_for and (key[b] not in cand[a] or key[a] not in cand[b])):
                continue
            key[a], key[b] = key[b], key[a]
            aff = sorted(set(prob.by_sign[a]) | set(prob.by_sign[b]))
            newhits = [hit_fn(tuple(key[x] for x in W[wi])) for wi in aff]
            delta = sum(newhits) - sum(hit[wi] for wi in aff)
            if delta >= 0 or rng.random() < math.exp(delta / T):
                for wi, h in zip(aff, newhits):
                    hit[wi] = h
                score += delta
            else:
                key[a], key[b] = key[b], key[a]
        if score > best:
            best, best_key = score, list(key)
    return best, best_key


def string_hit_fn(formset, minlen):
    def f(vals):
        r = "".join(vals)
        return len(r) >= minlen and r in formset
    return f


def pair_hit_fn(pairset):
    def f(vals):
        return vals in pairset
    return f
