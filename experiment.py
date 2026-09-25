"""One search condition = real-lexicon restarts + search-matched fake-lexicon null.

For a condition (language, key type, N) this runs
  * R restarts of the search against the real lexicon on the training half;
  * one search against each of F fake lexicons (same procedure, same budget);
and for every resulting key measures the held-out rate against the lexicon it
was fitted to, plus the held-out scrambled rate.

  gap-over-fake (SD) = (mean real held-out rate - mean fake held-out rate) / sd(fake)

A condition passes only if the best-by-train real key also passes the fixed-key
validator AND gap-over-fake > 3 AND real held-out > real scrambled.
"""
import os
import random
import statistics
from collections import Counter, defaultdict

import indus_core as C
import search as S
import validator as V

_CACHE = {}


def _lex(lang, _unused=None):
    if lang not in _CACHE:
        _CACHE[lang] = C.load_lexicon(lang)
    return _CACHE[lang]


def index_texts(texts, top):
    idx = {s: i for i, s in enumerate(top)}
    ws, total = V.windows(texts, idx)
    return [tuple(idx[x] for x in w) for w in ws], total


def pairs_index(texts, top):
    """Adjacent sign pairs only (logographic keys)."""
    idx = {s: i for i, s in enumerate(top)}
    out, total = [], 0
    for segs in texts:
        for s in segs:
            for i in range(len(s) - 1):
                total += 1
                if s[i] in idx and s[i + 1] in idx:
                    out.append((idx[s[i]], idx[s[i + 1]]))
    return out, total


# ---------------------------------------------------------------- tasks
# A task is a plain dict so it can be shipped to a worker process.


def run_task(t):
    """Fit one key; return held-out real/scrambled rates for it."""
    lang, kind, N, seed, fake = t["lang"], t["kind"], t["N"], t["seed"], t["fake"]
    train, held, top = t["train"], t["held"], t["top"][:N]
    lex = t.get("lex") or _lex(lang, None)

    def target(mode):
        return C.forms(lex, mode) if fake is None else C.fake_forms(lex, mode, fake)
    iters = t["iters"]
    if kind in ("full", "skel"):
        mode = kind
        fs = target(mode)
        units = C.unit_inventory(lex, mode)
        W, _ = index_texts(train, top)
        prob = S.Problem(W, N)
        hit = S.string_hit_fn(fs, C.MINLEN[mode])
        score, key = S.anneal(prob, units, hit, seed, iters=iters)
        kmap = {top[i]: key[i] for i in range(N)}
        ev = lambda tx: V.rate(kmap, tx, fs, C.MINLEN[mode])
    elif kind == "logo":
        roots, pairset = compound_graph(lex, fake)
        P, _ = pairs_index(train, top)
        prob = S.Problem(P, N)
        score, key = S.anneal(prob, roots, S.pair_hit_fn(pairset), seed, iters=iters)
        kmap = {top[i]: key[i] for i in range(N)}
        ev = lambda tx: logo_rate(kmap, tx, pairset)
    elif kind == "mixed":
        mode = "full"
        fs = target(mode)
        n_syl = t.get("n_syl", 80)
        units = C.unit_inventory(lex, mode)
        words = sorted(w for w in fs if 3 <= len(w) <= 6)
        cand = [units if i < n_syl else words for i in range(N)]
        W, _ = index_texts(train, top)
        prob = S.Problem(W, N)
        hit = S.string_hit_fn(fs, C.MINLEN[mode])
        score, key = S.anneal(prob, None, hit, seed, iters=iters, values_for=cand)
        kmap = {top[i]: key[i] for i in range(N)}
        ev = lambda tx: V.rate(kmap, tx, fs, C.MINLEN[mode])
    else:
        raise ValueError(kind)
    real = ev(held)
    scr = statistics.mean(ev(V.scramble(held, 100 + i)) for i in range(3))
    return {"lang": lang, "kind": kind, "N": N, "seed": seed, "fake": fake,
            "train_score": score, "held": real, "held_scrambled": scr, "key": kmap}


# ---------------------------------------------------------------- logographic


def compound_graph(lex, rewire_seed=None, max_root=6, min_root=int(os.environ.get("LOGO_MINROOT", 3))):
    """Roots = short lexicon words; a pair (a, b) is attested if a+b, or a+b
    with the initial stop of b doubled (Tamil sandhi: maram + kal -> maramkkal
    is rare, but pu + kal -> pukkal is common), is itself a lexicon word.
    rewire_seed: degree-preserving shuffle of the second elements (the null)."""
    key = ("graph", len(lex), rewire_seed, min_root)
    if key in _CACHE:
        return _CACHE[key]
    roots = {w for w in lex if min_root <= len(w) <= max_root}
    pairs = set()
    for w in lex:
        for i in range(min_root, len(w) - min_root + 1):
            a, b = w[:i], w[i:]
            if a in roots and b in roots:
                pairs.add((a, b))
            if (len(b) > 2 and b[0] == b[1] and b[0] in "kctp" and a in roots
                    and b[1:] in roots):
                pairs.add((a, b[1:]))
    pairs = sorted(pairs)
    if rewire_seed is not None:
        firsts = [a for a, _ in pairs]
        seconds = [b for _, b in pairs]
        random.Random(rewire_seed).shuffle(seconds)
        pairs = sorted(set(zip(firsts, seconds)))
    used = sorted({a for a, _ in pairs} | {b for _, b in pairs})
    res = (used, set(pairs))
    _CACHE[key] = res
    return res


def logo_rate(kmap, texts, pairset):
    hits = total = 0
    for segs in texts:
        for s in segs:
            for i in range(len(s) - 1):
                total += 1
                a, b = kmap.get(s[i]), kmap.get(s[i + 1])
                if a is not None and b is not None and (a, b) in pairset:
                    hits += 1
    return hits / total if total else 0.0


# ---------------------------------------------------------------- summary


def summarise(results, n_real):
    real = [r for r in results if r["fake"] is None]
    fake = [r for r in results if r["fake"] is not None]
    rh = [r["held"] for r in real]
    fh = [r["held"] for r in fake]
    mu_f, sd_f = statistics.mean(fh), statistics.pstdev(fh)
    gap = (statistics.mean(rh) - mu_f) / sd_f if sd_f > 0 else float("nan")
    best = max(real, key=lambda r: r["train_score"])
    bf = max(fake, key=lambda r: r["train_score"])
    return {
        "real_held_mean": statistics.mean(rh), "real_held_sd": statistics.pstdev(rh),
        "real_scr_mean": statistics.mean(r["held_scrambled"] for r in real),
        "fake_held_mean": mu_f, "fake_held_sd": sd_f, "gap_sd": gap,
        "best_real_held": best["held"], "best_real_train": best["train_score"],
        "best_fake_train": bf["train_score"],
        "n_real": len(real), "n_fake": len(fake), "best_key": best["key"],
    }


def stability(results, top_n=20):
    """Signs whose modal value recurs across real restarts; flag the ones whose
    modal value is also modal across fake-lexicon runs (letter-frequency, not language)."""
    real = [r["key"] for r in results if r["fake"] is None]
    fake = [r["key"] for r in results if r["fake"] is not None]
    out = []
    for s in real[0]:
        c = Counter(k[s] for k in real)
        v, n = c.most_common(1)[0]
        cf = Counter(k[s] for k in fake)
        fv = cf.most_common(1)[0][0] if cf else None
        out.append((n / len(real), s, v, cf.get(v, 0) / max(1, len(fake)), v == fv))
    out.sort(reverse=True)
    return out[:top_n]
