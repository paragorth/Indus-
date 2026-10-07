"""v87 shared analysis helpers: bank loading, ABC runs over chunks, controls."""
import os, json, random, math
from collections import Counter, defaultdict
import numpy as np
import v87_lib as L

FE = L.TRAIN + L.HELD
TI = [FE.index(k) for k in L.TRAIN]; HI = [FE.index(k) for k in L.HELD]


def load_bank(name):
    R = []
    for l in open(os.path.join(L.CK, 'bank_%s.jsonl' % name)):
        try: R.append(json.loads(l))
        except Exception: pass
    R.sort(key=lambda r: r['seed'])
    F = np.array([r['feats'] for r in R], float); P = np.array([r['props'] for r in R], float)
    good = np.isfinite(F).all(1)
    R = [r for r, g in zip(R, good) if g]
    return R, F[good], P[good]


def fvec(lines):
    f = L.feats(lines); return np.array([float(f[k]) for k in FE])


def regenerate(seed, prior='base'):
    import v87_bank as B
    rng = random.Random(seed); sid = rng.choice(B.SIDS)
    sc = (B.wide_scheme if prior == 'wide' else L.random_scheme)(rng)
    ents, lines, n = L.simulate(B.S[sid], sc, rng)
    return sid, sc, ents, lines


def trigram_resynth(lines, seed=0):
    """meaningless control: glyph order-2 Markov words (start/end states) trained on the chunk, same line lengths."""
    rng = random.Random(seed)
    T = defaultdict(Counter)
    for Lw in lines:
        for w in Lw:
            s = ('^', '^') + tuple(w) + ('$',)
            for i in range(2, len(s)):
                T[(s[i - 2], s[i - 1])][s[i]] += 1
    TT = {k: (list(v), np.cumsum(list(v.values())) / sum(v.values())) for k, v in T.items()}
    def word():
        a, b = '^', '^'; w = []
        while len(w) < 20:
            ks, cs = TT[(a, b)]; c = ks[int(np.searchsorted(cs, rng.random()))]
            if c == '$': break
            w.append(c); a, b = b, c
        return tuple(w) if w else ('x',)
    return [[word() for _ in Lw] for Lw in lines]


def glyph_shuffle(lines, seed=0):
    """kill control: all glyphs of the chunk permuted, word lengths and line lengths kept."""
    rng = random.Random(seed)
    g = [x for Lw in lines for w in Lw for x in w]; rng.shuffle(g)
    out = []; i = 0
    for Lw in lines:
        nl = []
        for w in Lw:
            nl.append(tuple(g[i:i + len(w)])); i += len(w)
        out.append(nl)
    return out


def word_shuffle(lines, seed=0):
    rng = random.Random(seed)
    ws = [w for Lw in lines for w in Lw]; rng.shuffle(ws)
    out = []; i = 0
    for Lw in lines:
        out.append(ws[i:i + len(Lw)]); i += len(Lw)
    return out


def summarize(posts):
    """average of per-chunk posteriors."""
    o = {}
    for p in L.PROPS:
        o[p] = float(np.mean([x[p] for x in posts]))
        o[p + '_sd_chunks'] = float(np.std([x[p] for x in posts]))
    o['dist'] = float(np.mean([x['dist'] for x in posts]))
    o['types_per_1000'] = float(1000 * math.exp(o['ptt']))
    return o


def run_abc(F, P, vecs, scale, q=0.02, mask=None):
    return [L.abc(F[:, TI], P, v[TI], scale, q=q, mask=mask) for v in vecs]


def ppc(F, post_list, vecs):
    """posterior predictive percentile of each held-out statistic (pooled accepted sims vs chunk values)."""
    res = {}
    for j, k in zip(HI, L.HELD):
        pct = []
        for po, v in zip(post_list, vecs):
            sims = F[po['idx'], j]
            pct.append(float((sims < v[j]).mean()))
        res[k] = dict(obs=float(np.mean([v[j] for v in vecs])), pred=float(np.mean([np.median(F[po['idx'], j]) for po in post_list])),
                      pct=float(np.mean(pct)))
    return res
