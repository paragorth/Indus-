"""v16 cycle 3: if the spaces are a mechanical rule, the rule should be readable off the glyph stream.

(A) Space predictability: with the spaces deleted, predict at every internal glyph gap whether a written
    space stands there, from the k glyphs on each side (k = 1, 2, 3), and from the same context plus the
    run length since the last space (the 'after a fixed glyph count' hypothesis). Model: count table on even
    lines with add-0.5 smoothing and back-off to k-1; scored on odd lines as cross-entropy (bits per gap,
    as a fraction of the unconditional space entropy) and accuracy. A mechanical spacer gives ~0 residual
    entropy; a real language leaves a lot.
    Controls: true-spaced Latin/Italian (real word breaks), planted-spaced Latin/Italian (a mechanical rule:
    must give ~0), Markov-2 resynthesis (local rule with randomness), glyph-shuffle (spaces unrelated).
(B) Units found in cycles 1-2 on the Voynich: description (counts, reuse, length, positional behaviour,
    relation to written words), junction excess, and the line-initial chain measured on units.
"""
import os, sys, json, math
import numpy as np
from collections import defaultdict, Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v16_lib as L
from v16_cycle1 import build, NAMES

OUT = os.path.join(L.RESDIR, 'cycle3')
os.makedirs(OUT, exist_ok=True)


def gap_contexts(c, k, runlen=False):
    """For every internal gap (position i, not a line start): context tuple, label = written[i]."""
    idx = np.flatnonzero(~c.ls)
    pad = c.A  # boundary pad symbol
    ctxs = []
    lid = c.line_id; sym = c.sym; n = c.n; wr = c.written
    # run length since last written start (capped at 8)
    run = np.zeros(n, int)
    last = 0
    for i in range(n):
        if wr[i]: last = i
        run[i] = min(i - last, 8)
    for i in idx.tolist():
        L_ = tuple(int(sym[j]) if j >= 0 and lid[j] == lid[i] else pad for j in range(i - k, i))
        R_ = tuple(int(sym[j]) if j < n and lid[j] == lid[i] else pad for j in range(i, i + k))
        key = L_ + R_
        if runlen:
            key = key + (int(run[i - 1]) + 1,)   # glyphs since the last space, before this gap
        ctxs.append(key)
    return ctxs, wr[idx].astype(int), c.line_id[idx]


def predictability(c, k, runlen=False):
    ctxs, y, lid = gap_contexts(c, k, runlen)
    tr = lid % 2 == 0
    tabs = []
    # back-off chain: full context -> drop outermost glyphs symmetrically
    def reduce(key, level):
        if level == 0: return key
        kk = k - level
        core = key[k - kk:k + kk] if kk > 0 else ()
        return core + (key[2 * k:] if runlen else ())
    for level in range(k + 1):
        t = defaultdict(lambda: [0, 0])
        for key, lab, m in zip(ctxs, y, tr):
            if m: t[reduce(key, level)][lab] += 1
        tabs.append(t)
    p1 = y[tr].mean()
    H0 = -(p1 * math.log2(p1) + (1 - p1) * math.log2(1 - p1))
    ce = 0.0; corr = 0; nte = 0
    for key, lab, m in zip(ctxs, y, tr):
        if m: continue
        # interpolated back-off (Witten-Bell-like): start at coarsest, refine
        p = p1
        for level in range(k, -1, -1):
            a, b = tabs[level].get(reduce(key, level), (0, 0))
            nn = a + b
            if nn: p = (b + 2 * p) / (nn + 2)
        pl = p if lab else 1 - p
        ce += -math.log2(max(pl, 1e-12)); corr += int((p > 0.5) == bool(lab)); nte += 1
    return dict(k=k, runlen=runlen, H0=H0, ce=ce / nte, frac=ce / nte / H0, acc=corr / nte,
                base_acc=max(p1, 1 - p1), n=nte)


def part_a():
    path = os.path.join(OUT, 'predictability.json')
    if os.path.exists(path): return json.load(open(path))
    res = {}
    for name in NAMES:
        c = build(name)
        res[name] = [predictability(c, k, rl) for k in (1, 2, 3) for rl in (False, True)]
        print(name, [(r['k'], r['runlen'], round(r['frac'], 3), round(r['acc'], 3)) for r in res[name]], flush=True)
    json.dump(res, open(path, 'w'), indent=1)
    return res


if __name__ == '__main__':
    part_a()


# ---------------- (B) what do the re-spaced units look like? ----------------
def describe_units(c, st, rng, top=30):
    s, e = L.tokens(c, st)
    hv = L.token_hash(c, s, e)
    u, inv, cnt = np.unique(hv, return_inverse=True, return_counts=True)
    tokc = cnt[inv]
    # relation to written words
    wr = c.written
    cs = np.r_[0, np.cumsum(wr)]
    inner_spaces = cs[e] - cs[s + 1]           # written starts strictly inside (s, e)
    end_ok = np.r_[wr[e[:-1]], True] if len(e) else np.array([], bool)
    end_ok = np.array([True if x >= c.n else bool(wr[x]) for x in e])
    whole = (wr[s]) & end_ok & (inner_spaces == 0)
    cross = inner_spaces > 0
    sub = ~whole & ~cross
    # positional behaviour: MI(unit id among top-K, line-initial) and (unit, line-final)
    order = np.argsort(-cnt)[:top]
    remap = np.full(len(u), top); remap[order] = np.arange(top)
    uid = remap[inv]
    ini = c.ls[s].astype(int)
    fin = np.r_[c.ls[s[1:]], True].astype(int)
    def mi(a, b, A, B):
        t = np.bincount(a * B + b, minlength=A * B).reshape(A, B).astype(float); n = t.sum()
        pa = t.sum(1, keepdims=True) / n; pb = t.sum(0, keepdims=True) / n; p = t / n; m = p > 0
        return float(np.sum(p[m] * np.log2(p[m] / (pa @ pb)[m])))
    pos_ini = mi(uid, ini, top + 1, 2); pos_fin = mi(uid, fin, top + 1, 2)
    # line-initial chain on units: first unit of line r vs first unit of line r+1, excess over line shuffle
    fu = uid[ini == 1]
    fg = c.sym[s[ini == 1]]
    def chain(x, K):
        obs = mi(x[:-1], x[1:], K, K)
        nul = []
        for _ in range(50):
            y = rng.permutation(x); nul.append(mi(y[:-1], y[1:], K, K))
        return obs - float(np.mean(nul)), (obs - float(np.mean(nul))) / (float(np.std(nul)) + 1e-12)
    ch_u = chain(fu, top + 1); ch_g = chain(fg, c.A)
    tops = [(''.join(c.alphabet[x] for x in c.sym[s[np.flatnonzero(inv == j)[0]]:e[np.flatnonzero(inv == j)[0]]]), int(cnt[j])) for j in order[:15]]
    return dict(ntok=int(len(s)), ntype=int(len(u)), reuse=float(np.mean(tokc >= 2)),
                hapax_types=float(np.mean(cnt == 1)), whole=float(whole.mean()), sub=float(sub.mean()),
                cross=float(cross.mean()), mi_line_initial=pos_ini, mi_line_final=pos_fin,
                chain_units=ch_u, chain_glyph=ch_g, top=tops)
