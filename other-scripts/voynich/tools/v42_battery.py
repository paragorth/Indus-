"""v42 cycle 2 data: the full statistical battery per text block, at two scales.

Scale LONG = 3000-token blocks (up to 4 per corpus; generators are 3000 tokens long), scale SHORT = 250-token
blocks (up to 4 per corpus; the 37 Gaskell-Bowern gibberish writers have 80-480 tokens).
Metrics (alphabet-normalised where possible):
  zipf      log-log slope of frequency vs rank (ranks 1..V/2, at most 300)
  heaps     log-log slope of vocabulary vs tokens
  ttr, hapax, wl_cv
  h2r, h3r  conditional unit entropy given 1 / 2 previous units (word boundaries kept) / unigram entropy
  junc      MI(last unit of word i, first unit of word i+1) minus its word-shuffle mean, / h1
  arrow     v23 glyph arrow: KL(P(a,b) || P(b,a)) of adjacent units in the running stream, / lag-1 MI
  warrow    same on word-length pairs (alphabet free)
  linit     MI(first unit, line-initial or not) minus within-line word shuffle, / h1   (line effects)
  lfin      same for the last unit and line-final position
  rig, posmi  v31 slot dependence: unit-order rigidity over types; unit-position MI / h1
  gap       v33 gap ratio: stem x ending grid connectance / that of the block's own unit-trigram resynthesis,
            median over 14 segmentation rules (LONG only)
Output data/v42_ckpt/battery.json"""
import os, sys, json, random, math, zlib
import numpy as np
from collections import Counter, defaultdict
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v42_lib as L, v31_lib as L31

SCALES = {'LONG': 3000, 'SHORT': 250}
NB = 4


def H(c): return L31.H(c)


def condH(seqs, k):
    j = Counter(); c = Counter()
    for s in seqs:
        x = '^' * k + s + '$'
        for i in range(k, len(x)):
            j[x[i - k:i + 1]] += 1; c[x[i - k:i]] += 1
    return H(j) - H(c)


def kl_asym(cnt, eps=0.5):
    keys = set(cnt) | {(b, a) for a, b in cnt}
    tot = sum(cnt.values()) + eps * len(keys); s = 0.0
    for (a, b) in keys:
        p = (cnt.get((a, b), 0) + eps) / tot; q = (cnt.get((b, a), 0) + eps) / tot
        s += p * math.log2(p / q)
    return s


def mi_pairs(c):
    a = Counter(); b = Counter()
    for (x, y), v in c.items(): a[x] += v; b[y] += v
    return H(a) + H(b) - H(c)


def line_mi(lines, rng, first=True, nsh=10):
    def stat(ls):
        pr = []
        for l in ls:
            for i, w in enumerate(l):
                pr.append(((w[0] if first else w[-1]), (i == 0) if first else (i == len(l) - 1)))
        return L31.mi(pr)
    o = stat(lines); sh = []
    for _ in range(nsh):
        q = [l[:] for l in lines]
        for l in q: rng.shuffle(l)
        sh.append(stat(q))
    return o - float(np.mean(sh))


def gap_ratio(toks, rng):
    import v33_lib as V33
    rules = [V33.Rule('fix', k) for k in (1, 2, 3)] + [V33.Rule('pos', t) for t in (0.5, 0.6, 0.7, 0.8)] + \
            [V33.Rule('freq', K) for K in (10, 25, 50, 100)] + [V33.Rule('harris', 0)] + \
            [V33.Rule('rand', 0, seed=s) for s in (1, 2)]
    tri = V33.Trigram([toks[i:i + 10] for i in range(0, len(toks), 10)])
    gen = tri.gen(len(toks), rng)
    out = []
    for r in rules:
        a = V33.matrix(toks, V33.Rule(r.kind, r.param, r.seed).fit(toks), R=150, E=20).mean()
        b = V33.matrix(gen, V33.Rule(r.kind, r.param, r.seed).fit(gen), R=150, E=20).mean()
        if b > 0: out.append(a / b)
    return float(np.median(out))


def battery(lines, rng, scale):
    t = [w for l in lines for w in l]; n = len(t)
    F = {}
    cnt = Counter(t); fr = np.array(sorted(cnt.values(), reverse=True), float)
    R = max(5, min(len(fr) // 2, 300))
    F['zipf'] = float(np.polyfit(np.log(np.arange(1, R + 1)), np.log(fr[:R]), 1)[0])
    pts = np.unique(np.geomspace(20, n, 12).astype(int)); seen = set(); V = []; j = 0
    for i, w in enumerate(t, 1):
        seen.add(w)
        if j < len(pts) and i == pts[j]: V.append(len(seen)); j += 1
    F['heaps'] = float(np.polyfit(np.log(pts[:len(V)]), np.log(V), 1)[0])
    F['ttr'] = len(cnt) / n
    F['hapax'] = sum(1 for v in cnt.values() if v == 1) / n
    Lw = np.array([len(w) for w in t], float); F['wl_cv'] = Lw.std() / Lw.mean()
    h1 = H(Counter(c for w in t for c in w))
    F['h2r'] = condH(t, 1) / h1; F['h3r'] = condH(t, 2) / h1
    jp = lambda q: Counter((q[i][-1], q[i + 1][0]) for i in range(len(q) - 1))
    o = mi_pairs(jp(t)); sh = []
    for _ in range(10):
        q = t[:]; rng.shuffle(q); sh.append(mi_pairs(jp(q)))
    F['junc'] = (o - float(np.mean(sh))) / h1
    s = ' '.join(t); gp = Counter(zip(s, s[1:]))
    F['arrow'] = kl_asym(gp) / max(1e-9, mi_pairs(gp))
    lp = Counter((min(len(t[i]), 9), min(len(t[i + 1]), 9)) for i in range(n - 1))
    F['warrow'] = kl_asym(lp) / max(1e-9, mi_pairs(lp))
    ls = [l for l in lines if len(l) >= 3]
    F['linit'] = line_mi(ls, rng, True) / h1 if len(ls) >= 5 else float('nan')
    F['lfin'] = line_mi(ls, rng, False) / h1 if len(ls) >= 5 else float('nan')
    F['rig'] = L31.order_rigidity(list(cnt))
    pos = []
    for w in t:
        if len(w) == 1: pos.append((w, 's')); continue
        pos.append((w[0], 'f')); pos.append((w[-1], 'l')); pos += [(c, 'm') for c in w[1:-1]]
    F['posmi'] = L31.mi(pos) / h1
    if scale == 'LONG': F['gap'] = gap_ratio(t, rng)
    return F


def blocks(docs, n, nb=NB):
    """nb blocks of n consecutive tokens (keeping lines), spread over the corpus."""
    S = L31.samples(docs, N=n, maxs=nb)
    return S


def corpora():
    import v35_lib as V35
    C = L.all_corpora()
    keep = {}
    for k, (cls, d) in C.items():
        if cls == 'MAGIC': continue
        keep[k] = (cls, d)
    def wrap(ws, w=9):
        return [[ws[i:i + w] for i in range(0, len(ws), w)]]
    sym = {}
    def enc(tup):
        return ''.join(sym.setdefault(g, chr(0x4e00 + len(sym))) for g in tup)
    keep['C_Copiale'] = ('CIPH', wrap([enc(w) for w in V35.copiale_words()]))
    keep['C_Borg'] = ('CIPH', wrap([enc(w) for w in V35.borg_words()]))
    keep['V_ZL_A'] = ('TEST', L.voynich_docs('ZL3b', 'A')); keep['V_ZL_B'] = ('TEST', L.voynich_docs('ZL3b', 'B'))
    return keep


def jobs():
    C = corpora(); J = []
    variants = {}
    for k, (cls, d) in C.items():
        variants[(k, 'clean')] = (cls, d)
        if not k.startswith('S_'):
            variants[(k, 'n18')] = (cls, L.noise_docs(d, 0.18, seed=zlib.crc32(k.encode()) & 0xffff))
        else:
            variants[(k, 'col')] = (cls, L.map_docs(d, L.collapse))
    for (k, var), (cls, d) in variants.items():
        for sc, n in SCALES.items():
            nb = 12 if k.startswith(('S_CS', 'V_ZL', 'V_IT')) and k in ('S_CS', 'V_ZL', 'V_IT') else NB
            for i, b in enumerate(blocks(d, n, nb)):
                if sum(len(l) for l in b) < n: continue
                J.append((k, cls, var, sc, i, b))
    return J


def work(a):
    k, cls, var, sc, i, b = a
    rng = random.Random(zlib.crc32(f'{k}|{var}|{sc}|{i}'.encode()))
    try:
        F = battery(b, rng, sc)
    except Exception as e:
        return {'corpus': k, 'err': repr(e)}
    return {'corpus': k, 'cls': cls, 'var': var, 'scale': sc, 'i': i, 'F': F}


if __name__ == '__main__':
    J = jobs(); print('jobs', len(J), flush=True)
    with Pool(2) as p: R = p.map(work, J, chunksize=4)
    print('errors', [r for r in R if 'err' in r][:5])
    L.save('battery.json', [r for r in R if 'err' not in r])
    print('done', len(R))
