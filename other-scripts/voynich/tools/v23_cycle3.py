"""v23 cycle 3: massive random search for arrows of time ('probes'), held-out replication.

A probe is (level, random class map, lag k, ordered class pair (a,b)).  Per page,
D_p = #(a then b at lag k) - #(b then a at lag k)  (within lines for glyph/word level, within
paragraphs for line level).  Reversal flips the sign of D_p exactly.
6,000 random probes per corpus (2,000 per level).  Pages split at random into discovery (A) and
held-out (B) halves; a probe 'survives' if |z_A| >= 3 and z_B >= 2 in the predicted direction.
Nulls: the same corpus with every page read in a random direction (exact matching, 4 draws),
and reversible-chain surrogates.  Positive controls: real languages and the planted reversed Latin.
"""
import sys, os, json, math, random, time, zlib
from collections import Counter, defaultdict
import numpy as np
import scipy.sparse as sp
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v23_lib as L
from v23_cycle1 import build

NPROBE = 2000
LAGS_G, LAGS_W, LAGS_L = range(1, 7), range(1, 5), range(1, 3)


def page_mats(C, level, alphabet_ix, lag, tokfn):
    """per page antisymmetric pair-count matrix (sparse) M - M^T for given lag"""
    V = len(alphabet_ix); out = []
    for p in C:
        rows, cols = [], []
        for seq in tokfn(p):
            ids = [alphabet_ix[x] for x in seq]
            for i in range(len(ids) - lag):
                rows.append(ids[i]); cols.append(ids[i + lag])
        M = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(V, V)).tocsr()
        out.append(M - M.T)
    return out


def level_setup(C, level):
    if level == 'glyph':
        tokfn = lambda p: [list(L.SEP_L.join(L.SEP_W.join(l) for l in pa)) for pa in p]
    elif level == 'word':
        tokfn = lambda p: [l for pa in p for l in pa]
    else:  # line: each line -> its first word and last word as a 2-symbol unit? use first word
        tokfn = lambda p: [[l[0] for l in pa] for pa in p] + [[('$' + l[-1]) for l in pa] for pa in p]
    alpha = sorted({x for p in C for s in tokfn(p) for x in s})
    return tokfn, {x: i for i, x in enumerate(alpha)}, alpha


def random_classes(alpha, level, rng, gf):
    """returns class vector (len(alpha)) with values 0..m-1, description"""
    m = rng.choice([2, 3, 4])
    V = len(alpha)
    if level == 'glyph':
        seps = {L.SEP_W, L.SEP_L}
        c = np.array([m if a in seps else rng.randrange(m) for a in alpha]); return c, m + 1, 'gly-partition'
    kind = rng.choice(['random', 'first', 'last', 'len', 'freq'])
    strip = lambda a: a[1:] if a.startswith('$') else a
    if kind == 'random':
        c = np.array([rng.randrange(m) for _ in alpha])
    elif kind in ('first', 'last'):
        gl = sorted({(strip(a)[0] if kind == 'first' else strip(a)[-1]) for a in alpha})
        g = {x: rng.randrange(m) for x in gl}
        c = np.array([g[strip(a)[0] if kind == 'first' else strip(a)[-1]] for a in alpha])
    elif kind == 'len':
        th = sorted(rng.sample(range(2, 9), m - 1))
        c = np.array([sum(len(strip(a)) > t for t in th) for a in alpha])
    else:
        th = sorted(rng.sample(range(1, 7), m - 1))
        c = np.array([sum(math.log2(gf.get(strip(a), 0) + 1) > t for t in th) for a in alpha])
    if level == 'line':  # '$' symbols (line-final words) form their own pool of classes
        c = np.array([c[i] + (m if a.startswith('$') else 0) for i, a in enumerate(alpha)]); m *= 2
    return c, m, kind


def run(arg):
    name, mode = arg          # mode: 'real' or 'rand<k>' (each page in random direction)
    fn = os.path.join(L.CK, f'c3_{name}_{mode}.json')
    if os.path.exists(fn): return arg
    t0 = time.time()
    base = build(name)
    gf = L.gfreq(base)
    split = random.Random(5)
    A = set(split.sample(range(len(base)), len(base) // 2))
    res = {'name': name, 'mode': mode, 'levels': {}}
    for level in ('glyph', 'word', 'line'):
        CC = base
        if mode.startswith('rand'):
            r = random.Random(100 + int(mode[4:]))
            rv = L.rev_page if level == 'glyph' else L.rev_words_only
            CC = [rv(p) if r.random() < 0.5 else p for p in base]
        tokfn, aix, alpha = level_setup(CC, level)
        lags = LAGS_G if level == 'glyph' else LAGS_W if level == 'word' else LAGS_L
        mats = {k: page_mats(CC, level, aix, k, tokfn) for k in lags}
        rng = random.Random(zlib.crc32(level.encode()) + 17)
        surv, zs = [], []
        for i in range(NPROBE):
            c, m, kind = random_classes(alpha, level, rng, gf)
            k = rng.choice(list(lags))
            a, b = rng.sample(range(m), 2)
            u = (c == a).astype(float); v = (c == b).astype(float)
            D = np.array([u @ (M @ v) for M in mats[k]])
            DA = np.array([D[j] for j in range(len(CC)) if j in A]); DB = np.array([D[j] for j in range(len(CC)) if j not in A])
            zA = DA.sum() / math.sqrt((DA ** 2).sum() + 1e-9); zB = DB.sum() / math.sqrt((DB ** 2).sum() + 1e-9)
            zs.append((zA, zB))
            if abs(zA) >= 3 and np.sign(zB) == np.sign(zA) and abs(zB) >= 2:
                desc = {'kind': kind, 'lag': k, 'a': [alpha[j] for j in np.where(u)[0][:12]],
                        'b': [alpha[j] for j in np.where(v)[0][:12]], 'na': int(u.sum()), 'nb': int(v.sum()),
                        'zA': float(zA), 'zB': float(zB), 'sum': float(D.sum())}
                surv.append(desc)
        zs = np.array(zs)
        nA = int((np.abs(zs[:, 0]) >= 3).sum())
        res['levels'][level] = {'n_probe': NPROBE, 'n_discA': nA, 'n_surv': len(surv),
                                'rep_rate': len(surv) / max(nA, 1),
                                'corr_zA_zB': float(np.corrcoef(zs[:, 0], zs[:, 1])[0, 1]),
                                'median_absz': float(np.median(np.abs(zs[:, 0]))),
                                'top': sorted(surv, key=lambda d: -abs(d['zA']) - abs(d['zB']))[:15]}
    res['sec'] = time.time() - t0
    json.dump(res, open(fn, 'w'))
    print(name, mode, {l: (v['n_discA'], v['n_surv'], round(v['corr_zA_zB'], 3)) for l, v in res['levels'].items()},
          round(res['sec']), flush=True)
    return arg


if __name__ == '__main__':
    names = sys.argv[1:] or ['ZL', 'IT', 'LA', 'ITA', 'DE', 'CS', 'HE', 'PL_REV', 'MkG_1', 'MkW_1', 'RvG_0', 'RvW_0']
    jobs = [(n, 'real') for n in names] + [(n, f'rand{i}') for n in names[:7] for i in range(3)]
    with Pool(2) as pool:
        for _ in pool.imap_unordered(run, jobs): pass
