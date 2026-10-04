"""v19 cycle 2: (a) ONE WRITER, ONE ALPHABET - cross-stretch prediction.
If several stretches are sorted under the writer's own glyph order, the order learned on stretch A must sort
stretch B (held out, fixed order, no re-optimisation; a shuffled B has E[tau] = 0 exactly).
Statistic per cell: mean over ordered pairs (A,B), A != B, of tau_B(order_A); plus the consensus order
(Borda over learned orders) applied to every stretch with leave-one-out.
Null for the mean: orders learned on within-stretch-shuffled copies (same optimiser) predicting real B
-> kills optimiser/frequency artefacts; and real orders predicting shuffled B.
Positive controls: LatX (alphabetical headwords); PLANT8 = 8 Voynich pages planted under ONE shared order
('head' and 'noisy' style); negative: LatXVI, LatXVII, PLANT (8 pages, 8 different orders).
(b) block-permute null (lines permuted within the stretch, word order inside a line kept) for the cycle-1
Voynich pagewords/win5 top stretches.
"""
import json, os, random, sys, math, zlib
import numpy as np
from multiprocessing import Pool
from scipy.stats import kendalltau
from v19_lib import *
from v19_cycle1 import corpora, stretches_for, planted_lines

OUT = os.path.join(RES, 'c2'); os.makedirs(OUT, exist_ok=True)
C1 = os.path.join(RES, 'c1')


def planted_shared(lines, seed=23):
    rng = random.Random(seed)
    order, by = pages_of(lines)
    cand = [p for p in order if len(by[p]) >= 12]
    pages = rng.sample(cand, 8)
    syms = sorted({g for L in lines for w in L['words'] for g in w})
    perm = syms[:]; rng.shuffle(perm); rank = {g: i for i, g in enumerate(perm)}
    types = [list(w) for w in {tuple(w) for L in lines for w in L['words']} if len(w) >= 2]
    out = []
    for k, p in enumerate(pages):
        src = [dict(lines[i]) for i in by[p]]
        if k % 2 == 0:  # glossary headwords at line starts
            heads = sorted(rng.sample(types, len(src)), key=sortkey_fn(rank))
            for L, h in zip(src, heads):
                L['words'] = [h] + L['words'][1:]; L['para_start'] = False
        else:  # whole page sorted with 40% displaced
            flat = sorted([w for L in src for w in L['words']], key=sortkey_fn(rank)); m = len(flat)
            for _ in range(int(0.4 * m)):
                i = rng.randrange(m); w = flat.pop(i); flat.insert(rng.randrange(m), w)
            pos = 0
            for L in src:
                n = len(L['words']); L['words'] = flat[pos:pos + n]; pos += n
        for L in src:
            L['page'] = 'PS-%s' % p; out.append(L)
    return out, perm


def tau_fixed(ent, rank, depth, rev):
    kf = sortkey_fn(rank, depth, rev)
    keys = [tuple(kf(w)) for _, _, w in ent]
    u = {k: i for i, k in enumerate(sorted(set(keys)))}
    y = [u[k] for k in keys]
    if len(set(y)) < 2:
        return None
    t = kendalltau(np.arange(len(y)), y, variant='b')[0]  # tau-b; ties in key ignored
    return t


def cross(cname, kind, key, st, orders, rng):
    depth, rev = KEYS[key]
    ranks = [{g: i for i, g in enumerate(o)} for o in orders]
    n = len(st)
    if n < 3:
        return None
    pairs = [(a, b) for a in range(n) for b in range(n) if a != b]
    if len(pairs) > 6000:
        pairs = rng.sample(pairs, 6000)
    vals = [tau_fixed(st[b][1], ranks[a], depth, rev) for a, b in pairs]
    vals = [v for v in vals if v is not None and not math.isnan(v)]
    # shuffled-target control (E = 0)
    sh = []
    for a, b in pairs[:2000]:
        e = st[b][1][:]; rng.shuffle(e)
        v = tau_fixed(e, ranks[a], depth, rev)
        if v is not None and not math.isnan(v):
            sh.append(v)
    # leave-one-out consensus (Borda of the other orders)
    loo = []
    for b in range(n):
        sc = Counter()
        for a in range(n):
            if a == b:
                continue
            o = orders[a]
            for i, g in enumerate(o):
                sc[g] += i / max(1, len(o) - 1)
        cons = {g: i for i, g in enumerate(sorted(sc, key=lambda g: sc[g] / max(1, sum(1 for o in orders if g in o))))}
        v = tau_fixed(st[b][1], cons, depth, rev)
        if v is not None and not math.isnan(v):
            loo.append(v)
    return dict(mean=float(np.mean(vals)), se=float(np.std(vals) / math.sqrt(len(vals))), npairs=len(vals),
                shmean=float(np.mean(sh)) if sh else 0, shse=float(np.std(sh) / math.sqrt(max(1, len(sh)))),
                loo=float(np.mean(loo)) if loo else 0, looz=float(np.mean(loo) / (np.std(loo) / math.sqrt(len(loo)))) if len(loo) > 2 and np.std(loo) > 0 else 0)


def job(args):
    cname, kind, key = args
    fn = os.path.join(OUT, 'x_%s_%s_%s.json' % (cname, kind, key))
    if os.path.exists(fn):
        return fn
    rng = random.Random(zlib.crc32(fn.encode()))
    C, _ = corpora()
    if cname == 'PSHARED':
        lines, perm = planted_shared(C['ZL'])
        C['PSHARED'] = lines
    st = stretches_for(C, cname, kind)
    st = [s for s in st if len(s[1]) >= 8]
    if len(st) < 3:
        json.dump({}, open(fn, 'w')); return fn
    sym = encode(st)
    # learned orders on real stretches (re-run engine with R=0 to get orders)
    rows = run_engine(st, sym, key, R=0, restarts=6, ils=25, nrand=0, seed=3, tag='c2' + cname + kind + key)
    orders = [r['order_s'].split() for r in rows]
    # learned orders on shuffled copies of each stretch (optimiser/frequency control)
    stsh = []
    for sid, ent in st:
        e = ent[:]; rng.shuffle(e); stsh.append((sid, e))
    rows_sh = run_engine(stsh, sym, key, R=0, restarts=6, ils=25, nrand=0, seed=4, tag='c2s' + cname + kind + key)
    orders_sh = [r['order_s'].split() for r in rows_sh]
    real = cross(cname, kind, key, st, orders, rng)
    null = cross(cname, kind, key, st, orders_sh, rng)  # shuffled-learned orders predicting real stretches
    res = {'real': real, 'null_orders': null, 'n': len(st)}
    json.dump(res, open(fn + '.tmp', 'w')); os.replace(fn + '.tmp', fn)
    return fn


if __name__ == '__main__':
    jobs = [(c, k, key) for c in ['PSHARED', 'PLANT', 'LatX', 'LatXVI', 'LatXVII', 'ZL', 'IT']
            for k in ['pagewords', 'lineinit', 'linefinal', 'parainit', 'labels'] for key in ['F', 'L', 'R']]
    with Pool(2) as P:
        for fn in P.imap_unordered(job, jobs):
            print('done', os.path.basename(fn), flush=True)
