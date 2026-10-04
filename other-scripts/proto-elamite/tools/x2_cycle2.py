#!/usr/bin/env python3
"""X-2 cycle 2: structure-aware alignment (similarity flooding, joint SVD) with bootstrap stability,
calibrated on Linear B <-> Ur III, then applied to Linear A <-> Proto-Elamite.

A 'world' = one pair of corpora (or samples). Inside a world, BT bootstrap resamples of the
documents on both sides; per method, each A commodity's top B partner is recorded, giving a
modal partner and its stability (share of resamples), and a mean similarity matrix.
Worlds:
  ctl_lape_k  : LB 352 docs vs Ur III 1,510 docs (LA x PE sizes), k = 0..W-1, real
  ctl_lape_sk : same, shuffled (commodity labels and designation tokens permuted)
  ctl_pe_k / ctl_pe_sk : LB 1,510 vs Ur III 1,510
  lape        : Linear A tablets vs Proto-Elamite tablets (BT_MAIN resamples)
  lape_sk     : shuffled Linear A vs shuffled Proto-Elamite, k = 0..WS-1
Checkpoint: data/x2/c2_worlds.jsonl. 2 workers.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json, os, sys, time, zlib
from collections import Counter, defaultdict
from multiprocessing import Pool
import numpy as np
import x2_common as X
import x2_fast as F

W = int(os.environ.get('X2_W', 30))
WS = int(os.environ.get('X2_WS', 60))
BT = int(os.environ.get('X2_BT', 40))
BT_MAIN = int(os.environ.get('X2_BTMAIN', 400))
CK = os.path.join(X.DX, 'c2_worlds.jsonl')
G = json.load(open(X.GOLD_FILE))
_D = {}


def data(n):
    if n not in _D:
        _D[n] = X.load(n)
    return _D[n]


def world(A, B, bt, rng, keep_sim=True):
    """Fast path (x2_fast): per-document counts precomputed once; resample = weighted sum."""
    pa, pb = F.precompute(A), F.precompute(B)
    ga = {it: i for i, it in enumerate(pa['items'])}; gb = {it: i for i, it in enumerate(pb['items'])}
    res = {m: {'top': defaultdict(Counter), 'wtop': defaultdict(Counter), 'btop': defaultdict(Counter),
               'sim': np.zeros((pa['nI'], pb['nI'])), 'cnt': np.zeros((pa['nI'], pb['nI']))} for m in X.METHODS2}
    for b in range(bt):
        wa = F.boot_w(len(A), rng) if b else np.ones(len(A))
        wb = F.boot_w(len(B), rng) if b else np.ones(len(B))
        itA, itB, out = F.run_w(pa, wa, pb, wb)
        ia = np.array([ga[x] for x in itA]); ib = np.array([gb[x] for x in itB])
        same = np.array([[x[0] == y[0] for y in itB] for x in itA])
        for m, S in out.items():
            r = res[m]
            for a, (t, mg) in X.top_partner(itA, itB, S, 'c:').items():
                r['top'][a][t] += 1
            for a, (t, mg) in X.top_partner(itA, itB, S, 'w:').items():
                r['wtop'][a][t] += 1
            for a, (t, mg) in X.top_partner(itB, itA, S.T, 'c:').items():
                r['btop'][a][t] += 1
            if keep_sim:
                r['sim'][np.ix_(ia, ib)] += np.where(same, S, 0)
                r['cnt'][np.ix_(ia, ib)] += same
    out = {}
    for m, r in res.items():
        o = {'top': {a: dict(c) for a, c in r['top'].items()}, 'wtop': {a: dict(c.most_common(3)) for a, c in r['wtop'].items()},
             'btop': {a: dict(c) for a, c in r['btop'].items()}, 'bt': bt}
        if keep_sim:
            ii, jj = np.nonzero(r['cnt'] >= bt / 2)
            o['msim'] = {pa['items'][i] + '|' + pb['items'][j]: float(r['sim'][i, j] / r['cnt'][i, j]) for i, j in zip(ii, jj)}
        out[m] = o
    return out


def mrr_from_msim(msim, gold, pre='c:'):
    A = sorted({k.split('|')[0] for k in msim if k.startswith(pre)})
    Bs = sorted({k.split('|')[1] for k in msim if k.startswith(pre)})
    lab = [b[2:] for b in Bs]
    rr = []
    for s, tg in gold.items():
        if pre + s not in A or not any(x in lab for x in tg):
            continue
        row = np.array([msim.get(pre + s + '|' + b, -1e9) for b in Bs])
        o = np.argsort(-row)
        rank = next(r for r, j in enumerate(o) if lab[j] in tg) + 1
        rr.append(1 / rank)
    return (float(np.mean(rr)), len(rr), len(Bs)) if rr else None


def task(args):
    name, k = args
    rng = np.random.default_rng(zlib.crc32(("%s|%d" % (name, k)).encode()))
    if name.startswith('ctl'):
        LB, UR = data('LB'), data('UR3')
        na, nb = (352, 1510) if '_lape' in name else (1510, 1510)
        A = [LB[i] for i in rng.choice(len(LB), na, replace=False)]
        B = [UR[i] for i in rng.choice(len(UR), nb, replace=False)]
        bt = BT
    else:
        A, B = data('LA'), data('PE')
        bt = BT_MAIN if name == 'lape' else BT
    if name.endswith('_s'):
        A = X.shuffle_all(A, rng); B = X.shuffle_all(B, rng)
    out = world(A, B, bt, rng, keep_sim=True)
    for m in out:
        if name.startswith('ctl'):
            out[m]['gold_c'] = mrr_from_msim(out[m]['msim'], G['gold'])
            out[m]['gold_w'] = mrr_from_msim(out[m]['msim'], G['gold_words'], 'w:')
            out[m]['msim'] = {kk: v for kk, v in out[m]['msim'].items() if kk.startswith('c:')}
        elif name != 'lape':
            out[m].pop('msim')
    return {'name': name, 'k': k, 'res': out}


def main():
    done = set()
    if os.path.exists(CK):
        for l in open(CK):
            d = json.loads(l); done.add((d['name'], d['k']))
    tasks = [('lape', 0)]
    for k in range(max(W, WS)):
        if k < W:
            tasks += [('ctl_lape', k), ('ctl_lape_s', k), ('ctl_pe', k), ('ctl_pe_s', k)]
        if k < WS:
            tasks += [('lape_s', k)]
    tasks = [t for t in tasks if t not in done]
    print('tasks', len(tasks), file=sys.stderr)
    t0 = time.time()
    with Pool(2) as P, open(CK, 'a') as f:
        for i, r in enumerate(P.imap_unordered(task, tasks)):
            f.write(json.dumps(r) + '\n'); f.flush()
            print(i, r['name'], r['k'], round(time.time() - t0), file=sys.stderr)


if __name__ == '__main__':
    main()
