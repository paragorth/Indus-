"""pe28 cycle 3: LEAVE-ONE-TABLET-OUT propagation with distinct-line counting and stricter nulls.
For every PE tablet: fit the crossword solver (anchors + propagation, minv 3) on all OTHER tablets, then
predict that tablet's target lines (windows whose keys are all fixed, >= 1 new key).  Each line counts once.
Per key: held-out hit lines, distinct tablets, distinct counts x among hits (a rate must hold for different
counts; a constant line pair is a formula, not a rate).
Nulls on the fixed LOO predictions (1000x each): (i) global: values re-dealt within (system, final sign);
(ii) scale: the same but only among tablets in the same value-scale bin (log2 of the tablet's median value),
which keeps tablet-level number size; (iii) per key, the same nulls restricted to that key's lines;
search correction = max over keys of the per-key excess score.  Ur III control the same way (300 tablets)."""
import os, sys, json, time
import numpy as np
from math import log2
from collections import Counter, defaultdict
from fractions import Fraction as Fr
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe28_common import *  # noqa

NR = int(os.environ.get('NR', 1000))
t0 = time.time()
G = {}


def _fit(t):
    E, A, gs = G['E'], G['A'], G['gs']
    fx, inf = propagate([e for e in E if e['t'] != t], A, minv=3, gsys=gs)
    out = []
    for i, e in enumerate(E):
        if e['t'] != t:
            continue
        preds = []
        for w in e['W']:
            if all(k in fx for k in w) and any(k not in A for k in w):
                preds.append((sum(fx[k] * x for k, x in w.items()), tuple(sorted((k for k in w if k not in A))),
                              tuple(sorted(w.values()))))
        if preds:
            out.append((i, preds))
    return out


def loo(E, ntab, A, gs=None):
    G.update(E=E, A=A, gs=gs)
    with Pool(2) as p:
        R = p.map(_fit, range(ntab), chunksize=8)
    return [r for rr in R for r in rr]


def score(P, ys):
    hit_lines, per = 0, defaultdict(int)
    for i, preds in P:
        h = [p for p in preds if p[0] == ys[i]]
        if h:
            hit_lines += 1
            for k in h[0][1]:
                per[k] += 1
    return hit_lines, per


def analyse(E, P, label, scale_of):
    ys = [e['y'] for e in E]
    real, per = score(P, ys)
    det = defaultdict(list)
    for i, preds in P:
        h = [p for p in preds if p[0] == ys[i]]
        if h:
            for k in h[0][1]:
                det[k].append((E[i]['tid'], E[i]['line'], str(ys[i]), str(h[0][2])))
    rng = np.random.default_rng(7)
    pools = {}
    for mode in ('global', 'scale'):
        pl = defaultdict(list)
        for i, e in enumerate(E):
            key = (e['sys'], e['fin']) if mode == 'global' else (e['sys'], e['fin'], scale_of[e['t']])
            pl[key].append(i)
        pools[mode] = [np.array(v) for v in pl.values() if len(v) > 1]
    res = {'scored_lines': len(P), 'hits': real}
    for mode in ('global', 'scale'):
        nh = np.zeros(NR)
        nper = defaultdict(lambda: np.zeros(NR))
        for r in range(NR):
            yy = list(ys)
            for idx in pools[mode]:
                perm = rng.permutation(idx)
                for a, b in zip(idx, perm):
                    yy[a] = ys[b]
            h, pk = score(P, yy)
            nh[r] = h
            for k, c in pk.items():
                nper[k][r] = c
        res[mode] = {'null_mean': float(nh.mean()), 'p': float((1 + (nh >= real).sum()) / (1 + NR))}
        # per-key p and search-corrected p (max over keys of -log p in each null draw, approximated by
        # the count of keys whose null draw beats their real count)
        keys = sorted(per, key=lambda k: -per[k])
        rows = []
        for k in keys:
            arr = nper[k] if k in nper else np.zeros(NR)
            pk = float((1 + (arr >= per[k]).sum()) / (1 + NR))
            rows.append({'key': keystr(k), 'hits': per[k], 'null_mean': round(float(arr.mean()), 2), 'p': pk})
        # search correction: per draw, min over keys of the per-key tail prob of that draw's count
        allk = list(set(per) | set(nper))
        M = np.array([nper[k] if k in nper else np.zeros(NR) for k in allk])
        tail = np.zeros_like(M)
        for a in range(len(allk)):
            srt = np.sort(M[a])
            tail[a] = 1 - np.searchsorted(srt, M[a], side='left') / NR   # P(null >= value)
        mn = tail.min(0)
        for row, k in zip(rows, keys):
            row['p_corr'] = float((1 + (mn <= row['p']).sum()) / (1 + NR))
        res[mode]['keys'] = rows[:30]
    res['detail'] = {keystr(k): v for k, v in det.items()}
    res['distinct'] = {keystr(k): {'lines': len(v), 'tablets': len({d[0] for d in v}), 'counts': sorted({d[3] for d in v})}
                       for k, v in det.items()}
    print(label, res['scored_lines'], real, res['global']['null_mean'], res['global']['p'], res['scale']['null_mean'],
          res['scale']['p'], round(time.time() - t0), flush=True)
    for row in res['scale']['keys'][:12]:
        k = row['key']
        g = next(r for r in res['global']['keys'] if r['key'] == k) if any(r['key'] == k for r in res['global']['keys']) else {}
        print('   ', row, 'glob', g.get('null_mean'), g.get('p_corr'), res['distinct'][k], flush=True)
    return res


def scales(seqs):
    return [int(log2(max(float(np.median([float(q[2]) for q in s[1]])), 1))) for s in seqs]


if __name__ == '__main__':
    out = {}
    US = ur_seqs()
    rng = np.random.default_rng(0)
    idx = rng.choice(len(US), 300, replace=False)
    USs = [US[i] for i in idx]
    EU = build(USs, ur=True)
    PU = loo(EU, len(USs), {('gurusz', 'sze-bi', 'UR'): Fr(60)}, 'UR')
    out['ur3'] = analyse(EU, PU, 'UR3', scales(USs))
    S = pe_seqs()
    E = build(S)
    A = anchors_pe()
    P = loo(E, len(S), A)
    out['pe'] = analyse(E, P, 'PE', scales(S))
    json.dump(out, open(os.path.join(CK, 'cycle3.json'), 'w'), indent=1, default=str)
    print('done', round(time.time() - t0))
