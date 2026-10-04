#!/usr/bin/env python3
"""LA-39 cycle 3.
(a) Power of the word-context test (cycle 2 passes on LB sex markers): planted modifier-specific
    words on LA (a fraction q of each shared part's documents gets one invented word per part), and
    Linear B sex markers down-sampled to Linear A size.
(b) Effect profile of each LA added part (>= 4 tokens): type mean minus the base's other types,
    document bootstrap (1,000); which features move beyond chance (Holm).  Meaning classes.
(c) Measure-letter adjuncts (fraction-like letters B D E F H K L QIf written inside a logogram) vs
    syllabic adjuncts: smaller quantities relative to the base?  Null: class labels permuted among
    ligature types (10,000).
"""
import sys, os, json
import numpy as np
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la39_common as L
import la39_c2 as C2

rng = np.random.default_rng(393)


def ctx_plant(A, DW, q, reps=10, nperm=300):
    sh = L.shared_mods(A); hits = []
    for r in range(reps):
        DW2 = {k: set(v) for k, v in DW.items()}
        for m in sh:
            docs = sorted(set(A['doc'][A['mod'] == m]))
            for d in docs:
                if rng.random() < q: DW2.setdefault(d, set()).add('PLANT-%s-%d' % (m, r))
        C2._SIMS.clear()
        B = dict(A)   # new id -> fresh cache
        o = C2.context_stat(B, DW2, B['mod'], {})['all'][0]
        nv = [C2.context_stat(B, DW2, L.shuffle_mods(B, rng), {})['all'][0] for _ in range(nperm)]
        hits.append((1 + sum(v >= o for v in nv)) / (1 + nperm) <= 0.05)
    return int(sum(hits)), reps


def lb_downsample(LB, DW, n_tok, reps=10, nperm=300):
    G = {'sex': {'m', 'f', 'x'}}
    ud = np.unique(LB['doc']); hits = defaultdict(int)
    for r in range(reps):
        order = rng.permutation(ud); keep = set(); tot = 0
        cnt = defaultdict(int)
        for d in LB['doc']: cnt[d] += 1
        for d in order:
            if tot >= n_tok: break
            keep.add(d); tot += cnt[d]
        C2._SIMS.clear()
        k = np.array([d in keep for d in LB['doc']])
        B = {kk: (v[k] if isinstance(v, np.ndarray) else v) for kk, v in LB.items()}
        o = C2.context_stat(B, DW, B['mod'], G)
        nulls = defaultdict(list)
        for _ in range(nperm):
            for kk, v in C2.context_stat(B, DW, L.shuffle_mods(B, rng), G).items(): nulls[kk].append(v[0])
        for kk, (v, n) in o.items():
            hits[kk] += (1 + sum(x >= v for x in nulls[kk])) / (1 + nperm) <= 0.05
    return dict(hits), reps


def profiles(A, min_tok=4, nboot=1000):
    types = defaultdict(int)
    for b, m in zip(A['base'], A['mod']):
        if m: types[(b, m)] += 1
    ud = np.unique(A['doc']); didx = defaultdict(list)
    for i, d in enumerate(A['doc']): didx[d].append(i)
    out = {}
    for (b, m), n in sorted(types.items(), key=lambda x: -x[1]):
        if n < min_tok: continue
        kb = A['base'] == b
        if (kb & (A['mod'] != m)).sum() < 3: continue

        def eff(idx):
            X, bb, mm = A['X'][idx], A['base'][idx], A['mod'][idx]
            t = (bb == b) & (mm == m); o = (bb == b) & (mm != m)
            if t.sum() == 0 or o.sum() == 0: return None
            return X[t].mean(0) - X[o].mean(0)
        e0 = eff(np.arange(len(A['X'])))
        bs = []
        for _ in range(nboot):
            idx = np.concatenate([didx[d] for d in rng.choice(ud, len(ud))])
            e = eff(idx)
            if e is not None: bs.append(e)
        bs = np.array(bs)
        p = np.minimum(1, 2 * np.minimum((bs <= 0).mean(0), (bs >= 0).mean(0)) + 1 / len(bs))
        out['%s+%s' % (b, m)] = dict(n=n, eff=np.round(e0, 2).tolist(), p=np.round(p, 4).tolist())
    # Holm over all type x feature tests
    allp = sorted([(v['p'][f], k, f) for k, v in out.items() for f in range(len(v['p']))])
    M = len(allp); sig = []
    for r, (p, k, f) in enumerate(allp):
        if p * (M - r) > 0.05: break
        sig.append((k, A['feats'][f], out[k]['eff'][f], p))
    return out, sig, M


MEASURE = {'B', 'D', 'E', 'F', 'H', 'K', 'L', 'L2', 'L4', 'QIf', 'K+L', 'L4+L4'}


def measure_test(A, rows_raw, nperm=10000):
    """Relative quantity (log2 q of the ligature minus mean log2 q of the base's plain/other types),
    measure-letter adjunct types vs syllabic adjunct types; token-level on numbered tokens only."""
    X = A['X']; f = A['feats'].index('logq'); fn = A['feats'].index('noq')
    num = X[:, fn] < X[:, fn].max()   # tokens with a number
    raw = np.array([r[3][0] for r in rows_raw])  # unstandardised log2(1+q)
    types = sorted(set((b, m) for b, m in zip(A['base'], A['mod']) if m))
    rel = {}
    for b, m in types:
        t = (A['base'] == b) & (A['mod'] == m) & num
        o = (A['base'] == b) & (A['mod'] != m) & num
        if t.sum() and o.sum() >= 2: rel[(b, m)] = (raw[t].mean() - raw[o].mean(), int(t.sum()))
    keys = list(rel); vals = np.array([rel[k][0] for k in keys]); w = np.array([rel[k][1] for k in keys])
    lab = np.array([k[1] in MEASURE for k in keys])
    def stat(l): return np.average(vals[l], weights=w[l]) - np.average(vals[~l], weights=w[~l])
    o = stat(lab); nv = np.array([stat(rng.permutation(lab)) for _ in range(nperm)])
    p = (1 + (nv <= o).sum()) / (1 + nperm)
    detail = sorted([('%s+%s' % k, round(rel[k][0], 2), rel[k][1], k[1] in MEASURE) for k in keys], key=lambda x: x[1])
    return dict(diff=round(float(o), 3), null=round(float(nv.mean()), 3), P_lower=round(float(p), 4),
                n_measure=int(lab.sum()), n_syll=int((~lab).sum()), detail=detail)


if __name__ == '__main__':
    R = {}
    rows = L.la_commodity_rows(); LAc = L.to_arrays(rows)
    LBr = L.lb_rows(); LB = L.to_arrays(LBr)
    DWa, DWb = C2.doc_words_la(), C2.doc_words_lb()
    R['ctx_plant'] = {q: ctx_plant(LAc, DWa, q) for q in (0.3, 0.6, 1.0)}
    print('context plant (hits/reps)', R['ctx_plant'], flush=True)
    R['lb_down'] = lb_downsample(LB, DWb, len(LAc['X']))
    print('LB sex context at LA size', R['lb_down'], flush=True)
    prof, sig, M = profiles(LAc)
    R['profiles'] = prof; R['sig'] = sig
    print('profiles', len(prof), 'types; Holm-significant of', M, ':', sig, flush=True)
    R['measure'] = measure_test(LAc, rows)
    print('measure-letter vs syllabic', {k: v for k, v in R['measure'].items() if k != 'detail'}, flush=True)
    print(R['measure']['detail'], flush=True)
    json.dump(R, open(os.path.join(L.CK, 'c3.json'), 'w'), indent=1, default=str)
