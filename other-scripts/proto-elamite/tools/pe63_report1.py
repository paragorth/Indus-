#!/usr/bin/env python3
"""pe63 cycle 1 report: null thresholds, dossiers (components of significant groups), controls."""
import os, json, random, glob
from collections import defaultdict, Counter
import numpy as np
import pe63_common as C


def load(c, m, s):
    fn = os.path.join(C.CK, 'c1_%s_%s_%d.json' % (c, m, s))
    return json.load(open(fn)) if os.path.exists(fn) else None


def thresholds(corp):
    thr = defaultdict(float)
    q = defaultdict(list)
    for fn in glob.glob(os.path.join(C.CK, 'c1_%s_null_*.json' % corp)):
        R = json.load(open(fn))
        mx = defaultdict(float)
        for g in R['groups']:
            sc = C.size_class(len(g['G']))
            mx[sc] = max(mx[sc], g['h'])
        for k, v in mx.items():
            thr[k] = max(thr[k], v)
            q[k].append(v)
    return thr


def components(groups, thr):
    sig = [g for g in groups if g['h'] > thr.get(C.size_class(len(g['G'])), 1.0)]
    par = {}

    def f(x):
        par.setdefault(x, x)
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    for g in sig:
        for x in g['G'][1:]:
            par[f(x)] = f(g['G'][0])
    comp = defaultdict(set)
    for x in list(par):
        comp[f(x)].add(x)
    best = defaultdict(float)
    tm = {}
    for g in sig:
        r = f(g['G'][0])
        if g['h'] > best[r]:
            best[r] = g['h']; tm[r] = g['tm']
    out = [{'ids': sorted(v), 'best_h': best[r], 'tm': tm[r]} for r, v in comp.items()]
    out.sort(key=lambda d: (-len(d['ids']), -d['best_h']))
    return sig, out


def matched_z(T, ix, ids, rng, R=500):
    idx = {t['id']: i for i, t in enumerate(T)}
    G = [idx[i] for i in ids]
    m = len(G)
    real = ix.S[np.ix_(G, G)].sum() / (m * (m - 1))
    byL = defaultdict(list)
    for i, t in enumerate(T):
        byL[min(len(t['lines']), 15)].append(i)
    vals = []
    for _ in range(R):
        H = [rng.choice(byL[min(len(T[g]['lines']), 15)]) for g in G]
        if len(set(H)) < m:
            continue
        vals.append(ix.S[np.ix_(H, H)].sum() / (m * (m - 1)))
    vals = np.array(vals)
    return real, float((real - vals.mean()) / (vals.std() + 1e-9)), float((vals >= real).mean())


def ur3_dates(t):
    mo = yr = None
    for l in t['lines']:
        if l['s'] and l['s'][0] == 'iti':
            mo = ' '.join(l['s'][1:])
        if l['s'] and l['s'][0] == 'mu':
            yr = ' '.join(l['s'][1:])[:40]
    return mo, yr


def main():
    rows = []
    summ = {}
    rng = random.Random(63)
    for corp in ['PE', 'DR', 'UM']:
        R = load(corp, 'real', 0)
        if not R:
            continue
        T = {'PE': C.pe_tabs, 'DR': lambda: C.ur3_tabs('Puzr', 1500, 'DR'),
             'UM': lambda: C.ur3_tabs('Umma', 1500, 'UM')}[corp]()
        ix = C.Index(T)
        thr = thresholds(corp)
        sig, comps = components(R['groups'], thr)
        byid = {t['id']: t for t in T}
        zs = []
        for d in comps:
            d['S'], d['z'], d['p'] = matched_z(T, ix, d['ids'], rng)
            zs.append(d['z'])
        # null: how many significant groups does each null run itself have against the others' max?
        info = {'n_groups': R['n_groups'], 'thr': {k: round(v, 3) for k, v in sorted(thr.items())},
                'n_sig': len(sig), 'n_dossiers': len(comps),
                'n_tabs_in': sum(len(d['ids']) for d in comps),
                'sizes': Counter(len(d['ids']) for d in comps).most_common()}
        if corp != 'PE':
            same_m = same_y = pairs = 0
            for d in comps:
                for a in range(len(d['ids'])):
                    for b in range(a + 1, len(d['ids'])):
                        ma, ya = ur3_dates(byid[d['ids'][a]])
                        mb, yb = ur3_dates(byid[d['ids'][b]])
                        pairs += 1
                        same_m += (ma is not None and ma == mb)
                        same_y += (ya is not None and ya == yb)
            rm = ry = 0
            ids = list(byid)
            for _ in range(20000):
                a, b = rng.sample(ids, 2)
                ma, ya = ur3_dates(byid[a]); mb, yb = ur3_dates(byid[b])
                rm += (ma is not None and ma == mb); ry += (ya is not None and ya == yb)
            info.update({'pairs': pairs, 'same_month': same_m / max(pairs, 1), 'same_year': same_y / max(pairs, 1),
                         'rand_month': rm / 20000, 'rand_year': ry / 20000})
        summ[corp] = {'info': info, 'dossiers': comps}
    # planted
    pl = []
    for s in (1, 2, 3):
        R = load('PE', 'plant', s)
        if not R:
            continue
        thr = thresholds('PE')
        sig, comps = components(R['groups'], thr)
        tr = set(R['truth']['ids'])
        bestc = max(comps, key=lambda d: len(tr & set(d['ids']))) if comps else {'ids': []}
        pl.append({'seed': s, 'recovered': len(tr & set(bestc['ids'])), 'of': len(tr),
                   'extra': len(set(bestc['ids']) - tr), 'truth': R['truth']})
    summ['plant'] = pl
    json.dump(summ, open(os.path.join(C.CK, 'c1_summary.json'), 'w'), indent=1)
    for corp in ['PE', 'DR', 'UM']:
        if corp in summ:
            print(corp, json.dumps(summ[corp]['info']))
            for d in summ[corp]['dossiers'][:40]:
                print('   ', len(d['ids']), round(d['best_h'], 3), 'z%.1f' % d['z'], d['ids'][:14], d['tm'])
    print('plant', pl)


if __name__ == '__main__':
    main()
