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


def components(groups, thr, cap=40):
    """greedy dossiers: best significant group first; a later significant group joins a dossier when at
    least half of its members are already in that dossier and none is in another (no chaining)."""
    sig = [g for g in groups if g['h'] > thr.get(C.size_class(len(g['G'])), 1e9)]
    sig.sort(key=lambda g: -g['h'])
    owner, dos = {}, []
    for g in sig:
        own = Counter(owner[x] for x in g['G'] if x in owner)
        if not own:
            dos.append({'ids': set(g['G']), 'best_h': g['h'], 'tm': g['tm']})
            for x in g['G']:
                owner[x] = len(dos) - 1
        elif len(own) == 1:
            k, c = own.most_common(1)[0]
            new = [x for x in g['G'] if x not in owner]
            if c >= len(g['G']) / 2 and len(dos[k]['ids']) + len(new) <= cap:
                dos[k]['ids'] |= set(new)
                for x in new:
                    owner[x] = k
    out = [{'ids': sorted(d['ids']), 'best_h': d['best_h'], 'tm': d['tm']} for d in dos]
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
        T = [t for t in T if len(t['lines']) >= 3]
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
    for s in (1, 2, 3, 4, 5):
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
